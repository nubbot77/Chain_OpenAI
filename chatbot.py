from constants import openai_api
import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"]= openai_api

#streamlit framework
st.title('Langchain Demo using openai api')
input_text = st.text_input("Search the topic u want")

## Prompt templates
first_input_prompt = PromptTemplate(
    input_variables=["topic"],
    template = "tell me about Large language model {topic}",

)
second_input_prompt = PromptTemplate(
    input_variables=["model"],
    template = "when was the {model} released",
)
third_input_prompt = PromptTemplate(
    input_variables=["released_date"],
    template = "Mention 5 major events happened around {released_date} in world",
)

## memory

model_memory= ConversationBufferMemory(input_key='topic',memory_key='chat_history')
release_date_memory= ConversationBufferMemory(input_key='model',memory_key='chat_history')
descr_memory= ConversationBufferMemory(input_key='released_date',memory_key='chat_history')

#openai llms

llm = OpenAI(temperature=0.8)

##chaining llms

chain1= LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='model',memory=model_memory)

chain2= LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='released_date',memory=release_date_memory)
chain3= LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='events',memory=descr_memory)

join_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=["topic"],output_variables=["model","released_date","events"],verbose=True)

if input_text:
    st.write(join_chain({"topic":input_text}))

    with st.expander('Topic Name'):
        st.info(model_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)