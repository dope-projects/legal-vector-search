import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
import dspy
import re
import streamlit_scrollable_textbox as stx
from summarizer import Summarizer
#### DSPy

ollama_model = dspy.OllamaLocal(model='llama3',model_type='chat')

dspy.settings.configure(lm=ollama_model)

class Summarizer(dspy.Signature):
    """Document summarifying based on llama-3 and DSPy"""
    document = dspy.InputField(desc="A legal opinion document, a written explanation by a judge that accompanies an order or ruling in a case")
    summary = dspy.OutputField(desc="Summarization of legal opinion document")

summarize = dspy.Predict(Summarizer)


## STREAMLIT + MILVUS

st.title('Vector Search over Legal Database')

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="LangChainCollection",
)



instr = 'Ignite ideas..!🔥'

with st.form('chat_input_form'):
    # Create two columns; adjust the ratio to your liking
    col1, col2 = st.columns([3,1]) 

    # Use the first column for text input
    with col1:
        prompt = st.text_input(
            instr,
            value=instr,
            placeholder=instr,
            label_visibility='collapsed'
        )
    # Use the second column for the submit button
    with col2:
        submitted = st.form_submit_button('🔍')
    
    if prompt and submitted:
        result = vector_db.similarity_search(prompt)
        
        response = summarize(document = result[0].page_content)
        st.write("TLDR: ", response.summary)

        st.write(result[0].page_content)
        st.write(result[0].metadata)
        
        print(response.summary)
    #stx.scrollableTextbox(result[1].page_content)
    #st.write("source: ", result[1].metadata, )
