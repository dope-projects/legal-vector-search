import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
import dspy
import streamlit_scrollable_textbox as stx


ollama_model = dspy.OpenAI(api_base='http://localhost:11434/v1/', api_key='ollama', model='llama3', stop='\n\n', model_type='chat')

# This sets the language model for DSPy.
dspy.settings.configure(lm=ollama_model)

st.title('Vector Search over Legal Database')

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="LangChainCollection",
)

query = st.text_input("Search??")

if st.button("🔍"):
    result = vector_db.similarity_search(query)

    st.write(result[0].page_content)
    st.write("source: ", result[0].metadata)
    stx.scrollableTextbox(result[1].page_content)
    st.write("source: ", result[1].metadata)
