import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
import dspy
import streamlit_scrollable_textbox as stx

#### DSPy
ollama_model = dspy.OllamaLocal(model='llama3',model_type='chat')

#gpt4 = dspy.OpenAI(model="gpt-4-turbo-2024-04-09", max_tokens=1000, model_type="chat")

dspy.settings.configure(lm=ollama_model)

#class BasicQA(dspy.Signature):
 #   """Document summarifying based on llama-3 and DSPy"""
#
 #   document = dspy.InputField(desc="A legal opinion document, a written explanation by a judge that accompanies an order or ruling in a case")
  #  summary = dspy.OutputField(desc="Summarization of legal opinion document")

# generate_answer = dspy.Predict(BasicQA)

# summarize = dspy.ChainOfThought('document -> summary')

class Summarizer(dspy.Signature):
    """Document summarifying based on llama-3 and DSPy"""
    document = dspy.InputField(desc="A legal opinion document, a written explanation by a judge that accompanies an order or ruling in a case")
    summary = dspy.OutputField(desc="Summarization of legal opinion document")

summarize = dspy.Predict(Summarizer)

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
    response = summarize(document = result[0].page_content)
    st.write(response.summary)
    print(response.summary)
    #stx.scrollableTextbox(result[1].page_content)
    #st.write("source: ", result[1].metadata, )
