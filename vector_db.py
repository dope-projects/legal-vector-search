import os
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

loader = JSONLoader(file_path = "opinions_flattened_2k.json", jq_schema = '.messages[].plain_text')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
print('connection successful')

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)

print(docs)