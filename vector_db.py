import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pymilvus import connections


load_dotenv()

loader = JSONLoader(file_path = "opinions_flattened_2k.json", jq_schema = '.[].plain_text')
documents = loader.load()

def clean_text(text):
    clean_text = text.replace("\n", "")
    clean_text = clean_text.replace("                                    ", ",")
    clean_text = clean_text.replace("                       ", ",")
    clean_text = clean_text.replace("            ", ",")
    clean_text = clean_text.replace("          ", ",")
    clean_text = clean_text.replace("       ", ",")
    return clean_text

for document in documents:
    document.page_content = clean_text(document.page_content)


filtered_list = [documents[i] for i in \
                 range(len(documents)) if (len(documents[i].page_content) < 12000)]


text_splitter = CharacterTextSplitter(chunk_size=3072, chunk_overlap=256)
docs = text_splitter.split_documents(filtered_list)

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")


vector_db = Milvus.from_documents(
    filtered_list,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
print('vector db initalized successfuly!')


example = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(example)

print(docs)

connections.connect(
  alias="default", 
  host='localhost', 
  port='19530'
)

print("connected to the vector db!")

from pymilvus import utility
print("Milvus collection list:", utility.list_collections())
