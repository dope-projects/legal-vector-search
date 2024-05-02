# Legal Vector Search application!📄

This Streamlit application allows users to search for legal opinions in a Milvus vector database and generate summaries using the Llama-3 language model and DSPy.
<p align="center">
<img width="552" alt="image" src="https://github.com/dope-projects/llm-law-hackathon/assets/63906053/69df01a8-697b-4272-93dc-2a609ea18211">
</p>

## Prerequisites

Before running the Streamlit application, ensure that the following prerequisites are met:

1. Milvus standalone is running.
2. Ollama llama-3 language model is running.
3. The vector database is initialized with the legal opinion documents.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/dope-projects/llm-law-hackathon.git
cd llm-law-hackathon
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Milvus standalone server:

```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
bash standalone_embed.sh start
```

2. Start the Llama-3 language model on ollama server.

3. Initialize the vector database with the legal opinion documents:

```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="LangChainCollection",
)

# Add your legal opinion documents to the vector database
documents = [...]  # List of legal opinion documents
vector_db.add_documents(documents)
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Configuration

- The Milvus connection settings can be modified in the `vector_db` initialization code block.
- The Ollama-3 language model settings can be adjusted in the `ollama_model` initialization code block.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Milvus](https://milvus.io/) - Vector database
- [Ollama-3](https://github.com/ollama-ai/ollama) - LLM inference
- [DSPy](https://github.com/microsoft/dspy) - LLM Orchestrator
- [Streamlit](https://streamlit.io/) - Frontend framework
