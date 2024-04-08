from pymilvus import MilvusClient, connections
import json
from transformers import AutoTokenizer, AutoModel
import torch

# Establish a connection to the Milvus server
connections.connect(host='localhost', port='19530')
# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://localhost:19530"
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Function to vectorize a single sentence
def vectorize_text(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Pass the input through the model and extract the [CLS] token embedding
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for the [CLS] token at the beginning of the sequence
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding.tolist()  # Convert numpy array to list

# Example sentence to vectorize
sentence = "(Tex. Crim. App. 2015)"

# Vectorize the sentence
vectorized_sentence = vectorize_text(sentence)




# 6. Search with a single vector
# 6.1. Prepare query vectors

print(vectorized_sentence)
# 6.2. Start search
res = client.search(
    collection_name="llm_law_hackathon_3",     # target collection
    data=vectorized_sentence,                # query vectors
    limit=3,                           # number of returned entities
    anns_field="plain_text_vector"
)

print(res)