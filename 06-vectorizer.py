import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Load the JSON file containing the data
with open("opinions_flattened_2k.json", "r") as file:
    data = json.load(file)

# Define the text fields to be vectorized
text_fields = ["plain_text", "html_with_citations"]  # Add other text fields as needed

# Function to vectorize text and return a list for JSON serialization
def vectorize_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings for the [CLS] token at the beginning of the sequence
    vector = outputs.last_hidden_state[:,0,:].cpu().numpy()
    return vector.tolist()  # Convert numpy array to list

# Vectorize the text fields
for row in data:
    for field in text_fields:
        if field in row and row[field]:
            row[field + "_vector"] = vectorize_text(row[field])

# Save the updated data to a new JSON file
with open("opinions_flattened_2k_vectorized.json", "w") as file:
    json.dump(data, file)