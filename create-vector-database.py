# Script to create a vector database

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

data = [
    {"text": "The sky is blue.", "image": "sky.jpg"},
    {"text": "I love programming.", "image": "code.jpg"},
    {"text": "The quick brown fox jumps over the lazy dog.", "image": "fox.jpg"},
    # ... add more data as needed
]

# Initialize the transformer model for embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Extract texts and generate embeddings
texts = [item["text"] for item in data]
embeddings = model.encode(texts)

# Create the FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

faiss.write_index(index, "index.idx")

# Serialize the mapping (data) to a pickle file
with open("index.pkl", "wb") as f:
    pickle.dump(data, f)
