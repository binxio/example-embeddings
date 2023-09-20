import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load the FAISS index
loaded_index = faiss.read_index("index.idx")

# Deserialize the mapping from the pickle file
with open("index.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Now you can use loaded_index and loaded_data in your search functionality
query_text = "Beatiful colors"
model = SentenceTransformer('distilbert-base-nli-mean-tokens')  # You'd still need the embedding model for new queries
query_embedding = model.encode(query_text)

k = 5
distances, indices = loaded_index.search(query_embedding.reshape(1, -1), k)

print("Indices of closest texts:", indices[0])
print("Distances to closest texts:", distances[0])

for idx in indices[0]:
    print("Text:", loaded_data[idx]["text"])
    print("Image:", loaded_data[idx]["image"])
