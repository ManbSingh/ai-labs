# Create Embeddings + Store in FAISS

# We did 3 things:

# Load model
# Create embeddings
# Store in FAISS
# Search
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Your sample data (simulate citizenship content)
documents = [
    "Canada is a constitutional monarchy",
    "The Prime Minister leads the government",
    "Ottawa is the capital of Canada",
    "Citizens have rights and responsibilities"
]

# 3. Convert text → embeddings
embeddings = model.encode(documents)
print(embeddings.shape)

# Convert to numpy array (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# 4. Create FAISS index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)

# 5. Add embeddings to index
index.add(embeddings)

print(f"Stored {index.ntotal} documents")

# -------------------------------
# 🔎 SEARCH PART
# -------------------------------

query = "Who is the leader of canada?"

# Convert query → embedding
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Search top 2 results
k = 2
distances, indices = index.search(query_embedding, k)

print("\nTop matches:")
for i in indices[0]:
    print("-", documents[i])