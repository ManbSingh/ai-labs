from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# 1. LOAD PDF
# -------------------------------
pdf_path = "discover_canada.pdf"  # put your PDF in same folder

reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text()

print(f"Loaded PDF with {len(text)} characters")

# -------------------------------
# 2. CHUNK TEXT
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

documents = chunk_text(text)

print(f"Created {len(documents)} chunks")

# -------------------------------
# 3. LOAD EMBEDDING MODEL
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# 4. CREATE EMBEDDINGS
# -------------------------------
embeddings = model.encode(documents, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print(f"Embeddings shape: {embeddings.shape}")

# -------------------------------
# 5. CREATE FAISS INDEX
# -------------------------------
dimension = embeddings.shape[1]  # should be 384
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

print(f"Stored {index.ntotal} chunks in FAISS")

# -------------------------------
# 6. SEARCH FUNCTION
# -------------------------------
def search(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    print("\n🔎 Query:", query)
    print("\nTop results:\n")

    for i in indices[0]:
        print(documents[i])
        print("-" * 50)

# -------------------------------
# 7. TEST SEARCH
# -------------------------------
search("what is mace of house of commons")
search("What is the capital of Canada?")