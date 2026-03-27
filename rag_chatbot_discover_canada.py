import os
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# OPTIONAL: OpenAI (for final answer generation)
from openai import OpenAI

# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = "discover_canada.pdf"
INDEX_FILE = "faiss_index.bin"
DOC_FILE = "documents.pkl"

USE_OPENAI = True  # set False if you only want raw retrieval

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# STEP 1: BUILD OR LOAD INDEX
# -------------------------------
if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
    print("🔄 Loading existing FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)

else:
    print("📄 Building FAISS index from PDF...")

    # Load PDF
    reader = PdfReader(PDF_PATH)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    print(f"Loaded PDF with {len(text)} characters")

    # Chunking
    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    documents = chunk_text(text)
    print(f"Created {len(documents)} chunks")

    # Embeddings
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Stored {index.ntotal} chunks")

    # Save
    faiss.write_index(index, INDEX_FILE)

    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("💾 Index saved!")

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def retrieve(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]
    return results

# -------------------------------
# GENERATE ANSWER (LLM)
# -------------------------------
def generate_answer(query, context_chunks):
    print("\nRetrieved chunks sent to GPT:\n")
    for i, chunk in enumerate(context_chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
    if not USE_OPENAI:
        return "\n\n".join(context_chunks)

    client = OpenAI()  # requires OPENAI_API_KEY env variable

    context = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, respond: "I don't know."

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

# -------------------------------
# MAIN LOOP
# -------------------------------
print("\n🤖 RAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    query = input("❓ Ask: ")

    if query.lower() == "exit":
        break

    # Retrieve
    chunks = retrieve(query)

    # Generate
    answer = generate_answer(query, chunks)

    print("\n💬 Answer:\n", answer)
    print("\n" + "="*60 + "\n")