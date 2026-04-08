import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# OPTIONAL: OpenAI (for final answer generation)
from openai import OpenAI

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "manufacturing_sensors.csv"
INDEX_FILE = "faiss_index_predictive_maintenance_assistant.bin"
DOC_FILE = "documents4predictive_maintenance_assistant.pkl"

USE_OPENAI = True  # set False if you only want raw retrieval

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# ROW → TEXT FUNCTION
# -------------------------------
def row_to_text(row):
    status = "failure occurred" if row["failure"] == 1 else "normal operation"

    condition = []
    if row["temp_c"] > 80:
        condition.append("high temperature")
    if row["vibration"] > 50:
        condition.append("high vibration")

    condition_str = ", ".join(condition) if condition else "normal conditions"

    return (
        f"Machine {row['machine_id']} at {row['timestamp']} experienced {condition_str}. "
        f"Temperature: {row['temp_c']}°C, RPM: {row['rpm']}, vibration: {row['vibration']}. "
        f"Outcome: {status}."
    )

# -------------------------------
# STEP 1: BUILD OR LOAD INDEX
# -------------------------------
if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
    print("🔄 Loading existing FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)

else:
    print("📄 Building FAISS index from CSV...")

    df = pd.read_csv(CSV_PATH)

    # Convert rows to text
    documents = [row_to_text(row) for _, row in df.iterrows()]
    print(f"Converted {len(documents)} rows into text")

    # Embeddings
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Stored {index.ntotal} records")

    # Save
    faiss.write_index(index, INDEX_FILE)

    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("💾 Index saved!")

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def retrieve(query, k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]
    return results

# -------------------------------
# GENERATE ANSWER (LLM)
# -------------------------------
def generate_answer(query, context_chunks):
    print("\nRetrieved records sent to GPT:\n")
    for i, chunk in enumerate(context_chunks):
        print(f"--- Record {i+1} ---\n{chunk}\n")

    if not USE_OPENAI:
        return "\n\n".join(context_chunks)

    client = OpenAI()  # requires OPENAI_API_KEY env variable

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant for predictive maintenance in manufacturing.

Based ONLY on the historical machine records below:
{context}

Answer the user's question:
{query}

Provide:
- Likely issue
- Failure risk (High/Medium/Low)
- Recommended maintenance action

If unsure, say "I don't know".
Keep the answer clear and concise.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

# -------------------------------
# MAIN LOOP
# -------------------------------
print("\n🤖 Predictive Maintenance RAG Chatbot Ready! Type 'exit' to quit.\n")

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