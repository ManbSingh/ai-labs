from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "RAG is powerful",
    "I love machine learning",
    "Dogs are cute"
]

embeddings = model.encode(sentences)

print(embeddings.shape)