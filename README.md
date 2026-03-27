# ai-labs
<!--  A Canada citizenship test prep agent using RAG + all-MiniLM-L6-v2-->

<!-- Embedding model: -->
<!-- Converts text into vectors (numbers the computer can search). -->

<!-- Popular options: -->
<!-- OpenAI embeddings
Sentence Transformers (local, free)
We use sentence transformers in this setup (all-MiniLM-L6-v2) - Hugging Face -->

py -m pip install sentence-transformers

<!-- Vector DB: tool to store vectors and search efficiently -->
<!-- FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta -->
py -m pip install faiss-cpu 
