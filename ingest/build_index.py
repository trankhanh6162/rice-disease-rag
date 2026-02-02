from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

chunks_path = Path("data/processed/chunks.txt")
index_path = Path("data/index/faiss")

with open(chunks_path, encoding="utf-8") as f:
    docs = f.read().split("\n---\n")

docs = [d.strip() for d in docs if d.strip()]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(docs, embeddings)

index_path.parent.mkdir(parents=True, exist_ok=True)
db.save_local(index_path)

print(f"FAISS index built with {len(docs)} documents")
