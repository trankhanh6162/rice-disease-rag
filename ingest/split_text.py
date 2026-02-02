from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

input_path = Path("data/processed/raw.txt")
output_path = Path("data/processed/chunks.txt")

with open(input_path, encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_text(text)

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c.strip() + "\n---\n")

print(f"Split into {len(chunks)} chunks")
