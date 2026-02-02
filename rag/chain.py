from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm.ollama_llm import load_llm

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
db = FAISS.load_local(
    Path("data/index/faiss"),
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Ollama LLM
llm = load_llm()

def ask(question: str) -> str:
    docs = db.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Bạn là trợ lý hỗ trợ thông tin bệnh hại lúa.
Chỉ trả lời dựa trên tài liệu được cung cấp.
Không suy đoán, không thêm kiến thức ngoài.

Tài liệu:
{context}

Câu hỏi:
{question}

Trả lời ngắn gọn, rõ ràng:
"""
    return llm.invoke(prompt)
