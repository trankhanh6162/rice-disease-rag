from fastapi import FastAPI
from pydantic import BaseModel
from rag.chain import ask

app = FastAPI(title="Rice Disease RAG API")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = ask(req.question)
    return {"answer": answer}
