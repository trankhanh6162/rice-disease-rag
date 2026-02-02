from langchain_community.llms import Ollama

def load_llm():
    return Ollama(
        model="qwen2.5:3b",
        temperature=0.2,
    )
