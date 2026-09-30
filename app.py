import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from src.graph import build_rag_graph
from src.ingestion import run_ingestion

app = FastAPI(title="Agentic AI RAG API")
pdf_path = Path(__file__).resolve().parent / "data" / "Ebook-Agentic-AI.pdf"
index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
vector_store = run_ingestion(str(pdf_path), index_name)
graph = build_rag_graph(index_name=index_name, vector_store=vector_store)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float

@app.get("/")
def home():
    return {"message": "Agentic AI RAG API is running. Use /chat to ask queries."}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    initial_state = {"question": request.query, "context": [], "answer": "", "score": 0.0}
    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"]
    )
