from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import build_conversational_rag_chain

app = FastAPI(title="PDF RAG API")
chain = None

def get_chain():
    global chain
    if chain is None:
        chain = build_conversational_rag_chain()
    return chain

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = get_chain().invoke(
            {"input": payload.question},
            config={"configurable": {"session_id": payload.session_id}}
        )
        return ChatResponse(answer=result["answer"])
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process the question.") from e
