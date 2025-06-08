from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.text_agent import get_text_qa_chain

router = APIRouter()

qa_chain = get_text_qa_chain()

class Query(BaseModel):
    question: str

@router.post("/text-search")
async def text_search(query: Query):
    try:
        response = qa_chain.invoke(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
