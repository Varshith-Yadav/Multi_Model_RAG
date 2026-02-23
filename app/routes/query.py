from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from RAG.augmentation.prompt_builder import build_prompt
from RAG.generation.llm import generate
from RAG.retrieval.retriever import IndexNotReadyError, retrieve

router = APIRouter()


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


def _run_query(question: str, top_k: int):
    docs = retrieve(question, k=top_k)
    prompt = build_prompt(question, docs)
    answer = generate(prompt)
    return {"answer": answer, "sources": docs}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/query")
def ask_query(q: str, top_k: int = 5):
    try:
        return _run_query(q, top_k)
    except IndexNotReadyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query")
def ask(payload: QueryRequest):
    try:
        return _run_query(payload.q, payload.top_k)
    except IndexNotReadyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

