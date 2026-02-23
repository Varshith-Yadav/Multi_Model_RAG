from fastapi import FastAPI

from app.routes.query import router

app = FastAPI(title="Multi Model RAG API")

app.include_router(router)

