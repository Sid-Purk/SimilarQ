from app.api import verify_question,similarity_search
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
# from app.services.model_service import get_model
# from contextlib import asynccontextmanager


limiter=Limiter(key_func=get_remote_address)
app=FastAPI()
app.state.limiter=limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://similar-q.vercel.app'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(verify_question.router)
app.include_router(similarity_search.router)
