from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, github, health

app = FastAPI(title="IssueLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(github.router)
app.include_router(analysis.router)
