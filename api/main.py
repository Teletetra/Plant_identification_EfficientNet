from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.prediction import router
from src.utils.config import APP_NAME, CORS_ORIGINS

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Plant image classification API.",
)

origins = ["*"] if CORS_ORIGINS.strip() == "*" else [
    origin.strip() for origin in CORS_ORIGINS.split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "status": "running",
        "docs": "/docs",
    }
