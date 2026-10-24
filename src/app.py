from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.model import meta, predict_url


FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(
    title="Phishing URL Detection API",
    description="API for detecting phishing URLs from lexical URL features and domain popularity",
    version="2.0.0",
)


class URLRequest(BaseModel):
    url: str


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health():
    return {
        "message": "Phishing Detection API is running",
        "model": meta["model"],
        "trained_at": meta["trained_at"],
    }


@app.post("/predict")
def predict(request: URLRequest):
    result = predict_url(request.url)


    return result

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
