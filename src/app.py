from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import meta, predict_url

app = FastAPI(
    title="Phishing URL Detection API",
    description="API for detecting phishing URLs from lexical URL features and domain popularity",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
def root():
    return {
        "message": "Phishing Detection API is running",
        "model": meta["model"],
        "trained_at": meta["trained_at"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: URLRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="URL must not be empty")
    try:
        return predict_url(url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")