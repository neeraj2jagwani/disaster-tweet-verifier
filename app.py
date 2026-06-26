import re
import pickle
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.pkl")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"@(\w+)", r" \1 ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_input(text: str, keyword: str = "") -> str:
    if not isinstance(keyword, str):
        keyword = ""
    keyword = keyword.replace("%20", " ")
    return (clean_text(keyword) + " " + clean_text(text)).strip()


with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

if isinstance(artifact, dict):
    model = artifact["pipeline"]
    THRESHOLD = float(artifact.get("threshold", 0.5))
else:
    model = artifact
    THRESHOLD = 0.5

app = FastAPI(title="Disaster Tweet Verifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text Input")
    keyword: str = Field("", description="Optional disaster keyword")


class PredictResponse(BaseModel):
    label: int = Field(..., ge=0, le=1, description="Predicted class: 1 = real disaster")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence in label=1")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = build_input(req.text, req.keyword)
    proba = float(model.predict_proba([cleaned])[0, 1])
    label = int(proba >= THRESHOLD)
    return {"label": label, "score": round(proba, 4)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
