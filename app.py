from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from main import LABELS, get_device
from model import load_model

CHECKPOINT = "checkpoints/sentimentscope.pt"

device = get_device()
model, checkpoint = load_model(CHECKPOINT, device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer"])

app = FastAPI(title="SentimentScope")


class Review(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.post("/predict")
def predict(review: Review):
    encoding = tokenizer(review.text, padding="max_length", truncation=True,
                         max_length=checkpoint["config"]["context_size"], return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(encoding["input_ids"].to(device)), dim=1)[0]
    label = int(probs.argmax())
    return {"label": LABELS[label], "confidence": probs[label].item(), "positive": probs[1].item()}
