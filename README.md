# SentimentScope

Binary sentiment classification (positive / negative) of IMDB movie reviews with a small transformer trained from scratch in PyTorch. Reviews are tokenized with the `bert-base-uncased` tokenizer; the model is a 4-layer GPT-style encoder with mean pooling and a linear classification head.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py download                 # fetch + extract IMDB into data/
python main.py train                    # train, save checkpoints/sentimentscope.pt
python main.py evaluate                 # accuracy on the 25k test reviews
python main.py predict "Loved it!" "A waste of time."
```

Train options: `--epochs 3 --batch-size 32 --lr 3e-4 --max-length 128 --seed 42`.
Global options (before the command): `--data-dir data --checkpoint checkpoints/sentimentscope.pt`.

### Web UI

After training, start the web app and open http://127.0.0.1:8000:

```bash
python -m uvicorn app:app
```

Type a review and get positive / negative with a confidence bar. The API is also available directly: `POST /predict` with `{"text": "..."}`.

Runs on CUDA when available, otherwise CPU. The dataset is tokenized once up front to keep CPU training practical.

## Project layout

```
data.py     download, load reviews, train/val split, IMDBDataset
model.py    AttentionHead, MultiHeadAttention, FeedForward, Block, DemoGPT, save/load
engine.py   training loop and accuracy
main.py     download / train / evaluate / predict commands
app.py      FastAPI web app (serves index.html, POST /predict)
index.html  web UI
```

## Results

The reference notebook run (3 epochs, GPU) reached 76.96% validation and 76.67% test accuracy (target: > 75%).
