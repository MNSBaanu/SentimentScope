import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import MAX_LENGTH, IMDBDataset, download_imdb, load_split, train_val_split
from engine import calculate_accuracy, train
from model import DemoGPT, default_config, load_model, save_model

TOKENIZER_NAME = "bert-base-uncased"
LABELS = ["negative", "positive"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_download(args):
    print("Dataset ready at", download_imdb(args.data_dir))


def cmd_train(args):
    torch.manual_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    root = download_imdb(args.data_dir)
    train_data, val_data = train_val_split(load_split(root, "train"), seed=args.seed)

    print("Tokenizing ...")
    train_loader = DataLoader(IMDBDataset(train_data, tokenizer, args.max_length),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(IMDBDataset(val_data, tokenizer, args.max_length),
                            batch_size=args.batch_size)

    config = default_config(tokenizer.vocab_size, args.max_length)
    model = DemoGPT(config).to(device)
    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    save_model(model, config, TOKENIZER_NAME, args.checkpoint)
    print("Saved model to", args.checkpoint)


def cmd_evaluate(args):
    device = get_device()
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer"])
    test_df = load_split(download_imdb(args.data_dir), "test")
    test_loader = DataLoader(IMDBDataset(test_df, tokenizer, checkpoint["config"]["context_size"]),
                             batch_size=args.batch_size)
    print(f"Test Accuracy: {calculate_accuracy(model, test_loader, device):.2f}%")


def cmd_predict(args):
    device = get_device()
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer"])
    encoding = tokenizer(args.text, padding="max_length", truncation=True,
                         max_length=checkpoint["config"]["context_size"], return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(encoding["input_ids"].to(device)), dim=1)
    for text, p in zip(args.text, probs):
        label = int(p.argmax())
        print(f"{LABELS[label]} ({p[label]:.1%})\t{text}")


def main():
    parser = argparse.ArgumentParser(prog="sentimentscope")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/sentimentscope.pt")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download and extract the IMDB dataset").set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="Train the model and save a checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Report accuracy on the IMDB test set")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Classify one or more reviews")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
