#!/usr/bin/env python3
"""Smoke experiment for muse-lab.

A tiny PyTorch MLP on a synthetic two-moons dataset (numpy only, no downloads).
Designed to be run from the runner notebooks (Colab / Kaggle) after they clone
this repo. Prints human-readable progress and, as the LAST line, a machine
readable summary:  METRICS_JSON: {...}   <- Muse reads results from here.
"""
import argparse, json, math, time
import numpy as np


def make_moons(n=2048, noise=0.12, seed=42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, math.pi, n1)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    y1 = np.zeros(n1, dtype=np.int64)
    n2 = n - n1
    t2 = rng.uniform(0, math.pi, n2)
    x2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    y2 = np.ones(n2, dtype=np.int64)
    x = np.concatenate([x1, x2]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([y1, y2])
    idx = rng.permutation(n)
    return x[idx].astype(np.float32), y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"device: {device} ({gpu_name})  torch: {torch.__version__}")

    x, y = make_moons(n=args.n, seed=args.seed)
    split = int(len(x) * 0.8)
    xtr = torch.tensor(x[:split]).to(device); ytr = torch.tensor(y[:split]).to(device)
    xte = torch.tensor(x[split:]).to(device); yte = torch.tensor(y[split:]).to(device)

    model = nn.Sequential(
        nn.Linear(2, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, 2),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss = lossf(model(xtr), ytr)
        loss.backward(); opt.step()
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                acc = (model(xte).argmax(1) == yte).float().mean().item()
            print(f"epoch {ep:3d}  loss {loss.item():.4f}  test_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        test_acc = (model(xte).argmax(1) == yte).float().mean().item()
        train_acc = (model(xtr).argmax(1) == ytr).float().mean().item()
    metrics = {
        "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4),
        "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
        "seed": args.seed, "device": device, "gpu": gpu_name,
        "seconds": round(time.time() - t0, 2),
    }
    print("METRICS_JSON: " + json.dumps(metrics))


if __name__ == "__main__":
    main()
