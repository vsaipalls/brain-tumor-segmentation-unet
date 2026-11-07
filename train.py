"""Train the UNet. Keeps the weights with the best validation Dice.

Usage:
    python train.py --data data --epochs 20 --out best.pt
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import TumorDS, UNet, device, evaluate, load_data, loss_fn, use_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="best.pt")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    print("device:", device)

    X, Y, T, P, tr, va, te = load_data(a.data)
    train_loader = DataLoader(TumorDS(X, Y, tr, augment=True), batch_size=a.bs, shuffle=True)
    val_loader = DataLoader(TumorDS(X, Y, va), batch_size=a.bs)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = 0.0
    for ep in range(a.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        vd, _ = evaluate(model, val_loader)
        print(f"epoch {ep + 1}/{a.epochs}  train loss {total / len(tr):.4f}  val dice {vd.mean():.4f}")
        if vd.mean() > best:
            best = float(vd.mean())
            torch.save(model.state_dict(), a.out)
    print(f"best validation dice {best:.4f}, weights saved to {a.out}")


if __name__ == "__main__":
    main()
