"""Score the saved model on the held out test patients. Run this once, at the end.

Usage:
    python evaluate.py --data data --weights best.pt
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import TumorDS, UNet, device, evaluate, load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    X, Y, T, P, tr, va, te = load_data(a.data)
    model = UNet().to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))

    dice, iou = evaluate(model, DataLoader(TumorDS(X, Y, te), batch_size=a.bs))
    print(f"TEST overall  dice {dice.mean():.4f}  iou {iou.mean():.4f}  ({len(te)} slices)")
    te_T = T[te]
    for t, name in [(1, "meningioma"), (2, "glioma"), (3, "pituitary")]:
        m = torch.from_numpy(te_T == t)
        print(f"{name:11s} dice {dice[m].mean():.4f}  iou {iou[m].mean():.4f}  ({int(m.sum())} slices)")

    os.makedirs(a.results, exist_ok=True)
    model.eval()
    picks = np.random.RandomState(0).choice(te, 4, replace=False)
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for k, j in enumerate(picks):
        x = torch.from_numpy(X[j].astype(np.float32))[None].to(device)
        with torch.no_grad():
            p = (torch.sigmoid(model(x)) > 0.5)[0, 0].cpu().numpy()
        ax[0, k].imshow(X[j, 0].astype(np.float32), cmap="gray")
        ax[0, k].set_title("MRI")
        ax[1, k].imshow(Y[j, 0], cmap="gray")
        ax[1, k].set_title("ground truth")
        ax[2, k].imshow(p, cmap="gray")
        ax[2, k].set_title("prediction")
        for r in range(3):
            ax[r, k].axis("off")
    plt.tight_layout()
    plt.savefig(f"{a.results}/overlay.png", dpi=150)
    print("saved", f"{a.results}/overlay.png")


if __name__ == "__main__":
    main()
