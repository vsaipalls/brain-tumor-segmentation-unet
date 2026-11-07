"""Shared pieces: data loading, dataset, UNet, loss and metrics."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"  # mixed precision saves memory and time on GPU


def load_data(data_dir):
    """Load the preprocessed arrays and the saved patient level split."""
    X = np.load(f"{data_dir}/X512.npy")  # (N, 1, 512, 512) float16, values 0 to 1
    Y = np.load(f"{data_dir}/Y512.npy")  # (N, 1, 512, 512) uint8, values 0 or 1
    T = np.load(f"{data_dir}/T.npy")     # tumor type per slice: 1, 2 or 3
    P = np.load(f"{data_dir}/P.npy")     # patient id per slice
    s = np.load(f"{data_dir}/split.npz")
    return X, Y, T, P, s["tr"], s["va"], s["te"]


class TumorDS(Dataset):
    def __init__(self, X, Y, idx, augment=False):
        self.X, self.Y, self.idx, self.augment = X, Y, idx, augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(self.X[j].astype(np.float32))  # float16 to float32
        y = torch.from_numpy(self.Y[j].astype(np.float32))
        if self.augment and torch.rand(1) < 0.5:  # same horizontal flip for image and mask
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def block(i, o):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, c=(16, 32, 64, 128, 256)):
        super().__init__()
        ins = (1,) + c[:3]
        self.enc = nn.ModuleList([block(ins[i], c[i]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.bott = block(c[3], c[4])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (3, 2, 1, 0)]
        )
        self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in (3, 2, 1, 0)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)  # keep a copy for the decoder
            x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))  # upsample, join the skip, convolve
        return self.out(x)  # logits, one per pixel


def dice_loss(logits, y, eps=1e-6):
    p = torch.sigmoid(logits.float())
    inter = (p * y).sum((1, 2, 3))
    tot = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (tot + eps)).mean()


def loss_fn(logits, y):
    return dice_loss(logits, y) + F.binary_cross_entropy_with_logits(logits.float(), y)


@torch.no_grad()
def dice_iou(logits, y, eps=1e-6):
    """Per slice Dice and IoU after thresholding the probabilities at 0.5."""
    p = (torch.sigmoid(logits.float()) > 0.5).float()
    inter = (p * y).sum((1, 2, 3))
    ps, ys = p.sum((1, 2, 3)), y.sum((1, 2, 3))
    return (2 * inter + eps) / (ps + ys + eps), (inter + eps) / (ps + ys - inter + eps)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    D, I = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            out = model(x)
        d, i = dice_iou(out, y)
        D.append(d.cpu())
        I.append(i.cpu())
    return torch.cat(D), torch.cat(I)
