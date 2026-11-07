"""Turn the raw .mat files into arrays and make the patient level split.

Usage:
    python preprocess.py --raw path/to/unzipped/mat/files --out data
"""
import argparse
import glob
import os
from collections import Counter

import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

SIZE = 512


def load(path):
    with h5py.File(path, "r") as f:
        d = f["cjdata"]
        img = np.array(d["image"], dtype=np.float32)
        mask = np.array(d["tumorMask"], dtype=np.uint8)
        label = int(np.array(d["label"]).item())
        pid = "".join(chr(c) for c in np.array(d["PID"]).flatten())
    return img, mask, label, pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder holding the .mat files")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.raw}/**/*.mat", recursive=True))
    files = [f for f in files if os.path.basename(f) != "cvind.mat"]
    print(len(files), "slice files found (expect 3064)")

    X = np.zeros((len(files), 1, SIZE, SIZE), dtype=np.float16)
    Y = np.zeros((len(files), 1, SIZE, SIZE), dtype=np.uint8)
    T, P, resized = [], [], 0
    for i, p in enumerate(files):
        img, mask, label, pid = load(p)
        if img.shape != (SIZE, SIZE):  # a few slices are 256x256
            resized += 1
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        lo, hi = img.min(), img.max()
        X[i, 0] = ((img - lo) / (hi - lo + 1e-8)).astype(np.float16)  # per image scaling to 0..1
        Y[i, 0] = mask > 0
        T.append(label)
        P.append(pid)
    T, P = np.array(T), np.array(P)
    print("slices resized from a smaller size:", resized)

    # Patient level split: every slice of a patient goes to the same set.
    pids = np.unique(P)
    pid_type = {p: T[P == p][0] for p in pids}
    assert all(len(set(T[P == p])) == 1 for p in pids), "a patient has more than one tumor type"
    ptypes = np.array([pid_type[p] for p in pids])
    tr_p, tmp_p = train_test_split(pids, test_size=0.30, stratify=ptypes, random_state=a.seed)
    va_p, te_p = train_test_split(
        tmp_p, test_size=0.50, stratify=[pid_type[p] for p in tmp_p], random_state=a.seed
    )
    tr = np.where(np.isin(P, tr_p))[0]
    va = np.where(np.isin(P, va_p))[0]
    te = np.where(np.isin(P, te_p))[0]

    print("patients (train, val, test):", len(tr_p), len(va_p), len(te_p))
    print("slices   (train, val, test):", len(tr), len(va), len(te))
    print("patient overlap:", len(set(tr_p) & set(va_p)), len(set(tr_p) & set(te_p)),
          len(set(va_p) & set(te_p)), "(all must be 0)")
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        print(name, dict(Counter(T[idx].tolist())))

    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/X512.npy", X)
    np.save(f"{a.out}/Y512.npy", Y)
    np.save(f"{a.out}/T.npy", T)
    np.save(f"{a.out}/P.npy", P)
    np.savez(f"{a.out}/split.npz", tr=tr, va=va, te=te)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
