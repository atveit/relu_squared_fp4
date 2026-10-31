"""FineWeb GPT-2 token shards (modded-nanogpt format: 256 int32 header, then uint16 tokens), shared by all platforms."""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from .config import DATA_REPO, TRAIN_SHARD, VAL_SHARD, VAL_TOKENS


def ensure_data(data_dir: str) -> str:
    """Download the two shards this study uses (~200 MB each) if they are missing; returns data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    for name in (TRAIN_SHARD, VAL_SHARD):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/{name}"
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
    return data_dir


def tokens(path: str, n: int) -> np.ndarray:
    hdr = np.fromfile(path, dtype=np.int32, count=256)
    assert hdr[0] == 20240520, f"unexpected magic in {path}"
    return np.fromfile(path, dtype=np.uint16, count=min(n, int(hdr[2])), offset=1024)


def load(data_dir: str, steps: int, batch: int, seq: int, val_tokens: int = VAL_TOKENS):
    train = tokens(os.path.join(data_dir, TRAIN_SHARD), steps * batch * seq + 1)
    val = tokens(os.path.join(data_dir, VAL_SHARD), val_tokens + 1)
    return train, val


def train_batch(train: np.ndarray, step: int, batch: int, seq: int):
    off = step * batch * seq
    c = train[off:off + batch * seq + 1].astype(np.int32)
    return c[:-1].reshape(batch, seq), c[1:].reshape(batch, seq)


def val_batches(val: np.ndarray, seq: int, batch: int = 16):
    for i in range(0, len(val) - batch * seq - 1, batch * seq):
        c = val[i:i + batch * seq + 1].astype(np.int32)
        yield c[:-1].reshape(batch, seq), c[1:].reshape(batch, seq)
