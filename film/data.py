"""Shared, lazily computed data (soundtrack spectrum, beat times, images)."""
import os
from functools import lru_cache

import numpy as np

from .core import FPS, ROOT

BUILD = os.path.join(ROOT, "build")


@lru_cache(maxsize=1)
def spectrum():
    p = os.path.join(BUILD, "spectrum.npy")
    if not os.path.exists(p):
        return np.zeros((int(240 * FPS), 256), np.float32)
    from scipy.ndimage import gaussian_filter1d
    db = np.load(p)
    nb = db.shape[1]
    db = db + np.linspace(0, 12, nb)[None, :]          # tilt: lift the highs (~ +2 dB/octave)
    db = gaussian_filter1d(db, 1.2, axis=1)
    lo, hi = np.percentile(db, 45), np.percentile(db, 99.7)
    return np.clip((db - lo) / (hi - lo), 0, 1).astype(np.float32)


def spec_at(t):
    s = spectrum()
    i = int(round(t * FPS))
    return s[max(0, min(len(s) - 1, i))]


@lru_cache(maxsize=1)
def kicks():
    p = os.path.join(BUILD, "soundtrack_kicks.npy")
    if os.path.exists(p):
        k = np.load(p)
        return k[np.argsort(k[:, 0])]
    return np.zeros((0, 2))


def beat_pulse(t, decay=0.18):
    k = kicks()
    if len(k) == 0:
        return 0.0
    i = np.searchsorted(k[:, 0], t, side="right") - 1
    if i < 0:
        return 0.0
    dt = t - k[i, 0]
    return float(min(1.0, k[i, 1]) * np.exp(-dt / decay))


@lru_cache(maxsize=None)
def image(name, size=None, gray=True):
    import skimage.data as sd
    from skimage.transform import resize
    img = getattr(sd, name)()
    img = img.astype(np.float32)
    img /= img.max()
    if gray and img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    if size is not None:
        img = resize(img, size, anti_aliasing=True).astype(np.float32)
    return img


@lru_cache(maxsize=1)
def brain():
    b = np.load(os.path.join(ROOT, "assets", "brain.npy")).astype(np.float32)
    s = b[5]
    return s / s.max()
