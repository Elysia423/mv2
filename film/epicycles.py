"""Turn glyph outlines into closed parametric paths and their Fourier epicycles."""
import os
from functools import lru_cache

import numpy as np

from .core import FONT_DIR


def glyph_contours(text, font_file, size=100.0):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), text, prop=FontProperties(fname=os.path.join(FONT_DIR, font_file)), size=size)
    polys = tp.to_polygons(closed_only=True)
    out = []
    for p in polys:
        p = np.asarray(p, float)
        if len(p) < 4:
            continue
        p = p * [1, -1]
        # drop tiny specks
        ext = np.ptp(p, axis=0)
        if max(ext) < size * 0.02:
            continue
        out.append(p)
    return out


def order_contours(contours):
    """Greedy left-to-right-ish ordering; rotate each contour to start near the previous end."""
    remaining = list(range(len(contours)))
    start = min(remaining, key=lambda i: contours[i][:, 0].min() + 0.3 * contours[i][:, 1].min())
    order = []
    pos = None
    while remaining:
        if pos is None:
            i = start
        else:
            def cost(j):
                d = np.hypot(*(contours[j] - pos).T).min()
                return d + 0.35 * max(0.0, contours[j][:, 0].mean() - pos[0]) * 0.0 + 0.6 * max(0.0, pos[0] - contours[j][:, 0].mean())
            i = min(remaining, key=cost)
        c = contours[i]
        if pos is not None:
            k = int(np.argmin(np.hypot(*(c - pos).T)))
            c = np.roll(c, -k, axis=0)
        else:
            k = int(np.argmin(c[:, 0] + c[:, 1]))
            c = np.roll(c, -k, axis=0)
        c = np.vstack([c, c[:1]])
        order.append(c)
        pos = c[-1]
        remaining.remove(i)
    return order


def build_path(contours, n=4096, jump_weight=0.35):
    """Concatenate contours into one closed path sampled uniformly (by weighted arc length).

    Returns complex samples z (n,), pen mask (n,) bool (False on jumps between contours).
    """
    segs = []
    pens = []
    cs = order_contours(contours)
    for i, c in enumerate(cs):
        segs.append(c)
        pens.append(np.ones(len(c), bool))
        nxt = cs[(i + 1) % len(cs)][0]
        jump = np.linspace(c[-1], nxt, 12)[1:-1]
        segs.append(jump)
        pens.append(np.zeros(len(jump), bool))
    P = np.vstack(segs)
    pen = np.concatenate(pens)
    d = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    w = np.where(pen, 1.0, jump_weight)
    # weight a segment by the pen state at its start
    s = np.concatenate([[0], np.cumsum(d * w)])
    total = s[-1]
    u = np.linspace(0, total, n, endpoint=False)
    Pc = np.vstack([P, P[:1]])
    x = np.interp(u, s, Pc[:, 0])
    y = np.interp(u, s, Pc[:, 1])
    idx = np.clip(np.searchsorted(s, u, side="right") - 1, 0, len(pen) - 1)
    pmask = pen[idx]
    # a sample is 'pen down' only if the segment it sits on starts and ends on a contour
    nxt_pen = pen[np.clip(idx + 1, 0, len(pen) - 1)]
    pmask = pmask & nxt_pen
    z = x + 1j * y
    cen = (z.real.min() + z.real.max()) / 2 + 1j * (z.imag.min() + z.imag.max()) / 2
    z = z - cen
    return z, pmask


class Epicycles:
    def __init__(self, z, pen=None):
        self.n = len(z)
        self.z = z
        self.pen = pen if pen is not None else np.ones(self.n, bool)
        c = np.fft.fft(z) / self.n
        k = np.fft.fftfreq(self.n, 1.0 / self.n).astype(int)
        self.c0 = c[0]
        order = np.argsort(-np.abs(c[1:])) + 1
        self.k = k[order]
        self.c = c[order]
        self._cache = {}

    def terms(self, m, mode="mag"):
        if mode == "mag":
            return self.k[:m], self.c[:m]
        # lowest frequencies first (true partial sums)
        idx = np.argsort(np.abs(self.k) + 0.1 * (self.k < 0))[:m]
        return self.k[idx], self.c[idx]

    def curve(self, m, samples=None, mode="mag"):
        key = (m, samples, mode)
        if key not in self._cache:
            ns = samples or self.n
            s = np.arange(ns) / ns
            k, c = self.terms(m, mode)
            z = np.full(ns, self.c0, complex)
            for a in range(0, len(k), 128):
                z += np.exp(2j * np.pi * np.outer(s, k[a:a + 128])) @ c[a:a + 128]
            self._cache[key] = z
        return self._cache[key]

    def chain(self, s, m, mode="mag"):
        """Circle centres along the chain at parameter s in [0,1). Returns complex array len m+1."""
        k, c = self.terms(m, mode)
        v = c * np.exp(2j * np.pi * k * s)
        return self.c0 + np.concatenate([[0], np.cumsum(v)])

    def radii(self, m, mode="mag"):
        return np.abs(self.terms(m, mode)[1])

    def pen_at(self, s):
        return self.pen[int(s * self.n) % self.n]


@lru_cache(maxsize=8)
def glyph_epicycles(text, font_file, size=100.0, n=4096, jump_weight=0.35):
    cs = glyph_contours(text, font_file, size)
    z, pen = build_path(cs, n, jump_weight)
    return Epicycles(z, pen)


def pen_runs(mask_prefix):
    """Split index range into runs where mask is True: returns list of (a,b)."""
    m = np.asarray(mask_prefix, bool)
    if len(m) == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))
