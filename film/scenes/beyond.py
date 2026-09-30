"""'And beyond': eight short vignettes of the Fourier transform at work."""
import math

import cairo
import cv2
import numpy as np

from .. import story as S
from ..core import (BLUE, CYAN, GOLD, H, LUT_DIVERGE, LUT_FIRE, LUT_GOLD, LUT_GRAY, LUT_ICE, MAGENTA, ORANGE, RED,
                    TEAL, VIOLET, W, WHITE, apply_lut, arrow, background, circle, clamp, draw_math, draw_text,
                    draw_text_chars, ease_out, ease_out_back, ease_out_expo, glow_dot, hsv, lerp, mix, paint_image,
                    polyline, remap, rounded_rect, set_rgba, smooth, smoother, stroke_poly, surface_from_array, window)
from ..data import beat_pulse, brain
from ..three import Camera, line3, line3_depth, axis3
from .base import Scene


def vignette_title(ctx, lt, zh, en, sub=None, x=110, y=150, layout="tl"):
    """Shot title. layout: 'tl' top-left, 'tr' top-right, 'tc' top-centre, 'v' vertical along the left edge."""
    a = smooth(lt / 0.4)
    if layout == "v":
        for i, ch in enumerate(zh):
            ai = smooth((lt - 0.06 * i) / 0.3)
            draw_text(ctx, ch, 110, 250 + i * 84, size=72, font="serif", weight=800, color=WHITE, alpha=ai,
                      glow=10, glow_alpha=0.35)
        yy = 250 + len(zh) * 84 + 10
        draw_text(ctx, en, 110, yy, size=22, font="latin", weight=600, color=GOLD, alpha=0.85 * smooth((lt - 0.2) / 0.5),
                  tracking=0.1)
        if sub:
            for i, ch in enumerate(sub):
                draw_text(ctx, ch, 190, 262 + i * 38, size=30, font="sans", weight=400, color=(0.85, 0.9, 1.0),
                          alpha=0.8 * smooth((lt - 0.5 - 0.02 * i) / 0.4))
        return
    anchor = {"tl": "left", "tr": "right", "tc": "center"}[layout]
    if layout == "tr":
        x = W - 110
    elif layout == "tc":
        x = W / 2
    draw_text_chars(ctx, zh, x, y, lt, size=66, font="serif", weight=800, color=WHITE, alpha=a, anchor=anchor,
                    tracking=0.12, stagger=0.06, dur=0.35, rise=12, glow=10, glow_alpha=0.35)
    draw_text(ctx, en, x + (3 if anchor == "left" else 0), y + 56, size=24, font="latin", weight=600, color=GOLD,
              alpha=0.85 * smooth((lt - 0.2) / 0.5), anchor=anchor, tracking=0.18)
    if sub:
        draw_text(ctx, sub, x + (2 if anchor == "left" else 0), y + 98, size=30, font="sans", weight=400,
                  color=(0.85, 0.9, 1.0), alpha=0.8 * smooth((lt - 0.5) / 0.5), anchor=anchor, tracking=0.06)


class Vignette(Scene):
    fade_in = 0.25
    fade_out = 0.25
    bloom = 0.9

    def __init__(self, start):
        self.start = start
        self.end = start + 4.0

    def effects(self, t, lt):
        return {"flash": 0.12 * math.exp(-max(0.0, lt) / 0.15)}


# ---------------------------------------------------------------------- header
class BeyondHeader(Scene):
    """Title card over the real Hubble eXtreme Deep Field, slowly pushing in."""
    start, end = S.T_BEYOND, S.T_BEYOND + 4.0
    fade_out = 0.25
    bloom = 0.6

    def prepare(self):
        from ..data import image
        self.xdf = image("hubble_deep_field", None, gray=False)[..., :3]

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()
        h, w = self.xdf.shape[:2]
        z = 1.0 + 0.12 * lt / 4.0
        surf = surface_from_array(self.xdf * 0.8)
        sc = max(W / w, H / h) * z
        paint_image(ctx, surf, W / 2, H / 2, w * sc, h * sc, smooth(lt / 0.3))
        g = cairo.RadialGradient(W / 2, H / 2, 100, W / 2, H / 2, W * 0.6)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.55)
        g.add_color_stop_rgba(1, 0, 0, 0, 0.15)
        ctx.set_source(g)
        ctx.paint()
        s_ = 1.25 - 0.25 * ease_out_expo(lt / 1.2)
        draw_text(ctx, "不止于此", W / 2, H / 2 - 30, size=150, font="serif", weight=900, color=WHITE,
                  alpha=smooth(lt / 0.2), tracking=0.18, glow=18, glow_alpha=0.5, scale=s_)
        draw_text(ctx, "AND  BEYOND", W / 2, H / 2 + 90, size=32, font="latin", weight=300, color=GOLD, alpha=0.9,
                  tracking=0.9, reveal=ease_out((lt - 0.6) / 1.2, 2))
        draw_text_chars(ctx, "从星辰到基因，从大脑到人工智能", W / 2, H / 2 + 170, lt - 1.3, size=36, font="sans",
                        weight=400, color=(0.9, 0.93, 1.0), alpha=0.95, tracking=0.2, stagger=0.05)
        draw_text(ctx, "哈勃极深场 · NASA / ESA（真实照片）", W - 60, H - 170, size=24, font="sans", weight=400,
                  color=WHITE, alpha=0.6 * smooth((lt - 1.0) / 0.5), anchor="right")

    def effects(self, t, lt):
        return {"flash": 0.7 * math.exp(-lt / 0.2), "chroma": 4 * math.exp(-lt / 0.4)}


# ---------------------------------------------------------------------- 1. light
FRAUNHOFER = [(656.3, "H"), (589.0, "Na"), (527.0, "Fe"), (517.3, "Mg"), (486.1, "H"), (430.8, "Ca/Fe"), (410.2, "H")]


def wl_rgb(wl):
    """Approximate visible wavelength (nm) -> RGB."""
    if wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    f = 1.0 if 420 <= wl <= 680 else 0.4 + 0.6 * (1 - min(abs(wl - 550) - 130, 50) / 50)
    return (r * f, g * f, b * f)


# (air wavelength nm, depth, half-width nm) of the strongest solar absorption lines, plus telluric O2 bands
SOLAR_LINES = [
    (656.28, 0.82, 0.10), (486.13, 0.80, 0.09), (434.05, 0.74, 0.08), (410.17, 0.70, 0.07),
    (589.59, 0.82, 0.05), (588.99, 0.86, 0.05), (518.36, 0.82, 0.05), (517.27, 0.78, 0.05), (516.73, 0.72, 0.05),
    (527.04, 0.62, 0.03), (526.95, 0.55, 0.03), (422.67, 0.82, 0.05), (430.8, 0.55, 0.25), (404.58, 0.72, 0.03),
    (406.36, 0.62, 0.03), (407.17, 0.62, 0.03), (432.58, 0.70, 0.03), (438.35, 0.75, 0.03), (440.48, 0.66, 0.03),
    (492.05, 0.55, 0.02), (495.76, 0.60, 0.02), (501.21, 0.45, 0.02), (516.89, 0.55, 0.02), (522.72, 0.55, 0.02),
    (532.80, 0.60, 0.02), (537.15, 0.58, 0.02), (539.71, 0.58, 0.02), (540.58, 0.55, 0.02), (543.45, 0.50, 0.02),
    (544.69, 0.55, 0.02), (561.56, 0.45, 0.02), (606.55, 0.40, 0.02), (613.66, 0.45, 0.02), (623.07, 0.42, 0.02),
    (625.26, 0.40, 0.02), (641.17, 0.45, 0.02), (649.50, 0.40, 0.02), (643.08, 0.40, 0.02), (612.22, 0.45, 0.02),
    (616.22, 0.50, 0.02), (644.98, 0.40, 0.02), (671.8, 0.30, 0.02),
]
TELLURIC = [(686.7, 688.4, 0.9), (627.6, 629.0, 0.55), (694.0, 697.0, 0.35)]
LINE_LABELS = [(656.28, "Hα"), (589.29, "Na D"), (517.3, "Mg b"), (486.13, "Hβ"), (527.0, "Fe"), (430.8, "G"),
               (434.05, "Hγ"), (422.67, "Ca")]


def solar_spectrum(wl, seed=4):
    """Continuum x absorption lines at the real Fraunhofer positions (weak lines scattered at random)."""
    T = 5772.0
    lam = wl * 1e-9
    bb = 1 / (lam ** 5 * (np.exp(1.4388e-2 / (lam * T)) - 1))
    bb = bb / bb.max()
    trans = np.ones_like(wl)
    for c, d, w in SOLAR_LINES:
        w = w * 2.5  # widened so the lines survive on screen
        trans *= 1 - d / (1 + ((wl - c) / w) ** 2)
    for a0, a1, d in TELLURIC:
        band = np.clip(1 - np.abs(wl - (a0 + a1) / 2) / ((a1 - a0) / 2), 0, 1)
        trans *= 1 - d * band * (0.6 + 0.4 * np.cos(wl * 60) ** 2)
    rng = np.random.default_rng(seed)
    n = 900
    cs = np.sort(rng.uniform(400, 700, n) ** 1.0)
    ds = rng.uniform(0.04, 0.35, n) * np.clip((720 - cs) / 200, 0.3, 1.2)
    ws = rng.uniform(0.02, 0.05, n)
    for c, d, w in zip(cs, ds, ws):
        m = np.abs(wl - c) < 0.2
        trans[m] *= 1 - d / (1 + ((wl[m] - c) / w) ** 2)
    return bb, trans


class Prism(Vignette):
    bloom = 0.35
    ROWS = 12
    WL0, WL1 = 400.0, 700.0

    def prepare(self):
        n = 3000
        span = (self.WL1 - self.WL0) / self.ROWS
        rows = []
        for r in range(self.ROWS):
            hi = self.WL1 - r * span
            wl = np.linspace(hi - span, hi, n)
            bb, tr = solar_spectrum(wl)
            col = np.array([wl_rgb(x) for x in wl[::10]])
            col = np.repeat(col, 10, axis=0)[:n]
            rows.append(np.clip(col * (0.35 + 0.35 * bb[:, None]) * tr[:, None], 0, 1))
        self.rows = rows
        self.span = span

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        cx, cy, side = 470.0, 560.0, 300.0
        hgt = side * math.sqrt(3) / 2
        A = (cx, cy - hgt * 2 / 3)
        B = (cx - side / 2, cy + hgt / 3)
        C = (cx + side / 2, cy + hgt / 3)
        pin = (lerp(A[0], B[0], 0.55), lerp(A[1], B[1], 0.55))
        pout_mid = (lerp(A[0], C[0], 0.58), lerp(A[1], C[1], 0.58))
        beam = ease_out(lt / 0.45, 2)
        src = (0.0, pin[1] - 120)
        bx, by = lerp(src[0], pin[0], beam), lerp(src[1], pin[1], beam)
        for wdt, al in [(18, 0.06), (8, 0.15), (3, 0.9)]:
            ctx.move_to(*src)
            ctx.line_to(bx, by)
            set_rgba(ctx, WHITE, al)
            ctx.set_line_width(wdt)
            ctx.stroke()
        polyline(ctx, [A, B, C], close=True)
        g = cairo.LinearGradient(B[0], B[1], C[0], A[1])
        g.add_color_stop_rgba(0, 0.5, 0.7, 1.0, 0.10)
        g.add_color_stop_rgba(1, 0.8, 0.9, 1.0, 0.22)
        ctx.set_source(g)
        ctx.fill_preserve()
        set_rgba(ctx, (0.8, 0.9, 1.0), 0.8)
        ctx.set_line_width(2)
        ctx.stroke()
        # the atlas panel
        px0, px1, py0, rh, gap = 880.0, 1810.0, 250.0, 34.0, 8.0
        fan = ease_out((lt - 0.35) / 0.5, 2)
        if fan > 0:
            nr = 80
            for i in range(nr):
                u = i / (nr - 1)
                col = wl_rgb(700 - 300 * u)
                pm = (pout_mid[0] + (u - 0.5) * 16, pout_mid[1] + (u - 0.5) * 26)
                ctx.move_to(*pin)
                ctx.line_to(*pm)
                set_rgba(ctx, col, 0.08)
                ctx.set_line_width(2)
                ctx.stroke()
                ty = py0 + u * (self.ROWS * (rh + gap))
                ctx.move_to(*pm)
                ctx.line_to(lerp(pm[0], px0, fan), lerp(pm[1], ty, fan))
                set_rgba(ctx, col, 0.10)
                ctx.set_line_width(4)
                ctx.stroke()
        for r, row in enumerate(self.rows):
            a = smooth((lt - 0.75 - 0.05 * r) / 0.3)
            if a <= 0:
                continue
            y = py0 + r * (rh + gap)
            paint_image(ctx, surface_from_array(row[None, :, :] * np.ones((4, 1, 1))), px0, y, px1 - px0, rh, a,
                        anchor="topleft", filt=cairo.FILTER_BEST)
        la = smooth((lt - 1.5) / 0.4)
        if la > 0:
            for wl, lab in LINE_LABELS:
                r = int((self.WL1 - wl) // self.span)
                x = px0 + (wl - (self.WL1 - (r + 1) * self.span)) / self.span * (px1 - px0)
                y = py0 + r * (rh + gap)
                draw_text(ctx, lab, x, y - 13, size=24, font="sans", weight=700, color=WHITE, alpha=0.9 * la)
        # helium: the yellow emission line seen in the 1868 eclipse
        ah = smooth((lt - 2.2) / 0.3)
        if ah > 0:
            wl = 587.56
            r = int((self.WL1 - wl) // self.span)
            x = px0 + (wl - (self.WL1 - (r + 1) * self.span)) / self.span * (px1 - px0)
            y = py0 + r * (rh + gap)
            set_rgba(ctx, (1.0, 1.0, 0.8), ah)
            ctx.rectangle(x - 2, y - 6, 4, rh + 12)
            ctx.fill()
            circle(ctx, x, y + rh / 2, 30, (1.0, 0.95, 0.6), 0.9 * ah, width=2.5)
            yb = py0 + self.ROWS * (rh + gap) + 20
            ctx.move_to(x, y + rh / 2 + 30)
            ctx.line_to(x, yb)
            set_rgba(ctx, (1.0, 0.95, 0.6), 0.8 * ah)
            ctx.set_line_width(1.5)
            ctx.stroke()
            draw_text(ctx, "氦 He · 1868 年日食中首次观测到", x + 14, yb + 16, size=30, font="sans", weight=700,
                      color=(1.0, 0.95, 0.6), alpha=ah, anchor="left", glow=6, glow_alpha=0.5)
        vignette_title(ctx, lt, "光谱", "SPECTRUM", "棱镜：把阳光按频率展开", y=130)


# ---------------------------------------------------------------------- 2. DNA
class DNA(Vignette):
    def prepare(self):
        N = 512
        img = np.zeros((N, N), np.float32)
        P, R, turns = 64.0, 20.0, 7
        z = np.arange(-turns * P / 2, turns * P / 2, P / 40)
        for ph in [0.0, 2 * np.pi * 3 / 8]:
            x = N / 2 + R * np.cos(2 * np.pi * z / P + ph)
            y = N / 2 + z
            for xi, yi in zip(x, y):
                ix, iy = int(round(xi)), int(round(yi))
                if 0 <= ix < N and 0 <= iy < N:
                    img[iy, ix] += 1
        # base pairs (10 per turn)
        for zb in np.arange(-turns * P / 2, turns * P / 2, P / 10):
            iy = int(round(N / 2 + zb))
            if 0 <= iy < N:
                img[iy, int(N / 2 - R * 0.7):int(N / 2 + R * 0.7)] += 0.25
        img = cv2.GaussianBlur(img, (0, 0), 1.8)
        yy = np.arange(N)
        win = np.exp(-((yy - N / 2) / (turns * P / 2.6)) ** 2)[:, None] * np.ones((1, N))
        F = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
        L = np.log1p(F / F.mean() * 30)
        c = L[N // 2 - 96:N // 2 + 96, N // 2 - 96:N // 2 + 96]
        c = (c - np.percentile(c, 55)) / (np.percentile(c, 99.8) - np.percentile(c, 55))
        c = cv2.resize(np.clip(c, 0, 1), (440, 440), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:440, 0:440]
        rr = np.hypot(yy - 220, xx - 220) / 220
        c = c * np.clip((1 - rr) / 0.12, 0, 1)
        self.diff = np.clip(cv2.GaussianBlur(c, (0, 0), 1.6) * 1.15, 0, 1)

    def helix(self, ctx, t, lt, cx, cy, a):
        turns, P, R = 3.2, 190.0, 80.0
        n = 260
        z = np.linspace(-turns * P / 2, turns * P / 2, n)
        rot = lt * 1.3
        pts = []
        for s, ph in enumerate([0.0, 2 * np.pi * 3 / 8]):
            ang = 2 * np.pi * z / P + ph + rot
            x = cx + R * np.cos(ang)
            dep = np.sin(ang)
            y = cy + z * 0.95
            pts.append((x, y, dep))
        # rungs
        for i in range(0, n, 8):
            x1, y1, d1 = pts[0][0][i], pts[0][1][i], pts[0][2][i]
            x2, y2, d2 = pts[1][0][i], pts[1][1][i], pts[1][2][i]
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            set_rgba(ctx, mix(GOLD, WHITE, 0.3), a * (0.25 + 0.2 * (d1 + d2) / 2 + 0.2))
            ctx.set_line_width(2.5)
            ctx.stroke()
        for s, (x, y, dep) in enumerate(pts):
            col = CYAN if s == 0 else MAGENTA
            for i in range(n - 1):
                ctx.move_to(x[i], y[i])
                ctx.line_to(x[i + 1], y[i + 1])
                set_rgba(ctx, col, a * (0.35 + 0.65 * (dep[i] + 1) / 2))
                ctx.set_line_width(3 + 3 * (dep[i] + 1) / 2)
                ctx.stroke()

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        a = smooth(lt / 0.5)
        self.helix(ctx, t, lt, 640.0, 520.0, a)
        # X-ray beam
        bx = ease_out((lt - 0.35) / 0.4, 2)
        if bx > 0:
            ctx.move_to(740, 520)
            ctx.line_to(lerp(740, 1150, bx), 520)
            set_rgba(ctx, (0.7, 0.8, 1.0), 0.5)
            ctx.set_line_width(3)
            ctx.stroke()
            arrow(ctx, 900, 520, lerp(900, 1110, bx), 520, WHITE, 0.6 * bx, 2, 14)
            draw_text(ctx, "X 射线", 925, 490, size=28, font="sans", weight=500, color=WHITE, alpha=0.8 * bx, anchor="left")
        rev = smooth((lt - 0.7) / 0.8)
        if rev > 0:
            vis = apply_lut(self.diff * rev, LUT_GOLD)
            surf = surface_from_array(vis)
            paint_image(ctx, surf, 1400, 520, 440, 440, 1.0)
            circle(ctx, 1400, 520, 222, (0.8, 0.7, 0.5), 0.35 * rev, width=1.5)
            draw_text(ctx, "Photo 51 · 1952", 1400, 780, size=30, font="latin", weight=500, color=GOLD, alpha=rev,
                      tracking=0.1)
            draw_text(ctx, "罗莎琳德·富兰克林", 1400, 812, size=28, font="sans", weight=400, color=WHITE,
                      alpha=0.6 * rev)
        vignette_title(ctx, lt, "双螺旋", "DNA · PHOTO 51 · 1952", "衍射图记录分子的傅里叶变换", layout="v")


# ---------------------------------------------------------------------- 3. MRI
class MRI(Vignette):
    def prepare(self):
        b = brain()
        b = cv2.resize(b, (384, 384), interpolation=cv2.INTER_CUBIC)
        self.img = np.clip(b, 0, None)
        self.K = np.fft.fftshift(np.fft.fft2(self.img))
        mag = np.log1p(np.abs(self.K))
        self.kvis = np.clip((mag - np.percentile(mag, 30)) / (np.percentile(mag, 99.9) - np.percentile(mag, 30)), 0, 1)
        n = self.K.shape[0]
        rows = np.arange(n)
        self.order = rows[np.argsort(np.abs(rows - n // 2) + 0.1 * (rows < n // 2))]

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        n = self.K.shape[0]
        prog = clamp((lt - 0.35) / 2.9) ** 1.5
        nrows = max(1, int(prog * n))
        rows = self.order[:nrows]
        mask = np.zeros((n, 1), np.float32)
        mask[rows] = 1
        kv = self.kvis * mask
        img = np.abs(np.fft.ifft2(np.fft.ifftshift(self.K * mask)))
        img = np.clip(img / (self.img.max() * 0.9), 0, 1)
        a = smooth(lt / 0.4)
        kx, ix, y, s = 620.0, 1330.0, 520.0, 500.0
        paint_image(ctx, surface_from_array(apply_lut(kv, LUT_ICE)), kx, y, s, s, a)
        paint_image(ctx, surface_from_array(apply_lut(img, LUT_GRAY)), ix, y, s, s, a)
        for x, c in [(kx, CYAN), (ix, WHITE)]:
            set_rgba(ctx, c, 0.35 * a)
            ctx.set_line_width(1.2)
            ctx.rectangle(x - s / 2 - 6, y - s / 2 - 6, s + 12, s + 12)
            ctx.stroke()
        # current acquisition line
        if prog < 1:
            r = self.order[nrows - 1]
            yy = y - s / 2 + (r + 0.5) / n * s
            set_rgba(ctx, GOLD, 0.9)
            ctx.set_line_width(2)
            ctx.move_to(kx - s / 2, yy)
            ctx.line_to(kx + s / 2, yy)
            ctx.stroke()
        arrow(ctx, kx + s / 2 + 30, y, ix - s / 2 - 30, y, GOLD, a * 0.9, 2.5, 16)
        draw_math(ctx, r"$\mathcal{F}^{-1}$", (kx + ix) / 2, y - 40, size=36, color=GOLD, alpha=a)
        draw_text(ctx, "k 空间（频率）", kx, y + s / 2 + 40, size=30, font="sans", weight=500, color=CYAN, alpha=a)
        draw_text(ctx, "大脑图像", ix, y + s / 2 + 40, size=30, font="sans", weight=500, color=WHITE, alpha=a)
        vignette_title(ctx, lt, "核磁共振", "MRI · k-SPACE", "扫描仪测量的，其实是频率", y=110, layout="tc")


# ---------------------------------------------------------------------- 4. gravitational waves
class GW(Vignette):
    """Two black holes spiral in (illustration) next to the real LIGO GW150914 strain and its wavelet map."""
    T_MERGE = S.GW_MERGE
    T_START = S.T_GW
    PEAK = -0.015           # peak of the real signal relative to the catalogue event time (s)
    WIN = (-0.20, 0.05)     # real-time window shown

    def f_of(self, t):
        tm = self.T_MERGE - self.T_START
        tau = max(self.T_MERGE - t, 1e-3)
        return min(420.0, 55 * (tau / tm) ** (-3 / 8)) if t < self.T_MERGE else 420.0

    def real_t(self, t):
        k = (self.PEAK - self.WIN[0]) / (self.T_MERGE - (self.T_START + 0.2))
        return self.PEAK + (t - self.T_MERGE) * k

    def prepare(self):
        import os
        from ..core import ROOT
        ts = np.arange(self.T_START - 1.0, self.T_MERGE + 3.0, 0.002)
        f = np.array([self.f_of(x) if x >= self.T_START else 55.0 for x in ts])
        self.ts = ts
        self.ph = np.cumsum(f / 60.0 * 2 * np.pi * 0.002)  # slowed-down visual orbital phase
        d = np.load(os.path.join(ROOT, "assets", "ligo", "gw150914.npz"))
        sel = (d["t"] >= self.WIN[0]) & (d["t"] <= self.WIN[1])
        self.rt, self.rh, self.rl = d["t"][sel], d["h"][sel], d["l"][sel]
        m = max(np.abs(self.rh).max(), np.abs(self.rl).max())
        self.rh, self.rl = self.rh / m, self.rl / m
        tf = d["tf"][:, sel]
        tf = np.log1p(tf / np.percentile(tf, 50))
        self.tf = np.clip(tf / tf.max(), 0, 1) ** 2.6
        self.tf_freqs = d["freqs"]

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.02, 0.03, 0.08))
        cx, cy = 470.0, 560.0
        merged = t >= self.T_MERGE
        f = self.f_of(t)
        sep = 120 * (55.0 / f) ** (2 / 3) if not merged else 0.0
        ph = float(np.interp(t, self.ts, self.ph))
        ripple_t = t - self.T_MERGE
        for gy in np.arange(240, 880, 36):
            xs = np.linspace(80, 860, 110)
            d = np.hypot(xs - cx, gy - cy)
            wave = np.sin(d * 0.06 - (t * 9)) * np.exp(-d / 420) * (8 + 18 * (1 - min(1, sep / 120)))
            if merged:
                wave += 30 * np.sin(d * 0.05 - ripple_t * 14) * np.exp(-((d - ripple_t * 420) / 90) ** 2)
            ys = gy + wave - 2600 / (d + 60) * (gy - cy) / (d + 1) * 0.6
            stroke_poly(ctx, np.stack([xs, ys], 1), BLUE, 1.2, 0.38)
        if not merged:
            for s_ in (0, 1):
                ang = ph + s_ * math.pi
                x = cx + sep / 2 * math.cos(ang)
                y = cy + sep / 2 * math.sin(ang) * 0.45
                circle(ctx, x, y, 40, ORANGE, 0.12, width=14)
                circle(ctx, x, y, 27, (0, 0, 0), 1.0, fill=True)
                circle(ctx, x, y, 29, GOLD, 0.9, width=3)
        else:
            circle(ctx, cx, cy, 32, (0, 0, 0), 1.0, fill=True)
            circle(ctx, cx, cy, 34, GOLD, 0.9, width=3)
            for k in range(4):
                r = (ripple_t - k * 0.18) * 520
                if r > 0:
                    circle(ctx, cx, cy, r, mix(WHITE, CYAN, k / 3), max(0, 0.7 - ripple_t * 0.4), width=3)
        draw_text(ctx, "示意动画", cx, 860, size=28, font="sans", weight=400, color=WHITE, alpha=0.5)
        # ---- real strain, revealed in slow motion
        tr = self.real_t(t)
        n = int(np.searchsorted(self.rt, tr))
        px0, px1, wy = 980.0, 1820.0, 330.0
        X = px0 + (self.rt - self.WIN[0]) / (self.WIN[1] - self.WIN[0]) * (px1 - px0)
        if n > 1:
            stroke_poly(ctx, np.stack([X[:n], wy - 95 * self.rl[:n]], 1), ORANGE, 1.8, 0.8)
            stroke_poly(ctx, np.stack([X[:n], wy - 95 * self.rh[:n]], 1), CYAN, 2.2, 0.95, glow=0.6)
            glow_dot(ctx, X[n - 1], wy - 95 * self.rh[n - 1], 4, WHITE, 1.0, 4)
        draw_text(ctx, "汉福德 H1", px0, wy - 140, size=28, font="sans", weight=500, color=CYAN, anchor="left")
        draw_text(ctx, "利文斯顿 L1", px0 + 200, wy - 140, size=28, font="sans", weight=500, color=ORANGE, anchor="left")
        draw_text(ctx, "真实数据 · 已白化", px1, wy - 140, size=28, font="sans", weight=400, color=WHITE, alpha=0.7,
                  anchor="right")
        # ---- wavelet time-frequency map of the real data
        fy0, fy1 = 520.0, 820.0
        m = self.tf.copy()
        m[:, n:] = 0
        vis = apply_lut(m[::-1], LUT_FIRE)
        paint_image(ctx, surface_from_array(vis), px0, fy0, px1 - px0, fy1 - fy0, 1.0, anchor="topleft")
        set_rgba(ctx, WHITE, 0.25)
        ctx.set_line_width(1)
        ctx.rectangle(px0, fy0, px1 - px0, fy1 - fy0)
        ctx.stroke()
        draw_text(ctx, "小波时频图：频率 35 → 250 Hz", px0, fy1 + 34, size=28, font="sans", weight=500, color=ORANGE,
                  anchor="left")
        vignette_title(ctx, lt, "引力波", "GW150914 · LIGO", None, y=150)

    def effects(self, t, lt):
        e = super().effects(t, lt)
        if t >= self.T_MERGE:
            e["flash"] = max(e.get("flash", 0), 0.5 * math.exp(-(t - self.T_MERGE) / 0.2))
            e["chroma"] = 4 * math.exp(-(t - self.T_MERGE) / 0.4)
        return e


# ---------------------------------------------------------------------- 5. wifi / OFDM
class WiFi(Vignette):
    def prepare(self):
        rng = np.random.default_rng(5)
        self.syms = rng.choice([0.35, 0.6, 0.8, 1.0], size=(64, 40))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        nsc = 24
        x0, x1, base = 260.0, 1660.0, 780.0
        df = (x1 - x0) / (nsc + 3)
        xs = np.linspace(x0 - 40, x1 + 40, 1400)
        sym_i = int((t - self.start) / 0.5)
        total = np.zeros_like(xs)
        for k in range(nsc):
            ap = ease_out_back((lt - 0.3 - k * 0.04) / 0.4)
            if ap <= 0:
                continue
            fc = x0 + (k + 2) * df
            amp = self.syms[k, sym_i % 40]
            y = np.sinc((xs - fc) / df) * amp * ap
            total += y
            col = hsv(k / nsc * 0.8, 0.75, 1.0)
            stroke_poly(ctx, np.stack([xs, base - 330 * y], 1), col, 2.0, 0.85)
            glow_dot(ctx, fc, base - 330 * amp * ap, 3.5, col, 0.9, 4)
        set_rgba(ctx, WHITE, 0.3)
        ctx.set_line_width(1)
        ctx.move_to(x0 - 40, base)
        ctx.line_to(x1 + 40, base)
        ctx.stroke()
        draw_text(ctx, "频率 →", x1 + 40, base + 30, size=28, font="sans", weight=500, color=WHITE, alpha=0.6,
                  anchor="right")
        a = smooth((lt - 1.1) / 0.5)
        draw_text(ctx, "每一种颜色，都是一路独立的数据", 960, base - 420, size=32, font="sans", weight=400,
                  color=(0.85, 0.9, 1.0), alpha=0.85 * a)
        draw_text(ctx, "峰值处，其余载波恰好为零 —— 正交", 960, base + 60, size=28, font="sans", weight=400,
                  color=GOLD, alpha=0.8 * smooth((lt - 1.7) / 0.5))
        vignette_title(ctx, lt, "Wi-Fi · 5G", "OFDM", "正交频分复用", y=150, layout="tr")


# ---------------------------------------------------------------------- 6. quantum
class Quantum(Vignette):
    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.04, 0.03, 0.10))
        sig = math.exp(0.95 * math.sin(2 * math.pi * (lt - 0.3) / 2.6))
        panels = [(560.0, "位置  x", CYAN, sig, 1), (1360.0, "动量  p", MAGENTA, 1 / sig, 0)]
        a = smooth(lt / 0.4)
        base = 700.0
        for cx, lab, col, s, show_phase in panels:
            xs = np.linspace(-1, 1, 500)
            u = xs * 4.0
            env = np.exp(-u ** 2 / (2 * (s * 0.55) ** 2)) / math.sqrt(s)
            X = cx + xs * 330
            if show_phase:
                re = env * np.cos(u * 9 - lt * 6)
                stroke_poly(ctx, np.stack([X, base - 230 * re * 0.8], 1), col, 1.4, 0.45 * a)
            dens = env ** 2
            pts = np.stack([X, base - 300 * dens * 0.8], 1)
            polyline(ctx, pts)
            ctx.line_to(X[-1], base)
            ctx.line_to(X[0], base)
            ctx.close_path()
            g = cairo.LinearGradient(0, base - 300, 0, base)
            g.add_color_stop_rgba(0, col[0], col[1], col[2], 0.45 * a)
            g.add_color_stop_rgba(1, col[0], col[1], col[2], 0.02)
            ctx.set_source(g)
            ctx.fill()
            stroke_poly(ctx, pts, col, 3.0, a, glow=1.0)
            set_rgba(ctx, WHITE, 0.3 * a)
            ctx.set_line_width(1)
            ctx.move_to(cx - 340, base)
            ctx.line_to(cx + 340, base)
            ctx.stroke()
            # width bar
            wbar = s * 0.55 / 4.0 * 330 * 2
            set_rgba(ctx, col, 0.9 * a)
            ctx.set_line_width(2)
            ctx.move_to(cx - wbar / 2, base + 30)
            ctx.line_to(cx + wbar / 2, base + 30)
            for sx in (-1, 1):
                ctx.move_to(cx + sx * wbar / 2, base + 22)
                ctx.line_to(cx + sx * wbar / 2, base + 38)
            ctx.stroke()
            draw_text(ctx, lab, cx, base + 72, size=32, font="sans", weight=500, color=col, alpha=a)
        draw_math(ctx, r"$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$", 960, 330, size=48, color=WHITE,
                  alpha=smooth((lt - 0.5) / 0.5), glow=8)
        draw_text(ctx, "量子计算：Shor 算法的核心，正是“量子傅里叶变换”", 960, 400, size=30, font="sans",
                  weight=400, color=GOLD, alpha=0.85 * smooth((lt - 1.4) / 0.5))
        arrow(ctx, 890, 520, 1030, 520, GOLD, 0.8 * a, 2, 14)
        arrow(ctx, 1030, 545, 890, 545, GOLD, 0.8 * a, 2, 14)
        draw_math(ctx, r"$\mathcal{F}$", 960, 480, size=34, color=GOLD, alpha=a)
        vignette_title(ctx, lt, "量子", "QUANTUM", "不确定性原理", layout="v")


# ---------------------------------------------------------------------- 7. AI positional encoding
class AIPos(Vignette):
    def prepare(self):
        npos, d = 48, 128
        pos = np.arange(npos)[:, None]
        i = np.arange(d // 2)[None, :]
        ang = pos / (10000 ** (2 * i / d))
        pe = np.zeros((npos, d))
        pe[:, 0::2] = np.sin(ang)
        pe[:, 1::2] = np.cos(ang)
        self.pe = pe

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        npos, d = self.pe.shape
        x0, y0, cw, ch = 640.0, 300.0, 7.5, 11.0
        rev = ease_out((lt - 0.2) / 1.3, 2)
        ncol = int(rev * d)
        vis = apply_lut(np.clip(self.pe[:, :max(1, ncol)] * 0.5 + 0.5, 0, 1), LUT_DIVERGE) * 0.72
        paint_image(ctx, surface_from_array(vis), x0, y0, cw * max(1, ncol), ch * npos, 1.0, anchor="topleft",
                    filt=cairo.FILTER_NEAREST)
        set_rgba(ctx, (0.5, 0.7, 1.0), 0.3)
        ctx.set_line_width(1)
        ctx.rectangle(x0 - 4, y0 - 4, cw * d + 8, ch * npos + 8)
        ctx.stroke()
        draw_text(ctx, "维度 → 频率由高到低", x0 + cw * d / 2, y0 - 30, size=28, font="sans", weight=500,
                  color=WHITE, alpha=0.7)
        # tokens
        toks = list("我们用波理解世界")
        hi = int((lt - 0.9) / 0.3)
        for k, tok in enumerate(toks):
            a = smooth((lt - 0.4 - k * 0.08) / 0.25)
            y = y0 + (k * 6 + 3) * ch
            on = 0 <= hi and k == hi % len(toks)
            rounded_rect(ctx, 430, y - 22, 56, 44, 8)
            set_rgba(ctx, GOLD if on else (0.3, 0.4, 0.7), (0.9 if on else 0.35) * a)
            ctx.fill()
            draw_text(ctx, tok, 458, y, size=34, font="sans", weight=600, color=(0.05, 0.05, 0.1) if on else WHITE,
                      alpha=a)
            ctx.move_to(495, y)
            ctx.line_to(x0 - 10, y)
            set_rgba(ctx, GOLD if on else WHITE, (0.9 if on else 0.2) * a)
            ctx.set_line_width(2 if on else 1)
            ctx.stroke()
            if on:
                set_rgba(ctx, GOLD, 0.9)
                ctx.set_line_width(2)
                ctx.rectangle(x0 - 2, y - ch / 2 - 1, cw * ncol + 4, ch + 2)
                ctx.stroke()
        # a few of the underlying sines
        for j, (col, c) in enumerate([(0, CYAN), (20, VIOLET), (60, MAGENTA)]):
            a = smooth((lt - 1.0 - j * 0.2) / 0.3)
            ys = np.linspace(y0, y0 + ch * npos, 200)
            p = (ys - y0) / ch
            ang = p / (10000 ** (2 * (col // 2) / d))
            stroke_poly(ctx, np.stack([1690 + j * 75 + 26 * np.sin(ang), ys], 1), c, 2.0, 0.9 * a)
        vignette_title(ctx, lt, "人工智能", "TRANSFORMER · 2017", "位置编码", y=150)


# ---------------------------------------------------------------------- 8. FFT
class FFT(Vignette):
    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        n, stages = 16, 4
        x0, x1, y0, y1 = 400.0, 1120.0, 300.0, 840.0
        xs = np.linspace(x0, x1, stages + 1)
        ys = np.linspace(y0, y1, n)
        grow = ease_out(lt / 0.8, 2)
        # edges
        for s in range(stages):
            span = n >> (s + 1)
            for i in range(n):
                j = i ^ span
                for (a_, b_) in [(i, i), (i, j)]:
                    ctx.move_to(xs[s], ys[a_])
                    ctx.line_to(lerp(xs[s], xs[s + 1], grow), lerp(ys[a_], ys[b_], grow))
                    set_rgba(ctx, (0.4, 0.55, 1.0), 0.28)
                    ctx.set_line_width(1.2)
                    ctx.stroke()
        # pulses
        period = 0.6
        ph = (lt % period) / period
        s = int(lt / period) % stages
        for i in range(n):
            span = n >> (s + 1)
            j = i ^ span
            for b_ in (i, j):
                px = lerp(xs[s], xs[s + 1], ph)
                py = lerp(ys[i], ys[b_], ph)
                glow_dot(ctx, px, py, 3.5, hsv(0.55 + 0.3 * i / n, 0.7, 1.0), 0.9, 5)
        for s in range(stages + 1):
            for i in range(n):
                circle(ctx, xs[s], ys[i], 5, WHITE, 0.8 * grow, fill=True)
        # comparison
        a = smooth((lt - 0.6) / 0.4)
        bx = 1260.0
        draw_text(ctx, "100 万个数据点", bx, 330, size=34, font="sans", weight=500, color=WHITE, alpha=a, anchor="left")
        draw_text(ctx, "直接计算", bx, 420, size=30, font="sans", weight=400, color=(0.8, 0.85, 1.0), alpha=a,
                  anchor="left")
        draw_math(ctx, r"$N^2 \approx 10^{12}$", bx + 330, 420, size=34, color=RED, alpha=a)
        draw_text(ctx, "FFT", bx, 510, size=32, font="latin", weight=700, color=GOLD, alpha=a, anchor="left")
        draw_math(ctx, r"$N\log_2 N \approx 2\times10^{7}$", bx + 330, 510, size=34, color=GOLD, alpha=a)
        b = smooth((lt - 1.2) / 0.4)
        L1 = 520 * b
        set_rgba(ctx, RED, 0.8)
        ctx.rectangle(bx, 580, L1, 16)
        ctx.fill()
        set_rgba(ctx, GOLD, 0.95)
        ctx.rectangle(bx, 620, max(2, L1 / 50000 * 50), 16)
        ctx.fill()
        draw_text(ctx, "快 50000 倍", bx, 700, size=40, font="sans", weight=700, color=GOLD,
                  alpha=smooth((lt - 1.7) / 0.4), anchor="left", glow=8, glow_alpha=0.4)
        draw_text(ctx, "高斯 · 1805（手稿，1866 年才出版）", bx, 770, size=28, font="sans", weight=400,
                  color=(0.85, 0.9, 1.0), alpha=0.8 * smooth((lt - 2.2) / 0.4), anchor="left")
        vignette_title(ctx, lt, "FFT", "COOLEY–TUKEY · 1965", "快速傅里叶变换", y=150, layout="tr")


# ---------------------------------------------------------------------- Fourier optics: Webb's diffraction spikes
class Optics(Vignette):
    def prepare(self):
        N = 512
        yy, xx = np.mgrid[0:N, 0:N] - N / 2
        ap = np.zeros((N, N), np.float32)
        d = 22.0  # hexagon flat-to-flat (px)

        def hexmask(cx, cy):
            x, y = np.abs(xx - cx), np.abs(yy - cy)
            r = d / 2 * 0.96
            return (y <= r) & (x * math.sqrt(3) / 2 + y / 2 <= r)

        centres = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                ring = max(abs(q), abs(r), abs(s))
                if 1 <= ring <= 2:
                    cx = d * (q + r / 2) * 1.0
                    cy = d * math.sqrt(3) / 2 * r
                    centres.append((cy, cx))
        for cx, cy in centres:
            ap[hexmask(cx, cy)] = 1.0
        ap[(np.abs(xx) < 1.2) & (yy < 0)] = 0.0  # a strut
        self.ap = ap
        F = (np.abs(np.fft.fftshift(np.fft.fft2(ap))) ** 2).astype(np.float32)
        # broadband starlight: average the pattern over a range of wavelengths (radial rescaling)
        acc = np.zeros_like(F)
        for sc in np.linspace(0.6, 1.4, 24):
            M = cv2.getRotationMatrix2D((N / 2, N / 2), 0, sc)
            acc += cv2.warpAffine(F, M, (N, N), flags=cv2.INTER_LINEAR)
        psf = np.log1p(acc / acc.max() * 1e7)
        psf = psf / psf.max()
        c = psf[N // 2 - 160:N // 2 + 160, N // 2 - 160:N // 2 + 160]
        self.psf = np.clip((c - 0.3) / 0.7, 0, 1) ** 1.3
        crop = ap[N // 2 - 80:N // 2 + 80, N // 2 - 80:N // 2 + 80]
        self.ap_vis = crop

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.3)
        if not hasattr(self, "hst"):
            from ..data import image
            xdf = image("hubble_deep_field", None, gray=False)[..., :3]
            self.hst = xdf[578 - 60:578 + 60, 755 - 60:755 + 60]
        # left column: Webb mirror (aperture) and Hubble's real star
        a = smooth(lt / 0.35)
        vis = apply_lut(self.ap_vis * 0.85, LUT_GOLD)
        paint_image(ctx, surface_from_array(vis), 430, 470, 340, 340, a)
        draw_text(ctx, "韦布：18 块六边形镜面", 430, 680, size=30, font="sans", weight=500, color=GOLD, alpha=a)
        b = ease_out((lt - 0.4) / 0.4, 2)
        if b > 0:
            arrow(ctx, 640, 470, lerp(640, 820, b), 470, WHITE, 0.8 * b, 2.5, 16)
            draw_math(ctx, r"$|\mathcal{F}|^2$", 730, 420, size=38, color=WHITE, alpha=b)
        r = smooth((lt - 0.7) / 0.6)
        if r > 0:
            vis = apply_lut(self.psf * r, LUT_ICE)
            paint_image(ctx, surface_from_array(vis), 1080, 470, 460, 460, 1.0)
            draw_text(ctx, "算出来的星芒：6 + 2 道", 1080, 730, size=30, font="sans", weight=500, color=CYAN, alpha=r)
        c = smooth((lt - 1.4) / 0.5)
        if c > 0:
            paint_image(ctx, surface_from_array(self.hst.astype(np.float32) / 255.0), 1620, 470, 340, 340, c,
                        filt=cairo.FILTER_BEST)
            set_rgba(ctx, WHITE, 0.3 * c)
            ctx.set_line_width(1.2)
            ctx.rectangle(1450, 300, 340, 340)
            ctx.stroke()
            draw_text(ctx, "哈勃真实照片：十字支架", 1620, 680, size=30, font="sans", weight=500, color=WHITE, alpha=c)
            draw_text(ctx, "→ 4 道星芒", 1620, 722, size=30, font="sans", weight=500, color=WHITE, alpha=c)
        vignette_title(ctx, lt, "傅里叶光学", "FOURIER OPTICS", "透镜以光速完成傅里叶变换", y=130)


# ---------------------------------------------------------------------- spherical harmonics
class SphHarm(Vignette):
    L = 3

    def prepare(self):
        from scipy.special import sph_harm_y
        self.shapes = {}
        th = np.linspace(0, np.pi, 25)
        ph = np.linspace(0, 2 * np.pi, 33)
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        for l in range(self.L + 1):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, abs(m), TH, PH)
                if m < 0:
                    Yr = math.sqrt(2) * (-1) ** m * Y.imag
                elif m > 0:
                    Yr = math.sqrt(2) * (-1) ** m * Y.real
                else:
                    Yr = Y.real
                R = np.abs(Yr)
                R = R / (R.max() + 1e-9)
                P = np.stack([R * np.sin(TH) * np.cos(PH), R * np.cos(TH), R * np.sin(TH) * np.sin(PH)], -1)
                self.shapes[(l, m)] = (P, np.sign(Yr))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.03, 0.03, 0.10))
        yaw = 0.6 + lt * 0.7
        size = 105.0
        for l in range(self.L + 1):
            y = 250 + l * 185
            for m in range(-l, l + 1):
                x = 1030 + m * 190
                a = smooth((lt - 0.1 * l - 0.03 * abs(m)) / 0.35)
                if a <= 0:
                    continue
                cam = Camera(yaw, 0.35, 5.2, center=(x, y), fov=30)
                cam.focal *= size / (H / 2) * 0.95
                P, sg = self.shapes[(l, m)]
                for i in range(0, P.shape[0], 2):
                    row, sr = P[i], sg[i]
                    for j in range(0, P.shape[1] - 1, 2):
                        k = min(P.shape[1], j + 3)
                        col = CYAN if sr[j] >= 0 else MAGENTA
                        line3(ctx, cam, row[j:k], col, 1.1, 0.8 * a)
                for j in range(0, P.shape[1], 3):
                    col_line = P[:, j]
                    s_line = sg[:, j]
                    for i in range(0, P.shape[0] - 1, 2):
                        k = min(P.shape[0], i + 3)
                        col = CYAN if s_line[i] >= 0 else MAGENTA
                        line3(ctx, cam, col_line[i:k], col, 1.1, 0.8 * a)
            draw_text(ctx, f"l = {l}", 250, y, size=30, font="latin", weight=500, color=WHITE, alpha=0.7)
        vignette_title(ctx, lt, "球谐函数", "SPHERICAL HARMONICS", "球面上的傅里叶级数", y=110, layout="tr")


# ---------------------------------------------------------------------- tides
class Tides(Vignette):
    COMP = [("M2", 12.42, 1.00), ("S2", 12.00, 0.46), ("N2", 12.66, 0.19), ("K1", 23.93, 0.58), ("O1", 25.82, 0.41)]
    DAYS = 30.0

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.02, 0.05, 0.09))
        u1 = smoother((lt - 0.4) / 1.6)
        cam = Camera(lerp(-0.7, -0.05, u1), lerp(0.45, 0.08, u1), 17.0, target=(0, 0.6, lerp(3.0, 0.0, u1)), fov=36)
        hrs = np.linspace(0, self.DAYS * 24, 1800)
        X = (hrs / (self.DAYS * 24) - 0.5) * 12.0
        total = np.zeros_like(hrs)
        collapse = u1
        for i, (nm, per, amp) in enumerate(self.COMP):
            y = amp * np.cos(2 * np.pi * hrs / per + i)
            total += y
            z = lerp(1.4 * (i + 1), 0.0, collapse)
            col = hsv(0.5 + 0.08 * i, 0.7, 1.0)
            line3(ctx, cam, np.stack([X, y * 0.8, np.full_like(X, z)], 1), col, 1.4, 0.7 * (1 - 0.8 * collapse))
            px, py, _ = cam.project(np.array([[6.3, 0, z]]))
            draw_text(ctx, f"{nm}  {per:.2f} h", px[0] + 10, py[0], size=28, font="mono", weight=500, color=col,
                      alpha=0.9 * (1 - collapse), anchor="left")
        line3(ctx, cam, np.stack([X, total * 0.8, np.zeros_like(X)], 1), WHITE, 2.4, 0.4 + 0.6 * collapse)
        a = smooth((lt - 1.9) / 0.4)
        if a > 0:
            # spring tides where M2 and S2 line up, neap tides half a beat later
            dw = 2 * np.pi * (1 / 12.00 - 1 / 12.42)
            beat = 2 * np.pi / dw / 24
            d0 = ((2 * np.pi - 1.0) % (2 * np.pi)) / dw / 24  # phase offset of S2 is 1 rad
            marks = []
            for k in range(3):
                marks += [(d0 + k * beat, "大潮"), (d0 + (k + 0.5) * beat, "小潮")]
            marks = [(d, lab) for d, lab in marks if 0.5 < d < self.DAYS - 0.5]
            for d, lab in marks:
                x = (d / self.DAYS - 0.5) * 12.0
                px, py, _ = cam.project(np.array([[x, 2.4, 0]]))
                draw_text(ctx, lab, px[0], py[0], size=30, font="sans", weight=600, color=GOLD, alpha=a)
            draw_text(ctx, "30 天", 1700, 760, size=28, font="sans", weight=400, color=WHITE, alpha=0.6 * a,
                      anchor="right")
        vignette_title(ctx, lt, "潮汐", "TIDES · KELVIN 1872", "开尔文潮汐预测机：用滑轮把正弦波加起来", y=110, layout="tc")


def beyond_scenes():
    cls = {"prism": Prism, "optics": Optics, "dna": DNA, "mri": MRI, "gw": GW, "sph": SphHarm, "tides": Tides,
           "wifi": WiFi, "quantum": Quantum, "ai": AIPos, "fft": FFT}
    out = [BeyondHeader()]
    for name, t0 in S.VIGNETTES:
        out.append(cls[name](t0))
    return out
