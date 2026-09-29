"""'And beyond': eight short vignettes of the Fourier transform at work."""
import math

import cairo
import cv2
import numpy as np

from .. import story as S
from ..core import (BLUE, CYAN, GOLD, H, LUT_DIVERGE, LUT_FIRE, LUT_GOLD, LUT_GRAY, LUT_ICE, MAGENTA, ORANGE, RED,
                    TEAL, VIOLET, W, WHITE, apply_lut, arrow, background, circle, clamp, draw_math, draw_text,
                    draw_text_chars, ease_out, ease_out_back, ease_out_expo, glow_dot, hsv, lerp, mix, paint_image,
                    polyline, remap, rounded_rect, set_rgba, smooth, stroke_poly, surface_from_array, window)
from ..data import beat_pulse, brain
from .base import Scene


def vignette_title(ctx, lt, zh, en, sub=None, x=110, y=150):
    a = smooth(lt / 0.4)
    draw_text_chars(ctx, zh, x, y, lt, size=66, font="serif", weight=800, color=WHITE, alpha=a, anchor="left",
                    tracking=0.12, stagger=0.06, dur=0.35, rise=12, glow=10, glow_alpha=0.35)
    draw_text(ctx, en, x + 3, y + 56, size=15, font="latin", weight=600, color=GOLD, alpha=0.85 * smooth((lt - 0.2) / 0.5),
              anchor="left", tracking=0.45)
    if sub:
        draw_text(ctx, sub, x + 2, y + 98, size=22, font="sans", weight=400, color=(0.85, 0.9, 1.0),
                  alpha=0.8 * smooth((lt - 0.5) / 0.5), anchor="left", tracking=0.06)


class Vignette(Scene):
    fade_in = 0.25
    fade_out = 0.25
    bloom = 0.9

    def __init__(self, start):
        self.start = start
        self.end = start + 6.0

    def effects(self, t, lt):
        return {"flash": 0.12 * math.exp(-max(0.0, lt) / 0.15)}


# ---------------------------------------------------------------------- header
class BeyondHeader(Scene):
    start, end = 144.0, 148.0
    fade_out = 0.25
    bloom = 1.0

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, strength=1.3, hue=(0.05, 0.04, 0.12))
        # radial burst of sine rays
        n = 36
        for i in range(n):
            ang = i / n * 2 * math.pi + 0.05 * lt
            L0 = 140 + 40 * math.sin(i * 1.7)
            L1 = L0 + 900 * ease_out(lt / 1.6, 3)
            r = np.linspace(L0, L1, 160)
            amp = 10 * np.sin(r * 0.05 - lt * 8 + i) * np.exp(-(r - L0) / 600)
            xs = W / 2 + r * math.cos(ang) - amp * math.sin(ang)
            ys = H / 2 + r * math.sin(ang) + amp * math.cos(ang)
            col = hsv(i / n, 0.7, 1.0)
            stroke_poly(ctx, np.stack([xs, ys], 1), col, 1.6, 0.5 * (1 - smooth((lt - 2.8) / 1.0)))
        s = 1.25 - 0.25 * ease_out_expo(lt / 1.2)
        draw_text(ctx, "不止于此", W / 2, H / 2 - 30, size=150, font="serif", weight=900, color=WHITE,
                  alpha=smooth(lt / 0.2), tracking=0.3, glow=18, glow_alpha=0.5, scale=s)
        draw_text(ctx, "AND  BEYOND", W / 2, H / 2 + 90, size=24, font="latin", weight=300, color=GOLD, alpha=0.9,
                  tracking=0.9, reveal=ease_out((lt - 0.6) / 1.2, 2))
        draw_text_chars(ctx, "从星辰到基因，从大脑到人工智能", W / 2, H / 2 + 170, lt - 1.3, size=30, font="sans",
                        weight=400, color=(0.85, 0.9, 1.0), alpha=0.9, tracking=0.2, stagger=0.05)

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


class Prism(Vignette):
    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        cx, cy, side = 760.0, 500.0, 330.0
        hgt = side * math.sqrt(3) / 2
        A = (cx, cy - hgt * 2 / 3)
        B = (cx - side / 2, cy + hgt / 3)
        C = (cx + side / 2, cy + hgt / 3)
        # entry and exit points
        pin = (lerp(A[0], B[0], 0.55), lerp(A[1], B[1], 0.55))
        pout_mid = (lerp(A[0], C[0], 0.58), lerp(A[1], C[1], 0.58))
        beam = ease_out(lt / 0.8, 2)
        src = (0.0, pin[1] - 150)
        bx = lerp(src[0], pin[0], beam)
        by = lerp(src[1], pin[1], beam)
        for wdt, al in [(18, 0.06), (8, 0.15), (3, 0.9)]:
            ctx.move_to(*src)
            ctx.line_to(bx, by)
            set_rgba(ctx, WHITE, al)
            ctx.set_line_width(wdt)
            ctx.stroke()
        # prism body
        polyline(ctx, [A, B, C], close=True)
        g = cairo.LinearGradient(B[0], B[1], C[0], A[1])
        g.add_color_stop_rgba(0, 0.5, 0.7, 1.0, 0.10)
        g.add_color_stop_rgba(1, 0.8, 0.9, 1.0, 0.22)
        ctx.set_source(g)
        ctx.fill_preserve()
        set_rgba(ctx, (0.8, 0.9, 1.0), 0.8)
        ctx.set_line_width(2)
        ctx.stroke()
        fan = ease_out((lt - 0.7) / 0.8, 2)
        screen_x, y0, y1 = 1500.0, 250.0, 780.0
        if fan > 0:
            nr = 90
            for i in range(nr):
                u = i / (nr - 1)
                wl = 700 - 300 * u
                col = wl_rgb(wl)
                # inside the prism
                pm = (pout_mid[0] + (u - 0.5) * 18, pout_mid[1] + (u - 0.5) * 30)
                ctx.move_to(*pin)
                ctx.line_to(*pm)
                set_rgba(ctx, col, 0.08)
                ctx.set_line_width(2)
                ctx.stroke()
                ty = lerp(y0, y1, u)
                ex = lerp(pm[0], screen_x, fan)
                ey = lerp(pm[1], ty, fan)
                ctx.move_to(*pm)
                ctx.line_to(ex, ey)
                set_rgba(ctx, col, 0.10)
                ctx.set_line_width(4)
                ctx.stroke()
        # spectrum strip with absorption lines
        st = smooth((lt - 1.4) / 0.6)
        if st > 0:
            n = 300
            for i in range(n):
                u = i / n
                wl = 700 - 300 * u
                set_rgba(ctx, wl_rgb(wl), st)
                ctx.rectangle(screen_x, lerp(y0, y1, u), 70, (y1 - y0) / n + 1)
                ctx.fill()
            for k, (wl, el) in enumerate(FRAUNHOFER):
                a = smooth((lt - 2.3 - 0.25 * k) / 0.3)
                if a <= 0:
                    continue
                u = (700 - wl) / 300
                y = lerp(y0, y1, u)
                set_rgba(ctx, (0, 0, 0), 0.92 * a)
                ctx.rectangle(screen_x - 2, y - 3.5, 74, 7)
                ctx.fill()
                ctx.move_to(screen_x + 80, y)
                ctx.line_to(screen_x + 110, y)
                set_rgba(ctx, WHITE, 0.5 * a)
                ctx.set_line_width(1)
                ctx.stroke()
                draw_text(ctx, el, screen_x + 118, y, size=22, font="latin", weight=600, color=WHITE, alpha=0.95 * a,
                          anchor="left")
                draw_text(ctx, f"{wl:.0f} nm", screen_x + 118 + 18 * len(el) + 16, y + 1, size=13, font="mono",
                          weight=300, color=WHITE, alpha=0.5 * a, anchor="left")
        vignette_title(ctx, lt, "光", "LIGHT", "棱镜：把阳光按频率展开")


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
        bx = ease_out((lt - 0.6) / 0.6, 2)
        if bx > 0:
            ctx.move_to(740, 520)
            ctx.line_to(lerp(740, 1150, bx), 520)
            set_rgba(ctx, (0.7, 0.8, 1.0), 0.5)
            ctx.set_line_width(3)
            ctx.stroke()
            arrow(ctx, 900, 520, lerp(900, 1110, bx), 520, WHITE, 0.6 * bx, 2, 14)
            draw_text(ctx, "X 射线", 925, 490, size=20, font="sans", weight=500, color=WHITE, alpha=0.8 * bx, anchor="left")
        rev = smooth((lt - 1.2) / 1.2)
        if rev > 0:
            vis = apply_lut(self.diff * rev, LUT_GOLD)
            surf = surface_from_array(vis)
            paint_image(ctx, surf, 1400, 520, 440, 440, 1.0)
            circle(ctx, 1400, 520, 222, (0.8, 0.7, 0.5), 0.35 * rev, width=1.5)
            draw_text(ctx, "Photo 51 · 1952", 1400, 780, size=22, font="latin", weight=500, color=GOLD, alpha=rev,
                      tracking=0.1)
            draw_text(ctx, "罗莎琳德·富兰克林", 1400, 812, size=18, font="sans", weight=400, color=WHITE,
                      alpha=0.6 * rev)
        vignette_title(ctx, lt, "DNA", "DOUBLE HELIX", "X 射线衍射图 = 分子的傅里叶变换")


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
        prog = clamp((lt - 0.6) / 4.2) ** 1.5
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
        draw_text(ctx, "k 空间（频率）", kx, y + s / 2 + 40, size=22, font="sans", weight=500, color=CYAN, alpha=a)
        draw_text(ctx, "大脑图像", ix, y + s / 2 + 40, size=22, font="sans", weight=500, color=WHITE, alpha=a)
        vignette_title(ctx, lt, "核磁共振", "MRI", "扫描仪测量的，其实是频率", x=110, y=110)


# ---------------------------------------------------------------------- 4. gravitational waves
class GW(Vignette):
    T_MERGE = 168.5
    T_START = 165.5

    def f_of(self, t):
        tm = self.T_MERGE - self.T_START
        tau = max(self.T_MERGE - t, 1e-3)
        return min(420.0, 55 * (tau / tm) ** (-3 / 8)) if t < self.T_MERGE else 420.0

    def prepare(self):
        ts = np.arange(self.T_START - 1.0, self.T_MERGE + 3.0, 0.002)
        f = np.array([self.f_of(x) if x >= self.T_START else 55.0 for x in ts])
        self.ts = ts
        self.fs = f
        self.ph = np.cumsum(f / 60.0 * 2 * np.pi * 0.002)  # slowed-down visual orbital phase
        amp = np.where(ts < self.T_MERGE, (f / 420) ** 0.9, np.exp(-(ts - self.T_MERGE) / 0.25))
        self.h = amp * np.cos(np.cumsum(f / 18.0 * 2 * np.pi * 0.002))  # slowed-down strain for display

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.02, 0.03, 0.08))
        cx, cy = 560.0, 540.0
        merged = t >= self.T_MERGE
        f = self.f_of(t)
        sep = 130 * (55.0 / f) ** (2 / 3) if not merged else 0.0
        ph = float(np.interp(t, self.ts, self.ph))
        # spacetime grid warped by ripples
        ripple_t = t - self.T_MERGE
        for gy in np.arange(200, 900, 36):
            xs = np.linspace(100, 1020, 120)
            d = np.hypot(xs - cx, gy - cy)
            k = 0.06
            wave = np.sin(d * k - (t * 9)) * np.exp(-d / 420) * (8 + 18 * (1 - min(1, sep / 130)))
            if merged:
                wave += 30 * np.sin(d * 0.05 - ripple_t * 14) * np.exp(-((d - ripple_t * 420) / 90) ** 2)
            pull = -2600 / (d + 60)
            ys = gy + wave + pull * (gy - cy) / (d + 1) * 0.6
            stroke_poly(ctx, np.stack([xs, ys], 1), BLUE, 1.2, 0.38)
        # black holes
        if not merged:
            for s in (0, 1):
                ang = ph + s * math.pi
                x = cx + sep / 2 * math.cos(ang)
                y = cy + sep / 2 * math.sin(ang) * 0.45
                circle(ctx, x, y, 44, ORANGE, 0.12, width=14)
                circle(ctx, x, y, 30, (0, 0, 0), 1.0, fill=True)
                circle(ctx, x, y, 32, GOLD, 0.9, width=3)
        else:
            u = ripple_t
            circle(ctx, cx, cy, 34, (0, 0, 0), 1.0, fill=True)
            circle(ctx, cx, cy, 36, GOLD, 0.9, width=3)
            for k in range(4):
                r = (u - k * 0.18) * 520
                if r > 0:
                    circle(ctx, cx, cy, r, mix(WHITE, CYAN, k / 3), max(0, 0.7 - u * 0.4), width=3)
        # waveform panel
        px0, px1, wy = 1080.0, 1820.0, 360.0
        span = 3.0
        tt = np.linspace(t - span, t, 700)
        hh = np.interp(tt, self.ts, self.h)
        xs = np.linspace(px0, px1, len(tt))
        stroke_poly(ctx, np.stack([xs, wy - 90 * hh], 1), CYAN, 2.0, 0.95, glow=0.8)
        draw_text(ctx, "应变 h(t)", px0, wy - 130, size=18, font="sans", weight=500, color=CYAN, alpha=0.8, anchor="left")
        # time-frequency panel
        fy0, fy1 = 520.0, 800.0
        set_rgba(ctx, (0.05, 0.08, 0.2), 0.8)
        ctx.rectangle(px0, fy0, px1 - px0, fy1 - fy0)
        ctx.fill()
        ff = np.interp(tt, self.ts, self.fs)
        vis = tt >= self.T_START
        yy = fy1 - (np.log(ff) - np.log(40)) / (np.log(500) - np.log(40)) * (fy1 - fy0)
        amp = np.interp(tt, self.ts, np.where(self.ts < self.T_MERGE, (self.fs / 420) ** 0.9, 0))
        for i in range(0, len(tt) - 1, 3):
            if not vis[i] or amp[i] <= 0.01:
                continue
            ctx.move_to(xs[i], yy[i])
            ctx.line_to(xs[i + 3] if i + 3 < len(xs) else xs[-1], yy[i + 3] if i + 3 < len(yy) else yy[-1])
            set_rgba(ctx, mix(ORANGE, WHITE, amp[i]), 0.9 * amp[i] + 0.1)
            ctx.set_line_width(4 + 6 * amp[i])
            ctx.stroke()
        draw_text(ctx, "频率 ↑", px0, fy0 - 26, size=18, font="sans", weight=500, color=ORANGE, alpha=0.8, anchor="left")
        draw_text(ctx, "“啁啾”", px1, fy0 - 26, size=20, font="sans", weight=500, color=WHITE, alpha=0.8, anchor="right")
        vignette_title(ctx, lt, "引力波", "GRAVITATIONAL WAVES", "2015 · LIGO", y=150)

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
        draw_text(ctx, "频率 →", x1 + 40, base + 30, size=18, font="sans", weight=500, color=WHITE, alpha=0.6,
                  anchor="right")
        a = smooth((lt - 1.8) / 0.6)
        draw_text(ctx, "每一种颜色，都是一路独立的数据", 960, base - 420, size=24, font="sans", weight=400,
                  color=(0.85, 0.9, 1.0), alpha=0.85 * a)
        draw_text(ctx, "峰值处，其余载波恰好为零 —— 正交", 960, base + 60, size=18, font="sans", weight=400,
                  color=GOLD, alpha=0.8 * smooth((lt - 2.6) / 0.6))
        vignette_title(ctx, lt, "Wi-Fi · 5G", "OFDM", "正交频分复用", y=150)


# ---------------------------------------------------------------------- 6. quantum
class Quantum(Vignette):
    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.04, 0.03, 0.10))
        sig = math.exp(0.95 * math.sin(2 * math.pi * (lt - 0.6) / 3.2))
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
            draw_text(ctx, lab, cx, base + 72, size=24, font="sans", weight=500, color=col, alpha=a)
        draw_math(ctx, r"$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$", 960, 330, size=48, color=WHITE,
                  alpha=smooth((lt - 1.0) / 0.6), glow=8)
        arrow(ctx, 890, 520, 1030, 520, GOLD, 0.8 * a, 2, 14)
        arrow(ctx, 1030, 545, 890, 545, GOLD, 0.8 * a, 2, 14)
        draw_math(ctx, r"$\mathcal{F}$", 960, 480, size=34, color=GOLD, alpha=a)
        vignette_title(ctx, lt, "量子", "QUANTUM", "不确定性原理", y=150)


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
        rev = ease_out((lt - 0.4) / 1.8, 2)
        ncol = int(rev * d)
        vis = apply_lut(np.clip(self.pe[:, :max(1, ncol)] * 0.5 + 0.5, 0, 1), LUT_DIVERGE) * 0.72
        paint_image(ctx, surface_from_array(vis), x0, y0, cw * max(1, ncol), ch * npos, 1.0, anchor="topleft",
                    filt=cairo.FILTER_NEAREST)
        set_rgba(ctx, (0.5, 0.7, 1.0), 0.3)
        ctx.set_line_width(1)
        ctx.rectangle(x0 - 4, y0 - 4, cw * d + 8, ch * npos + 8)
        ctx.stroke()
        draw_text(ctx, "维度 → 频率由高到低", x0 + cw * d / 2, y0 - 30, size=18, font="sans", weight=500,
                  color=WHITE, alpha=0.7)
        # tokens
        toks = list("傅里叶改变了世界")
        hi = int((lt - 1.2) / 0.35)
        for k, tok in enumerate(toks):
            a = smooth((lt - 0.8 - k * 0.12) / 0.3)
            y = y0 + (k * 6 + 3) * ch
            on = 0 <= hi and k == hi % len(toks)
            rounded_rect(ctx, 430, y - 22, 56, 44, 8)
            set_rgba(ctx, GOLD if on else (0.3, 0.4, 0.7), (0.9 if on else 0.35) * a)
            ctx.fill()
            draw_text(ctx, tok, 458, y, size=28, font="sans", weight=600, color=(0.05, 0.05, 0.1) if on else WHITE,
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
            a = smooth((lt - 1.6 - j * 0.3) / 0.4)
            ys = np.linspace(y0, y0 + ch * npos, 200)
            p = (ys - y0) / ch
            ang = p / (10000 ** (2 * (col // 2) / d))
            stroke_poly(ctx, np.stack([1690 + j * 75 + 26 * np.sin(ang), ys], 1), c, 2.0, 0.9 * a)
        vignette_title(ctx, lt, "人工智能", "TRANSFORMER", "位置编码", y=150)


# ---------------------------------------------------------------------- 8. FFT
class FFT(Vignette):
    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        n, stages = 16, 4
        x0, x1, y0, y1 = 400.0, 1120.0, 300.0, 840.0
        xs = np.linspace(x0, x1, stages + 1)
        ys = np.linspace(y0, y1, n)
        grow = ease_out(lt / 1.2, 2)
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
        period = 1.0
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
        a = smooth((lt - 1.0) / 0.5)
        bx = 1260.0
        draw_text(ctx, "100 万个数据点", bx, 330, size=26, font="sans", weight=500, color=WHITE, alpha=a, anchor="left")
        draw_text(ctx, "直接计算", bx, 420, size=22, font="sans", weight=400, color=(0.8, 0.85, 1.0), alpha=a,
                  anchor="left")
        draw_math(ctx, r"$N^2 \approx 10^{12}$", bx + 330, 420, size=34, color=RED, alpha=a)
        draw_text(ctx, "FFT", bx, 510, size=24, font="latin", weight=700, color=GOLD, alpha=a, anchor="left")
        draw_math(ctx, r"$N\log_2 N \approx 2\times10^{7}$", bx + 330, 510, size=34, color=GOLD, alpha=a)
        b = smooth((lt - 2.0) / 0.5)
        L1 = 520 * b
        set_rgba(ctx, RED, 0.8)
        ctx.rectangle(bx, 580, L1, 16)
        ctx.fill()
        set_rgba(ctx, GOLD, 0.95)
        ctx.rectangle(bx, 620, max(2, L1 / 50000 * 50), 16)
        ctx.fill()
        draw_text(ctx, "快 50000 倍", bx, 700, size=40, font="sans", weight=700, color=GOLD,
                  alpha=smooth((lt - 2.6) / 0.4), anchor="left", glow=8, glow_alpha=0.4)
        vignette_title(ctx, lt, "FFT", "COOLEY–TUKEY · 1965", "快速傅里叶变换", y=150)


def beyond_scenes():
    cls = {"prism": Prism, "dna": DNA, "mri": MRI, "gw": GW, "wifi": WiFi, "quantum": Quantum, "ai": AIPos, "fft": FFT}
    out = [BeyondHeader()]
    for name, t0 in S.VIGNETTES:
        sc = cls[name](t0)
        out.append(sc)
    out[-1].fade_out = 0.6
    return out
