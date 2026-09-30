"""Beyond Fourier: the Laplace transform over the s-plane, and wavelets vs. the short-time Fourier transform."""
import math

import cairo
import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, RED, VIOLET, W, WHITE, background, clamp, draw_math, draw_text, ease_out,
                    glow_dot, hsv, lerp, mix, polyline, rounded_rect, set_rgba, smooth, smoother, stroke_poly, window)
from ..three import Camera, axis3, line3, solid_surface
from .base import Scene


class Laplace3D(Scene):
    """|H(s)| of a resonant second-order system over the s-plane; the jw-axis slice is the Fourier transform."""
    start, end = S.T_LAPLACE, S.T_WAVELET
    fade_in = 0.3
    fade_out = 0.3
    bloom = 1.0
    W0, ZETA = 2.0, 0.2

    def tf(self, s):
        return self.W0 ** 2 / (s * s + 2 * self.ZETA * self.W0 * s + self.W0 ** 2)

    def prepare(self):
        sig = np.linspace(-2.0, 1.0, 46)
        om = np.linspace(-4.0, 4.0, 70)
        SG, OM = np.meshgrid(sig, om, indexing="ij")
        mag = np.log10(np.abs(self.tf(SG + 1j * OM)) + 1e-9)
        self.X = OM * 1.0               # x: frequency (jw)
        self.Z = -SG * 1.6              # z: sigma (towards the viewer = negative sigma)
        self.Y = np.clip(mag, -1.2, 1.3) + 1.2
        om_line = np.linspace(-4.0, 4.0, 400)
        self.slice = (om_line, np.clip(np.log10(np.abs(self.tf(1j * om_line))), -1.2, 1.3) + 1.2)
        p = -self.ZETA * self.W0 + 1j * self.W0 * math.sqrt(1 - self.ZETA ** 2)
        self.poles = [p, p.conjugate()]

    def color(self, v):
        u = clamp(v / 2.5)
        if u < 0.5:
            return mix((0.05, 0.2, 0.7), CYAN, u / 0.5)
        return mix(CYAN, MAGENTA, (u - 0.5) / 0.5)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        grow = ease_out(lt / 1.0, 3)
        cam = Camera(0.55 - 0.25 * lt / 6, 0.5, 14.0, target=(0, 0.8, 0.2), fov=40)
        solid_surface(ctx, cam, self.X, self.Y * grow, self.Z, self.color, edge_alpha=0.6, fill_alpha=0.88, width=0.8)
        # poles
        for p in self.poles:
            x, z = p.imag, -p.real * 1.6
            line3(ctx, cam, np.array([[x, 0, z], [x, 2.5 * grow, z]]), RED, 2.5, 0.9)
            px, py, _ = cam.project(np.array([[x, 2.6 * grow, z]]))
            draw_text(ctx, "×", px[0], py[0], size=34, font="latin", weight=600, color=RED)
        # the imaginary-axis slice = Fourier transform (frequency response)
        sl = smooth((lt - 2.8) / 0.8)
        if sl > 0:
            om, mag = self.slice
            n = int(len(om) * sl)
            if n > 1:
                P = np.stack([om[:n], mag[:n] + 0.03, np.zeros(n)], 1)
                line3(ctx, cam, P, GOLD, 4.0, 1.0)
            # translucent cutting plane
            quad = np.array([[-4.2, 0, 0], [4.2, 0, 0], [4.2, 3.0, 0], [-4.2, 3.0, 0]])
            qx, qy, _ = cam.project(quad)
            polyline(ctx, np.stack([qx, qy], 1), close=True)
            ctx.set_source_rgba(1.0, 0.8, 0.3, 0.08 * sl)
            ctx.fill()
        # axes
        axis3(ctx, cam, (-4.4, 0, 0), (4.6, 0, 0), WHITE, 0.5, 1.4, head=12)
        axis3(ctx, cam, (0, 0, -1.8), (0, 0, 3.6), WHITE, 0.5, 1.4, head=12)
        lx, ly, _ = cam.project(np.array([[4.8, 0, 0], [0, 0, 3.9]]))
        draw_text(ctx, "jω 频率", lx[0] + 10, ly[0], size=22, font="sans", weight=500, color=WHITE, alpha=0.8,
                  anchor="left")
        draw_text(ctx, "σ 衰减", lx[1], ly[1] + 24, size=22, font="sans", weight=500, color=WHITE, alpha=0.8)
        draw_math(ctx, r"$F(s)=\int_0^{\infty} f(t)\,e^{-st}\,dt,\quad s=\sigma+i\omega$", W / 2, 140, size=38,
                  color=WHITE, alpha=smooth((lt - 0.3) / 0.5), glow=6)
        a = smooth((lt - 3.2) / 0.5)
        draw_text(ctx, "σ = 0 的切片 = 傅里叶变换", 1500, 300, size=26, font="sans", weight=600, color=GOLD, alpha=a,
                  glow=6, glow_alpha=0.3)
        draw_text(ctx, "极点 ×：在左半平面 → 系统稳定", 1500, 345, size=20, font="sans", weight=400, color=RED,
                  alpha=0.9 * smooth((lt - 1.2) / 0.5))


class Wavelet(Scene):
    """Time-frequency tilings: the STFT's fixed boxes vs. the wavelet's multi-resolution boxes."""
    start, end = S.T_WAVELET, S.T_HEAT
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.9
    T_TONE = 0.12    # the low tone's frequency (fraction of the band)
    T_CLICK = 0.62   # the click's time

    def energy(self, t0, t1, f0, f1):
        """Energy density of the test signal (a steady low tone + a sharp click) inside a time-frequency box.

        A steady tone spreads its energy over time, so its density is 1/df; a click spreads over frequency, so
        its density is 1/dt. Normalised so the best-matched box is 1.
        """
        e = 0.0
        if f0 <= self.T_TONE < f1:
            e += (1 / (f1 - f0)) / 16
        if t0 <= self.T_CLICK < t1:
            e += (1 / (t1 - t0)) / 16
        return min(1.0, e)

    def panel(self, ctx, x0, y0, w, h, boxes, a, col):
        for (t0, t1, f0, f1) in boxes:
            e = self.energy(t0, t1, f0, f1)
            X0, X1 = x0 + t0 * w, x0 + t1 * w
            Y0, Y1 = y0 + h - f1 * h, y0 + h - f0 * h
            ctx.rectangle(X0 + 1, Y0 + 1, X1 - X0 - 2, Y1 - Y0 - 2)
            c = mix((0.05, 0.08, 0.2), col, e)
            ctx.set_source_rgba(c[0], c[1], c[2], a * (0.35 + 0.65 * e))
            ctx.fill_preserve()
            set_rgba(ctx, col, 0.5 * a)
            ctx.set_line_width(1)
            ctx.stroke()
        draw_text(ctx, "时间 →", x0 + w, y0 + h + 28, size=16, font="sans", weight=500, color=WHITE, alpha=0.6 * a,
                  anchor="right")
        draw_text(ctx, "频率 ↑", x0 - 12, y0 + 12, size=16, font="sans", weight=500, color=WHITE, alpha=0.6 * a,
                  anchor="right")

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        # basis functions on top
        xs = np.linspace(0, 1, 400)
        for k, (x0, title, sub, col) in enumerate([(160.0, "短时傅里叶 STFT", "窗口宽度固定", CYAN),
                                                    (1010.0, "小波 WAVELET", "高频窄、低频宽", GOLD)]):
            a = smooth((lt - 0.2 - 0.4 * k) / 0.4)
            draw_text(ctx, title, x0, 150, size=30, font="sans", weight=700, color=col, alpha=a, anchor="left")
            draw_text(ctx, sub, x0, 192, size=20, font="sans", weight=400, color=WHITE, alpha=0.7 * a, anchor="left")
            for j, (f, width) in enumerate([(3, 0.28), (8, 0.28 if k == 0 else 0.11), (20, 0.28 if k == 0 else 0.045)]):
                cx = 0.18 + 0.32 * j
                env = np.exp(-((xs - cx) / (width / 2.5)) ** 2)
                y = env * np.cos(2 * np.pi * f * (xs - cx) / (0.3 if k == 0 else width * 1.2))
                pts = np.stack([x0 + xs * 750, 280 - 45 * y], 1)
                stroke_poly(ctx, pts, col, 1.8, 0.9 * a)
        # tilings
        stft = []
        for i in range(8):
            for j in range(8):
                stft.append((i / 8, (i + 1) / 8, j / 8, (j + 1) / 8))
        wav = []
        bands = [(0.0, 0.0625, 1), (0.0625, 0.125, 2), (0.125, 0.25, 4), (0.25, 0.5, 8), (0.5, 1.0, 16)]
        for f0, f1, n in bands:
            for i in range(n):
                wav.append((i / n, (i + 1) / n, f0, f1))
        a1 = smooth((lt - 0.6) / 0.5)
        a2 = smooth((lt - 1.2) / 0.5)
        self.panel(ctx, 160.0, 380.0, 750.0, 450.0, stft, a1, CYAN)
        self.panel(ctx, 1010.0, 380.0, 750.0, 450.0, wav, a2, GOLD)
        c = smooth((lt - 2.2) / 0.5)
        if c > 0:
            for x0 in (160.0, 1010.0):
                xx = x0 + self.T_CLICK * 750
                draw_text(ctx, "咔哒声", xx, 360, size=18, font="sans", weight=500, color=RED, alpha=c)
                yy = 380 + 450 - self.T_TONE * 450
                draw_text(ctx, "低音", x0 + 760, yy, size=18, font="sans", weight=500, color=RED, alpha=c,
                          anchor="left")


def family_scenes():
    return [Laplace3D(), Wavelet()]
