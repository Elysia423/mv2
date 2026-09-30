"""Beyond Fourier: the Laplace transform over the s-plane, and wavelets vs. the short-time Fourier transform."""
import math

import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, LUT_FIRE, LUT_ICE, MAGENTA, RED, VIOLET, W, WHITE, apply_lut, background, clamp,
                    draw_math, draw_text, ease_out, mix, paint_image, polyline, rounded_rect, set_rgba, smooth,
                    smoother, stroke_poly, surface_from_array)
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
        cam = Camera(0.55 - 0.25 * lt / 6, 0.5, 14.0, target=(0, 0.8, 0.2), fov=40, center=(W / 2 - 190, H / 2 + 10))
        solid_surface(ctx, cam, self.X, self.Y * grow, self.Z, self.color, edge_alpha=0.6, fill_alpha=0.88, width=0.8)
        # poles
        for p in self.poles:
            x, z = p.imag, -p.real * 1.6
            line3(ctx, cam, np.array([[x, 0, z], [x, 2.5 * grow, z]]), RED, 2.5, 0.9)
            px, py, _ = cam.project(np.array([[x, 2.6 * grow, z]]))
            draw_text(ctx, "×", px[0], py[0], size=34, font="latin", weight=600, color=RED)
        draw_text(ctx, "极点", px[0] + 22, py[0] - 4, size=30, font="sans", weight=600, color=RED,
                  alpha=smooth((lt - 1.0) / 0.4), anchor="left")
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
        draw_text(ctx, "jω 频率", lx[0] + 10, ly[0], size=30, font="sans", weight=500, color=WHITE, alpha=0.8,
                  anchor="left")
        draw_text(ctx, "σ 衰减", lx[1], ly[1] + 24, size=30, font="sans", weight=500, color=WHITE, alpha=0.8)
        draw_math(ctx, r"$F(s)=\int_0^{\infty} f(t)\,e^{-st}\,dt,\quad s=\sigma+i\omega$", W / 2, 140, size=38,
                  color=WHITE, alpha=smooth((lt - 0.3) / 0.5), glow=6)
        self.absorber(ctx, lt)

    def absorber(self, ctx, lt):
        """A shock absorber released at REL: its bounce e^(σt)·cos(ωt) is set by the two poles."""
        a = smooth((lt - 0.9) / 0.4)
        if a <= 0:
            return
        sig = -self.ZETA * self.W0
        om = self.W0 * math.sqrt(1 - self.ZETA ** 2)
        k = 2.2                      # screen seconds -> model time
        REL = 1.4

        def u(tau):
            return np.where(tau < 0, 1.0, np.exp(sig * k * np.maximum(tau, 0)) * np.cos(om * k * np.maximum(tau, 0)))

        cx, top, rest, amp = 1590.0, 455.0, 640.0, 70.0
        draw_text(ctx, "减震器", cx, 405, size=34, font="sans", weight=600, color=WHITE, alpha=a)
        set_rgba(ctx, WHITE, 0.6 * a)
        ctx.set_line_width(3)
        ctx.move_to(cx - 60, top)
        ctx.line_to(cx + 60, top)
        ctx.stroke()
        y = rest + amp * float(u(lt - REL))
        # spring: zig-zag from the ceiling to the block
        n = 14
        ys = np.linspace(top, y - 28, n + 1)
        xs = cx + np.array([0] + [(-1) ** i * 18 for i in range(1, n)] + [0])
        polyline(ctx, np.stack([xs, ys], 1))
        set_rgba(ctx, CYAN, 0.9 * a)
        ctx.set_line_width(2.5)
        ctx.stroke()
        rounded_rect(ctx, cx - 42, y - 28, 84, 56, 8)
        ctx.set_source_rgba(0.55, 0.25, 0.9, 0.85 * a)
        ctx.fill()
        # its trace, drawn as it happens, inside the decay envelope e^(σt)
        x0, x1, span = 1670.0, 1870.0, 3.6
        tau = np.linspace(0, max(0.0, min(lt - REL, span)), 300)
        env = np.linspace(0, span, 200)
        e = np.exp(sig * k * env)
        ctx.set_dash([5, 6])
        for sgn in (1, -1):
            polyline(ctx, np.stack([x0 + env / span * (x1 - x0), rest + sgn * amp * e], 1))
            set_rgba(ctx, GOLD, 0.55 * a)
            ctx.set_line_width(1.5)
            ctx.stroke()
        ctx.set_dash([])
        if len(tau) > 1 and lt > REL:
            stroke_poly(ctx, np.stack([x0 + tau / span * (x1 - x0), rest + amp * u(tau)], 1), VIOLET, 2.5, 0.95 * a,
                        glow=0.6)
        draw_math(ctx, r"$e^{\sigma t}$", x1 - 20, rest - amp - 26, size=30, color=GOLD, alpha=a)


class Wavelet(Scene):
    """Short-time Fourier vs wavelets on the same real signal: the GW150914 chirp (LIGO Hanford, whitened).

    A fixed 125 ms window smears the fast, rising chirp; Morlet wavelets (short at high frequency, long at low
    frequency) keep it sharp.
    """
    start, end = S.T_WAVELET, S.T_HEAT
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.9
    WIN = (-0.30, 0.06)
    NPER = 512      # 125 ms at 4096 Hz

    def prepare(self):
        import os
        from scipy.signal import stft
        from ..core import ROOT
        d = np.load(os.path.join(ROOT, "assets", "ligo", "gw150914.npz"))
        t, x, fr = d["t"], d["h"], d["freqs"]
        sel = (t >= self.WIN[0]) & (t <= self.WIN[1])
        ff, tt, Z = stft(x, 4096, nperseg=self.NPER, noverlap=self.NPER - 4, boundary="even")
        M = np.abs(Z) ** 2
        Mf = np.array([np.interp(fr, ff, M[:, j]) for j in range(M.shape[1])]).T   # onto the log-frequency grid
        st = np.array([np.interp(t[sel], t[0] + tt, row) for row in Mf])
        self.maps = [self._norm(st), self._norm(d["tf"][:, sel])]

    @staticmethod
    def _norm(A):
        return np.clip(A / np.percentile(A, 99.8), 0, 1) ** 0.7

    def panel(self, ctx, k, x0, y0, w, h, rev, a, col, lut):
        m = self.maps[k][::-1].copy()
        m[:, int(rev * m.shape[1]):] = 0
        paint_image(ctx, surface_from_array(apply_lut(m, lut)), x0, y0, w, h, a, anchor="topleft")
        set_rgba(ctx, col, 0.6 * a)
        ctx.set_line_width(1.5)
        ctx.rectangle(x0, y0, w, h)
        ctx.stroke()
        if 0 < rev < 1:
            xx = x0 + rev * w
            set_rgba(ctx, WHITE, 0.7 * a)
            ctx.set_line_width(2)
            ctx.move_to(xx, y0)
            ctx.line_to(xx, y0 + h)
            ctx.stroke()

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        # basis functions on top: fixed-width windows vs. wavelets that shrink as the frequency rises
        xs = np.linspace(0, 1, 400)
        for k, (x0, title, col) in enumerate([(160.0, "短时傅里叶 STFT", CYAN), (1010.0, "小波 WAVELET", GOLD)]):
            a = smooth((lt - 0.2 - 0.4 * k) / 0.4)
            draw_text(ctx, title, x0, 168, size=38, font="sans", weight=700, color=col, alpha=a, anchor="left")
            for j, (f, width) in enumerate([(3, 0.28), (8, 0.28 if k == 0 else 0.11), (20, 0.28 if k == 0 else 0.045)]):
                cx = 0.18 + 0.32 * j
                env = np.exp(-((xs - cx) / (width / 2.5)) ** 2)
                y = env * np.cos(2 * np.pi * f * (xs - cx) / (0.3 if k == 0 else width * 1.2))
                stroke_poly(ctx, np.stack([x0 + xs * 750, 262 - 45 * y], 1), col, 1.8, 0.9 * a)
        # the same real signal through both
        a = smooth((lt - 0.5) / 0.4)
        rev = smoother((lt - 0.7) / 2.2)
        self.panel(ctx, 0, 160.0, 380.0, 750.0, 450.0, rev, a, CYAN, LUT_ICE)
        self.panel(ctx, 1, 1010.0, 380.0, 750.0, 450.0, rev, a, GOLD, LUT_FIRE)
        draw_text(ctx, "引力波数据", W / 2, 346, size=30, font="sans", weight=600, color=WHITE, alpha=0.85 * a)
        # the STFT's fixed window, to scale
        wpx = self.NPER / 4096 / (self.WIN[1] - self.WIN[0]) * 750
        set_rgba(ctx, CYAN, 0.8 * a)
        ctx.set_line_width(2)
        for xx in (180.0, 180.0 + wpx):
            ctx.move_to(xx, 846)
            ctx.line_to(xx, 862)
        ctx.move_to(180.0, 854)
        ctx.line_to(180.0 + wpx, 854)
        ctx.stroke()
        draw_text(ctx, "125 ms", 180.0 + wpx + 12, 854, size=28, font="latin", weight=600, color=CYAN, alpha=a,
                  anchor="left")
        draw_text(ctx, "时间 →", 1760, 858, size=28, font="sans", weight=500, color=WHITE, alpha=0.6 * a,
                  anchor="right")
        draw_text(ctx, "频率 ↑", 148, 392, size=28, font="sans", weight=500, color=WHITE, alpha=0.6 * a,
                  anchor="right")


def family_scenes():
    return [Laplace3D(), Wavelet()]
