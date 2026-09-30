"""The heat equation in 3D (the origin), the flash-cut montage and the finale."""
import math

import cairo
import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, LUT_FIRE, MAGENTA, ORANGE, VIOLET, W, WHITE, apply_lut, background, circle, clamp,
                    draw_math, draw_text, draw_text_chars, ease_out, ease_out_expo, glow_dot, hsv, lerp, mix,
                    paint_image, polyline, set_rgba, smooth, smoother, stroke_poly, surface_from_array, window)
from ..data import beat_pulse
from ..three import Camera, axis3, line3, solid_surface
from .base import Scene


class Heat3D(Scene):
    """T(x, t) of a hot rod: the surface grows along time; high-frequency wrinkles die first."""
    start, end = S.T_HEAT, S.T_MONTAGE
    fade_in = 0.3
    fade_out = 0.1
    bloom = 1.0
    NM = 60
    NX = 90
    NT = 44

    def prepare(self):
        x = np.linspace(0, 1, self.NX)
        xf = np.linspace(0, 1, 800)
        u0 = np.zeros_like(xf)
        for a, b, v in [(0.10, 0.2, 1.0), (0.33, 0.37, 0.85), (0.52, 0.66, 0.95), (0.78, 0.8, 1.0), (0.86, 0.9, 0.7)]:
            u0[(xf >= a) & (xf <= b)] = v
        n = np.arange(1, self.NM + 1)
        self.b = 2 * np.trapezoid(u0[:, None] * np.sin(np.pi * np.outer(xf, n)), xf, axis=0)
        self.S = np.sin(np.pi * np.outer(x, n))
        taus = np.concatenate([[0.0], np.geomspace(2e-5, 0.06, self.NT - 1)])
        self.taus = taus
        self.T = np.array([np.clip(self.S @ (self.b * np.exp(-(n * np.pi) ** 2 * tau)), 0, 1.05) for tau in taus])
        self.x = x

    def color(self, v):
        c = apply_lut(np.array([clamp(v / 1.6)]), LUT_FIRE)[0]
        return (float(c[0]), float(c[1]), float(c[2]))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.06, 0.03, 0.03))
        grow = clamp((lt - 0.4) / 5.0)
        k = max(2, int(grow * self.NT))
        X = np.repeat(((self.x - 0.5) * 9.0)[None, :], k, 0)
        Z = np.repeat((np.arange(k) * -0.2 + 2.0)[:, None], self.NX, 1)
        Y = 1.6 * self.T[:k]
        cam = Camera(0.75 - 0.35 * lt / 8, 0.42, 13.5, target=(0, 0.5, -2.2), fov=40)
        solid_surface(ctx, cam, X[::-1], Y[::-1], Z[::-1], self.color, edge_alpha=0.55, fill_alpha=0.9, width=0.8)
        # the current profile at the front edge
        P = np.stack([X[-1], Y[-1] + 0.02, Z[-1]], 1)
        line3(ctx, cam, P, WHITE, 2.5, 0.9)
        axis3(ctx, cam, (-4.8, 0, 2.2), (4.8, 0, 2.2), WHITE, 0.4, 1.2, head=10)
        axis3(ctx, cam, (-4.8, 0, 2.2), (-4.8, 0, 2.2 - 0.2 * self.NT), WHITE, 0.4, 1.2, head=10)
        lx, ly, _ = cam.project(np.array([[4.9, 0, 2.2], [-4.9, 0, 2.2 - 0.2 * self.NT - 0.3]]))
        draw_text(ctx, "位置 x", lx[0] + 10, ly[0], size=28, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="left")
        draw_text(ctx, "时间 t", lx[1], ly[1] - 16, size=28, font="sans", weight=500, color=WHITE, alpha=0.7)
        # mode amplitudes
        tau = self.taus[k - 1]
        n = np.arange(1, 31)
        bn = np.abs(self.b[:30] * np.exp(-(n * np.pi) ** 2 * tau))
        bx0, by = 1300.0, 860.0
        amax = np.abs(self.b[:30]).max()
        for i in range(30):
            v = bn[i] / amax
            set_rgba(ctx, hsv(0.02 + 0.6 * i / 30, 0.8, 1.0), 0.85)
            ctx.rectangle(bx0 + i * 17, by - 110 * v, 11, 110 * v)
            ctx.fill()
        draw_text(ctx, "各频率成分", bx0, by + 24, size=28, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="left")
        draw_math(ctx, r"$\frac{\partial T}{\partial t}=\alpha\,\frac{\partial^2 T}{\partial x^2}$", 1560, 190, size=42,
                  color=WHITE, alpha=smooth((lt - 0.3) / 0.5), glow=6)
        draw_math(ctx, r"$T(x,t)=\sum_n b_n\,e^{-\alpha n^2\pi^2 t}\,\sin n\pi x$", 1560, 280, size=32,
                  color=GOLD, alpha=smooth((lt - 1.2) / 0.5), glow=6)


class Montage(Scene):
    start, end = S.T_MONTAGE, S.T_FINALE
    bloom = 0.9
    SRC = [4.5, 20.6, 25.2, 32.6, 40.1, 53.2, 56.8, 61.9,
           64.6, 77.3, 81.2, 90.6, 97.0, 102.9, 109.2, 116.6,
           123.1, 139.6, 143.4, 152.9, 155.6, 160.9, 182.6, 188.4]

    def __init__(self, scenes):
        self.scenes = scenes

    def cut(self, t):
        lt = t - self.start
        if lt < 4.0:
            i = int(lt / 0.5)
            c0 = self.start + i * 0.5
        else:
            i = 8 + int((lt - 4.0) / 0.25)
            c0 = self.start + 4.0 + (i - 8) * 0.25
        return min(i, len(self.SRC) - 1), c0

    def draw(self, cv, t, lt):
        i, c0 = self.cut(t)
        ts = self.SRC[i] + (t - c0)
        sc = next((s for s in self.scenes if s.start <= ts < s.end), None)
        if sc is None:
            return
        ctx = cv.ctx
        ctx.save()
        z = 1.08 - 0.08 * ease_out((t - c0) / 0.4)
        ctx.translate(W / 2, H / 2)
        ctx.scale(z, z)
        ctx.translate(-W / 2, -H / 2)
        sc.draw(cv, ts, ts - sc.start)
        ctx.restore()

    def effects(self, t, lt):
        i, c0 = self.cut(t)
        return {"flash": 0.22 * math.exp(-(t - c0) / 0.07), "chroma": 2.5 * math.exp(-(t - c0) / 0.15)}


class Finale(Scene):
    """Title card with an epicycle flower; the card flips over ('change the angle') to show the flower's spectrum,
    then everything collapses back into one circle."""
    start, end = S.T_FINALE, S.T_END
    bloom = 1.0
    KS = [1, -7, 17, -23]
    RS = [200, 150, 60, 25]
    FLIP = 3.8      # local time the flip starts
    FLIP_D = 0.9
    COLLAPSE = 9.5  # local time the card collapses into a dot
    BOOK = 11.8     # local time the bookend circle begins

    def rose(self, s):
        return sum(r * np.exp(1j * (k * s + 0.3 * k)) for k, r in zip(self.KS, self.RS))

    def front(self, ctx, lt, cx, cy):
        prog = clamp(lt / 3.4)
        sgrid = np.linspace(0, 2 * np.pi * prog, max(2, int(1600 * prog)))
        z = self.rose(sgrid) * 0.95
        pts = np.stack([cx + z.real, cy + z.imag], 1)
        for j in range(0, len(pts) - 1, 40):
            polyline(ctx, pts[j:j + 41])
            set_rgba(ctx, hsv(0.5 + 0.45 * j / 1600, 0.7, 1.0), 0.7)
            ctx.set_line_width(2.0)
            ctx.stroke()
        s_ = 2 * np.pi * prog
        pos = cx + 1j * cy
        for k, r in zip(self.KS, self.RS):
            v = r * 0.95 * np.exp(1j * (k * s_ + 0.3 * k))
            circle(ctx, pos.real, pos.imag, r * 0.95, (0.6, 0.8, 1.0), 0.25, width=1.2)
            ctx.move_to(pos.real, pos.imag)
            pos = pos + v
            ctx.line_to(pos.real, pos.imag)
            set_rgba(ctx, WHITE, 0.5)
            ctx.set_line_width(1.2)
            ctx.stroke()
        if prog < 1:
            glow_dot(ctx, pos.real, pos.imag, 4, WHITE, 1.0, 5)
        sc = 1.3 - 0.3 * ease_out_expo(lt / 1.4)
        draw_text(ctx, "傅里叶变换", cx, cy - 10, size=132, font="serif", weight=900, color=WHITE,
                  alpha=smooth(lt / 0.2), tracking=0.22, glow=18, glow_alpha=0.5, scale=sc)
        draw_text(ctx, "THE FOURIER TRANSFORM", cx, cy + 100, size=32, font="latin", weight=300,
                  color=(0.8, 0.9, 1.0), alpha=0.9, tracking=0.6, reveal=ease_out((lt - 1.0) / 1.0, 2))

    def back(self, ctx, lt, cx, cy):
        """The flower seen 'from the side': four spectral lines at k = -23, -7, 1, 17."""
        u = smooth((lt - self.FLIP - self.FLIP_D / 2) / 0.6)
        base = cy + 170
        x_of = lambda k: cx + k * 26
        set_rgba(ctx, WHITE, 0.35)
        ctx.set_line_width(1.5)
        ctx.move_to(cx - 700, base)
        ctx.line_to(cx + 700, base)
        ctx.stroke()
        for k, r in zip(self.KS, self.RS):
            hgt = r * 0.9 * u
            col = hsv(0.5 + 0.45 * (k + 23) / 40, 0.7, 1.0)
            ctx.move_to(x_of(k), base)
            ctx.line_to(x_of(k), base - hgt)
            set_rgba(ctx, col, 0.95)
            ctx.set_line_width(10)
            ctx.stroke()
            glow_dot(ctx, x_of(k), base - hgt, 6, col, u, 5)
            draw_text(ctx, f"k = {k}", x_of(k), base + 36, size=28, font="latin", weight=500, color=col, alpha=u)
        a = smooth((lt - self.FLIP - 0.6) / 0.5)
        draw_text_chars(ctx, "换一个角度，看见万物的频率", cx, cy - 210, lt - self.FLIP - 0.6, size=64, font="serif",
                        weight=700, color=GOLD, alpha=1.0, tracking=0.2, stagger=0.05, dur=0.4, glow=10,
                        glow_alpha=0.45)
        draw_math(ctx, r"$\hat{f}(\omega)=\int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt$", cx, cy - 105, size=40,
                  color=(0.85, 0.92, 1.0), alpha=0.9 * a, glow=6)
        draw_text(ctx, "JEAN-BAPTISTE JOSEPH FOURIER  ·  1768 – 1830", cx, cy + 300, size=30, font="latin",
                  weight=400, color=WHITE, alpha=0.75 * smooth((lt - self.FLIP - 1.4) / 0.6), tracking=0.14)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, strength=1.2, hue=(0.05, 0.04, 0.10))
        cx, cy = W / 2, 470.0
        col = smooth((lt - self.COLLAPSE) / 1.4)
        if col < 0.985:
            ctx.save()
            # collapse everything towards the centre (never to a singular transform)
            k = max(1 - col, 0.015)
            ctx.translate(cx, cy)
            ctx.scale(k, k)
            ctx.translate(-cx, -cy)
            fu = clamp((lt - self.FLIP) / self.FLIP_D)
            sx = math.cos(math.pi * fu)
            ctx.push_group()
            if fu < 0.5:
                self.front(ctx, lt, cx, cy)
            else:
                self.back(ctx, lt, cx, cy)
            pat = ctx.pop_group()
            ctx.translate(cx, cy)
            ctx.scale(max(abs(sx), 0.02), 1.0)
            ctx.translate(-cx, -cy)
            ctx.set_source(pat)
            ctx.paint_with_alpha(1 - col ** 3)
            ctx.restore()
            if 0.3 < fu < 0.7:
                g = 1 - abs(fu - 0.5) / 0.2
                glow_dot(ctx, cx, cy, 8, WHITE, g, 30)
        if 0.6 < col and lt < self.BOOK + 0.5:
            glow_dot(ctx, cx, cy, 6, WHITE, 1.0 - smooth((lt - self.BOOK) / 0.5), 6)
        # bookend: a single circle and its sine wave
        t0 = self.start + self.BOOK
        b = window(t, t0, S.T_END - 0.6, 1.0, 1.4)
        if b > 0:
            ccx, ccy, R = 700.0, 480.0, 120.0
            grow = ease_out((t - t0) / 1.0, 3)
            th = 2 * math.pi * 0.5 * (t - t0)
            tx, ty = ccx + R * math.cos(th), ccy - R * math.sin(th)
            circle(ctx, ccx, ccy, R * grow, CYAN, 0.55 * b, width=2)
            ctx.move_to(ccx, ccy)
            ctx.line_to(tx, ty)
            set_rgba(ctx, WHITE, 0.8 * b)
            ctx.set_line_width(2)
            ctx.stroke()
            X0 = 1000.0
            ts = np.linspace(max(t0, t - 5.0), t, 300)
            ys = ccy - R * np.sin(2 * math.pi * 0.5 * (ts - t0))
            xs = X0 + (t - ts) * 170
            stroke_poly(ctx, np.stack([xs, ys], 1), GOLD, 3.0, 0.95 * b, glow=1.0)
            ctx.set_dash([6, 8])
            set_rgba(ctx, WHITE, 0.3 * b)
            ctx.set_line_width(1.2)
            ctx.move_to(tx, ty)
            ctx.line_to(X0, ty)
            ctx.stroke()
            ctx.set_dash([])
            glow_dot(ctx, tx, ty, 5, WHITE, b, 5)
            glow_dot(ctx, X0, ty, 5, GOLD, b, 5)
            draw_text_chars(ctx, "一切，从一个圆开始", W / 2, 720, t - t0 - 0.8, size=48, font="serif", weight=600,
                            color=WHITE, alpha=b, tracking=0.25, stagger=0.06, dur=0.4)
            draw_text(ctx, "It all begins with a circle", W / 2, 785, size=28, font="latin", weight=400,
                      color=(0.8, 0.9, 1.0), alpha=0.7 * b * smooth((t - t0 - 1.4) / 0.6), tracking=0.04)
        c = window(t, S.T_END - 4.6, S.T_END - 0.4, 0.8, 1.0)
        if c > 0:
            draw_text(ctx, "本片所有画面由代码实时计算生成 · 所有声音由正弦波叠加合成", W / 2, 930, size=30, font="sans",
                      weight=400, color=WHITE, alpha=0.6 * c, tracking=0.08)
            draw_text(ctx, "Every frame computed by code · every sound built from sine waves", W / 2, 978, size=26,
                      font="latin", weight=400, color=WHITE, alpha=0.45 * c, tracking=0.02)

    def effects(self, t, lt):
        return {"flash": 0.8 * math.exp(-lt / 0.22), "chroma": 4 * math.exp(-lt / 0.45),
                "fade": 1 - smooth((t - (S.T_END - 1.0)) / 0.9)}
