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
        draw_text(ctx, "位置 x", lx[0] + 10, ly[0], size=20, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="left")
        draw_text(ctx, "时间 t", lx[1], ly[1] - 16, size=20, font="sans", weight=500, color=WHITE, alpha=0.7)
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
        draw_text(ctx, "各频率成分", bx0, by + 24, size=18, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="left")
        draw_math(ctx, r"$\frac{\partial T}{\partial t}=\alpha\,\frac{\partial^2 T}{\partial x^2}$", 1560, 190, size=42,
                  color=WHITE, alpha=smooth((lt - 0.3) / 0.5), glow=6)
        draw_math(ctx, r"$T(x,t)=\sum_n b_n\,e^{-\alpha n^2\pi^2 t}\,\sin n\pi x$", 1560, 280, size=32,
                  color=GOLD, alpha=smooth((lt - 1.2) / 0.5), glow=6)
        draw_text(ctx, "1807", 110, 130, size=66, font="latin", weight=200, color=WHITE, alpha=smooth(lt / 0.5),
                  anchor="left", glow=8, glow_alpha=0.3)
        draw_text(ctx, "热方程 · HEAT EQUATION", 114, 190, size=20, font="sans", weight=500, color=GOLD,
                  alpha=0.9 * smooth(lt / 0.5), anchor="left", tracking=0.1)


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
    start, end = S.T_FINALE, S.T_END
    bloom = 1.0
    KS = [1, -7, 17, -23]
    RS = [200, 150, 60, 25]

    def rose(self, s):
        return sum(r * np.exp(1j * (k * s + 0.3 * k)) for k, r in zip(self.KS, self.RS))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, strength=1.2, hue=(0.05, 0.04, 0.10))
        out = 1 - smooth((t - 217.2) / 1.4)
        cx, cy = W / 2, 480.0
        # spirograph drawn by epicycles behind the title
        if out > 0:
            ctx.push_group()
            prog = clamp(lt / 7.0)
            sgrid = np.linspace(0, 2 * np.pi * prog, max(2, int(1600 * prog)))
            z = self.rose(sgrid) * 0.95
            pts = np.stack([cx + z.real, cy + z.imag], 1)
            n = len(pts)
            seg = 40
            for j in range(0, n - 1, seg):
                u = j / 1600
                polyline(ctx, pts[j:j + seg + 1])
                set_rgba(ctx, hsv(0.5 + 0.45 * u, 0.7, 1.0), 0.7)
                ctx.set_line_width(2.0)
                ctx.stroke()
            s = 2 * np.pi * prog
            pos = cx + 1j * cy
            for k, r in zip(self.KS, self.RS):
                v = r * 0.95 * np.exp(1j * (k * s + 0.3 * k))
                circle(ctx, pos.real, pos.imag, r * 0.95, (0.6, 0.8, 1.0), 0.25, width=1.2)
                ctx.move_to(pos.real, pos.imag)
                pos = pos + v
                ctx.line_to(pos.real, pos.imag)
                set_rgba(ctx, WHITE, 0.5)
                ctx.set_line_width(1.2)
                ctx.stroke()
            if prog < 1:
                glow_dot(ctx, pos.real, pos.imag, 4, WHITE, 1.0, 5)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(0.75 * out)
            # title
            sc = 1.3 - 0.3 * ease_out_expo(lt / 1.4)
            draw_text(ctx, "傅里叶变换", cx, cy - 10, size=132, font="serif", weight=900, color=WHITE,
                      alpha=smooth(lt / 0.2) * out, tracking=0.22, glow=18, glow_alpha=0.5, scale=sc)
            draw_text(ctx, "THE FOURIER TRANSFORM", cx, cy + 100, size=24, font="latin", weight=300,
                      color=(0.8, 0.9, 1.0), alpha=0.9 * out, tracking=0.85, reveal=ease_out((lt - 1.2) / 1.2, 2))
            draw_text_chars(ctx, "换一个角度，看见万物的频率", cx, cy + 190, lt - 2.4, size=40, font="serif", weight=600,
                            color=GOLD, alpha=out, tracking=0.25, stagger=0.08, dur=0.5, glow=8, glow_alpha=0.45)
            draw_text(ctx, "JEAN-BAPTISTE JOSEPH FOURIER  ·  1768 – 1830", cx, cy + 270, size=18, font="latin",
                      weight=400, color=WHITE, alpha=0.65 * smooth((lt - 4.8) / 0.8) * out, tracking=0.35)
            draw_math(ctx, r"$\hat{f}(\omega)=\int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt$", cx, cy - 190,
                      size=34, color=(0.8, 0.9, 1.0), alpha=0.8 * out, reveal=ease_out((lt - 6.5) / 1.6, 2), glow=6)
        # bookend: a single circle and its sine wave
        b = window(t, 218.2, 225.4, 1.2, 1.4)
        if b > 0:
            ccx, ccy, R = 700.0, 500.0, 120.0
            th = 2 * math.pi * 0.5 * (t - 218.2)
            tx, ty = ccx + R * math.cos(th), ccy - R * math.sin(th)
            circle(ctx, ccx, ccy, R, CYAN, 0.55 * b, width=2)
            ctx.move_to(ccx, ccy)
            ctx.line_to(tx, ty)
            set_rgba(ctx, WHITE, 0.8 * b)
            ctx.set_line_width(2)
            ctx.stroke()
            X0 = 1000.0
            ts = np.linspace(max(218.2, t - 5.0), t, 300)
            ys = ccy - R * np.sin(2 * math.pi * 0.5 * (ts - 218.2))
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
            draw_text_chars(ctx, "一切，从一个圆开始", W / 2, 760, t - 219.2, size=38, font="serif", weight=600,
                            color=WHITE, alpha=b, tracking=0.25, stagger=0.08, dur=0.5)
            draw_text(ctx, "IT ALL BEGINS WITH A CIRCLE", W / 2, 815, size=15, font="latin", weight=500,
                      color=(0.8, 0.9, 1.0), alpha=0.6 * b * smooth((t - 220.0) / 0.8), tracking=0.4)
        c = window(t, 221.6, 225.6, 0.8, 1.0)
        if c > 0:
            draw_text(ctx, "本片所有画面由代码实时计算生成 · 所有声音由正弦波叠加合成", W / 2, 960, size=20, font="sans",
                      weight=400, color=WHITE, alpha=0.55 * c, tracking=0.12)
            draw_text(ctx, "EVERY FRAME COMPUTED BY CODE · EVERY SOUND BUILT FROM SINE WAVES", W / 2, 996, size=12,
                      font="latin", weight=500, color=WHITE, alpha=0.35 * c, tracking=0.3)

    def effects(self, t, lt):
        return {"flash": 0.8 * math.exp(-lt / 0.22), "chroma": 4 * math.exp(-lt / 0.45),
                "fade": 1 - smooth((t - 225.0) / 0.9)}
