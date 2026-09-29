"""Heat (the origin), the flash-cut montage and the finale."""
import math

import cairo
import numpy as np

from ..core import (CYAN, GOLD, H, LUT_FIRE, MAGENTA, ORANGE, VIOLET, W, WHITE, apply_lut, background, circle, clamp,
                    draw_math, draw_text, draw_text_chars, ease_out, ease_out_expo, glow_dot, hsv, lerp, mix,
                    paint_image, polyline, set_rgba, smooth, stroke_poly, surface_from_array, window)
from ..data import beat_pulse
from .base import Scene


class Heat(Scene):
    start, end = 196.0, 204.0
    fade_in = 0.4
    fade_out = 0.1
    bloom = 1.0
    NM = 80

    def prepare(self):
        x = np.linspace(0, 1, 800)
        u0 = np.zeros_like(x)
        for a, b, v in [(0.10, 0.2, 1.0), (0.33, 0.37, 0.85), (0.52, 0.66, 0.95), (0.78, 0.8, 1.0), (0.86, 0.9, 0.7)]:
            u0[(x >= a) & (x <= b)] = v
        n = np.arange(1, self.NM + 1)
        S = np.sin(np.pi * np.outer(x, n))
        self.b = 2 * np.trapezoid(u0[:, None] * S, x, axis=0)
        self.S = S
        self.x = x

    def tau(self, t):
        if t < 197.6:
            return 0.0
        u = clamp((t - 197.6) / 5.6)
        return 1.5e-5 * (6000 ** u) - 1.5e-5

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4, hue=(0.06, 0.03, 0.03))
        tau = self.tau(t)
        n = np.arange(1, self.NM + 1)
        bn = self.b * np.exp(-(n * np.pi) ** 2 * tau)
        T = np.clip(self.S @ bn, 0, 1.05)
        a = smooth(lt / 0.8)
        x0, x1, ry, rh = 260.0, 1660.0, 540.0, 64.0
        # rod
        strip = apply_lut(np.clip(T, 0, 1)[None, :] ** 0.9 * np.ones((8, 1)), LUT_FIRE)
        paint_image(ctx, surface_from_array(strip), (x0 + x1) / 2, ry, x1 - x0, rh, a)
        set_rgba(ctx, (0.6, 0.5, 0.5), 0.4 * a)
        ctx.set_line_width(1.5)
        ctx.rectangle(x0, ry - rh / 2, x1 - x0, rh)
        ctx.stroke()
        # temperature curve
        xs = x0 + self.x * (x1 - x0)
        stroke_poly(ctx, np.stack([xs, ry - rh / 2 - 30 - 210 * T], 1), mix(ORANGE, WHITE, 0.3), 2.5, a, glow=1.0)
        draw_text(ctx, "温度", x0 - 20, ry - rh / 2 - 150, size=20, font="sans", weight=500, color=ORANGE, alpha=0.8 * a,
                  anchor="right")
        # modes
        by = 830.0
        nb = 40
        bw = (x1 - x0) / nb
        amax = np.abs(self.b[:nb]).max()
        for k in range(nb):
            v = abs(bn[k]) / amax
            col = hsv(0.02 + 0.6 * k / nb, 0.8, 1.0)
            set_rgba(ctx, col, 0.85 * a)
            ctx.rectangle(x0 + k * bw + bw * 0.2, by - 130 * v, bw * 0.6, 130 * v)
            ctx.fill()
        draw_text(ctx, "低频", x0, by + 26, size=16, font="sans", weight=500, color=WHITE, alpha=0.6 * a, anchor="left")
        draw_text(ctx, "高频", x1, by + 26, size=16, font="sans", weight=500, color=WHITE, alpha=0.6 * a, anchor="right")
        ah = smooth((t - 199.5) / 0.6)
        draw_text(ctx, "高频先消失 → 温度越来越平滑", x1, by - 165, size=22, font="sans", weight=500, color=GOLD,
                  alpha=0.9 * ah * a, anchor="right")
        draw_text(ctx, "1807", 1810, 110, size=66, font="latin", weight=200, color=WHITE, alpha=a, anchor="right",
                  glow=8, glow_alpha=0.3)
        draw_text(ctx, "《论热的传播》", 1810, 175, size=24, font="serif", weight=600, color=GOLD, alpha=0.9 * a,
                  anchor="right", tracking=0.1)
        draw_text(ctx, "MÉMOIRE SUR LA PROPAGATION DE LA CHALEUR", 1810, 210, size=12, font="latin", weight=500,
                  color=WHITE, alpha=0.5 * a, anchor="right", tracking=0.3)


class Montage(Scene):
    start, end = 204.0, 212.0
    bloom = 0.9
    SRC = [5.5, 17.2, 30.8, 38.5, 45.0, 56.5, 66.0, 71.0,
           77.2, 86.9, 90.5, 99.0, 106.5, 121.8, 133.5, 141.5,
           150.8, 157.2, 163.8, 168.6, 175.0, 181.0, 187.0, 193.5]

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
    start, end = 212.0, 232.0
    bloom = 1.0
    KS = [1, -7, 17, -23]
    RS = [200, 150, 60, 25]

    def rose(self, s):
        return sum(r * np.exp(1j * (k * s + 0.3 * k)) for k, r in zip(self.KS, self.RS))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, strength=1.2, hue=(0.05, 0.04, 0.10))
        out = 1 - smooth((t - 223.2) / 1.4)
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
        b = window(t, 224.2, 231.4, 1.2, 1.4)
        if b > 0:
            ccx, ccy, R = 700.0, 500.0, 120.0
            th = 2 * math.pi * 0.5 * (t - 224.2)
            tx, ty = ccx + R * math.cos(th), ccy - R * math.sin(th)
            circle(ctx, ccx, ccy, R, CYAN, 0.55 * b, width=2)
            ctx.move_to(ccx, ccy)
            ctx.line_to(tx, ty)
            set_rgba(ctx, WHITE, 0.8 * b)
            ctx.set_line_width(2)
            ctx.stroke()
            X0 = 1000.0
            ts = np.linspace(max(224.2, t - 5.0), t, 300)
            ys = ccy - R * np.sin(2 * math.pi * 0.5 * (ts - 224.2))
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
            draw_text_chars(ctx, "一切，从一个圆开始", W / 2, 760, t - 225.2, size=38, font="serif", weight=600,
                            color=WHITE, alpha=b, tracking=0.25, stagger=0.08, dur=0.5)
            draw_text(ctx, "IT ALL BEGINS WITH A CIRCLE", W / 2, 815, size=15, font="latin", weight=500,
                      color=(0.8, 0.9, 1.0), alpha=0.6 * b * smooth((t - 226.0) / 0.8), tracking=0.4)
        c = window(t, 227.6, 231.6, 0.8, 1.0)
        if c > 0:
            draw_text(ctx, "本片所有画面由代码实时计算生成 · 所有声音由正弦波叠加合成", W / 2, 960, size=20, font="sans",
                      weight=400, color=WHITE, alpha=0.55 * c, tracking=0.12)
            draw_text(ctx, "EVERY FRAME COMPUTED BY CODE · EVERY SOUND BUILT FROM SINE WAVES", W / 2, 996, size=12,
                      font="latin", weight=500, color=WHITE, alpha=0.35 * c, tracking=0.3)

    def effects(self, t, lt):
        return {"flash": 0.8 * math.exp(-lt / 0.22), "chroma": 4 * math.exp(-lt / 0.45),
                "fade": 1 - smooth((t - 231.0) / 0.9)}
