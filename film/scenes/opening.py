"""Cold open (0-16 s) and title (16-24 s)."""
import math

import cairo
import numpy as np

from ..core import (CYAN, GOLD, H, MAGENTA, VIOLET, W, WHITE, background, circle, clamp, draw_math, draw_text,
                    draw_text_chars, ease_out, ease_out_expo, glow_dot, hsv, lerp, mix, polyline, remap, set_rgba,
                    smooth, stroke_poly, window)
from .. import story as S
from .base import Scene


class ColdOpen(Scene):
    start, end = 0.0, 11.5
    bloom = 1.0
    T0 = 2.2      # the dot starts to move
    TH = 6.0      # the extra circles appear
    TG = 8.6      # tension / glitch starts
    C = (600.0, 520.0)
    R = 150.0
    X0 = 980.0
    V = 170.0

    def amps(self, t):
        h = smooth((t - self.TH) / 1.0)
        return [(1, 1.0, 0.0), (2, 0.5 * h, 0.4), (3, 0.33 * h, 1.1)]

    def theta(self, t):
        return 2 * math.pi * 0.5 * (t - self.T0)

    def tip(self, t, with_chain=False):
        cx, cy = self.C
        pts = [(cx, cy)]
        th = self.theta(t)
        x, y = cx, cy
        for k, a, ph in self.amps(t):
            x += self.R * a * math.cos(k * th + ph)
            y -= self.R * a * math.sin(k * th + ph)
            pts.append((x, y))
        return pts if with_chain else (x, y)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        collapse = smooth((t - 10.7) / 0.7)
        background(ctx, t, strength=smooth(t / 2.0), dust=smooth(t / 1.8) * (1 - collapse))
        glitch = smooth((t - self.TG) / 2.2)
        dim = 1 - 0.7 * window(t, 9.0, 11.4, 0.4, 0.2)
        a_all = (1 - collapse) * dim
        cx, cy = self.C
        # --- the first dot
        born = smooth((t - 0.5) / 0.7)
        if t < self.T0:
            ring = smooth((t - 1.4) / 0.8)
            if ring > 0:
                ctx.new_path()
                ctx.arc_negative(cx, cy, self.R, 0, -2 * math.pi * ring)
                set_rgba(ctx, CYAN, 0.55 * ring)
                ctx.set_line_width(2)
                ctx.stroke()
            glow_dot(ctx, cx + self.R, cy, 5 + 2 * math.sin(t * 5), WHITE, born, 6)
            return
        chain = self.tip(t, True)
        # grid axis
        set_rgba(ctx, (0.4, 0.6, 1.0), 0.10 * a_all)
        ctx.set_line_width(1)
        ctx.move_to(self.X0, cy)
        ctx.line_to(W - 60, cy)
        ctx.stroke()
        # circles
        for i, (k, a, ph) in enumerate(self.amps(t)):
            if a < 1e-3:
                continue
            px, py = chain[i]
            col = [CYAN, VIOLET, MAGENTA][i]
            circle(ctx, px, py, self.R * a, col, 0.55 * a_all, width=2.0 if i == 0 else 1.6)
            ctx.move_to(px, py)
            ctx.line_to(*chain[i + 1])
            set_rgba(ctx, WHITE, 0.8 * a_all)
            ctx.set_line_width(2)
            ctx.stroke()
        tx, ty = chain[-1]
        # projection line
        ctx.set_dash([6, 8])
        set_rgba(ctx, WHITE, 0.35 * a_all)
        ctx.set_line_width(1.2)
        ctx.move_to(tx, ty)
        ctx.line_to(self.X0, ty)
        ctx.stroke()
        ctx.set_dash([])
        # wave trail
        span = (W - self.X0) / self.V
        ts = np.linspace(max(self.T0, t - span), t, 400)
        pts = []
        for tt in ts:
            _, y = self.tip(tt)
            pts.append((self.X0 + (t - tt) * self.V, y))
        pts = np.array(pts)
        col = mix(GOLD, (1.0, 0.25, 0.3), glitch)
        # a little jitter as tension rises
        if glitch > 0:
            rng = np.random.default_rng(int(t * 30))
            pts[:, 1] += rng.normal(0, 6 * glitch ** 2, len(pts))
        stroke_poly(ctx, pts, col, 3.0, 0.95 * a_all, glow=1.0)
        glow_dot(ctx, self.X0, ty, 5, col, a_all, 5)
        glow_dot(ctx, tx, ty, 5, WHITE, a_all, 5)
        # final collapse into a point
        if collapse > 0:
            glow_dot(ctx, cx, cy, 3 + 10 * (1 - collapse), WHITE, 1 - collapse, 8)
        # Lagrange
        a = window(t, 9.0, 11.4, 0.3, 0.4)
        if a > 0:
            draw_text_chars(ctx, "不可能。", W / 2, H / 2 - 20, t - 9.0, size=96, font="serif", weight=800,
                            color=WHITE, alpha=a, tracking=0.02, stagger=0.08, dur=0.3, rise=10, glow=14, glow_alpha=0.6)
            draw_text(ctx, "—— 拉格朗日  J.-L. LAGRANGE", W / 2 + 180, H / 2 + 80, size=32, font="sans", weight=400,
                      color=(0.85, 0.85, 0.9), alpha=a * smooth((t - 9.6) / 0.4), anchor="center", tracking=0.1)
            draw_text(ctx, "论文被搁置 15 年，直到 1822 年才以《热的解析理论》出版", W / 2, H / 2 + 150, size=30,
                      font="sans", weight=400, color=GOLD, alpha=0.85 * a * smooth((t - 9.5) / 0.3), tracking=0.06)

    def effects(self, t, lt):
        g = smooth((t - self.TG) / 2.2)
        return {"chroma": 3.0 * g ** 2, "fade": 1 - smooth((t - 11.0) / 0.4)}


class Title(Scene):
    start, end = 12.0, 18.0
    fade_out = 0.5
    bloom = 0.8

    def prepare(self):
        rng = np.random.default_rng(16)
        n = 420
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.gamma(2.0, 260, n)
        self.parts = (ang, spd, rng.uniform(0.6, 2.6, n), rng.uniform(0, 1, n))

    def silk(self, ctx, t, lt):
        grow = ease_out(lt / 2.5)
        n = 40
        xs = np.linspace(-40, W + 40, 220)
        env = np.exp(-((xs - W / 2) / (W * 0.36)) ** 2)
        for i in range(n):
            u = i / (n - 1)
            y0 = H / 2 + (u - 0.5) * 420 * grow
            ph = 1.1 * t + i * 0.22
            ys = y0 + 110 * grow * env * np.sin(xs * 0.0042 * (1 + 0.25 * math.sin(i * 0.7)) + ph) \
                + 28 * env * np.sin(xs * 0.011 - 1.7 * t + i * 0.5)
            col = mix(CYAN, MAGENTA, u)
            polyline(ctx, np.stack([xs, ys], 1))
            set_rgba(ctx, col, (0.14 + 0.14 * math.sin(i * 0.9 + t) ** 2) * grow)
            ctx.set_line_width(1.6)
            ctx.stroke()

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, strength=1.2, hue=(0.04, 0.05, 0.12))
        out = smooth((t - 16.9) / 0.7)
        ctx.push_group()
        self.silk(ctx, t, lt)
        # shock rings
        for k, d in enumerate([0.0, 0.12, 0.3]):
            u = lt - d
            if 0 < u < 2.5:
                r = 60 + 900 * ease_out(u / 2.5, 2)
                circle(ctx, W / 2, H / 2, r, mix(WHITE, CYAN, k / 2), 0.5 * (1 - u / 2.5) ** 2, width=3 - k)
        # particles
        ang, spd, sz, hue = self.parts
        u = lt
        if u < 4.0:
            d = spd * (1 - np.exp(-u * 1.6)) / 1.6
            xs = W / 2 + np.cos(ang) * d
            ys = H / 2 + np.sin(ang) * d * 0.6
            a = np.exp(-u / 1.4)
            for x, y, s, h in zip(xs, ys, sz, hue):
                c = mix(CYAN, GOLD, h)
                ctx.arc(x, y, s, 0, 2 * math.pi)
                set_rgba(ctx, c, a * 0.8)
                ctx.fill()
        # horizontal flare
        fl = math.exp(-lt / 0.6)
        if fl > 0.01:
            g = cairo.LinearGradient(0, 0, W, 0)
            g.add_color_stop_rgba(0, 0.5, 0.8, 1, 0)
            g.add_color_stop_rgba(0.5, 0.8, 0.95, 1, 0.9 * fl)
            g.add_color_stop_rgba(1, 0.5, 0.8, 1, 0)
            ctx.set_source(g)
            ctx.rectangle(0, H / 2 - 2, W, 4)
            ctx.fill()
        # title
        s = 1.12 - 0.12 * ease_out_expo(lt / 1.6)
        a = smooth(lt / 0.25)
        draw_text(ctx, "傅里叶变换", W / 2, H / 2 - 40, size=150, font="serif", weight=900, color=WHITE, alpha=a,
                  tracking=0.28, glow=16, glow_alpha=0.45, scale=s)
        draw_text(ctx, "THE FOURIER TRANSFORM", W / 2, H / 2 + 105, size=34, font="latin", weight=300,
                  color=(0.8, 0.92, 1.0), alpha=0.9, tracking=0.85, reveal=ease_out((lt - 0.9) / 1.2, 2))
        draw_text_chars(ctx, "万物，皆是波的叠加", W / 2, H / 2 + 185, lt - 2.0, size=34, font="serif", weight=500,
                        color=GOLD, alpha=0.95, tracking=0.18, stagger=0.07, dur=0.5, glow=8, glow_alpha=0.5)
        draw_math(ctx, r"$\hat{f}(\omega)=\int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt$", W / 2, H / 2 + 290,
                  size=40, color=(0.8, 0.9, 1.0), alpha=0.95, reveal=ease_out((lt - 2.9) / 1.4, 2), glow=6)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(1 - out)

    def effects(self, t, lt):
        return {"flash": 0.85 * math.exp(-lt / 0.22), "chroma": 4.0 * math.exp(-lt / 0.5)}
