"""Chapter 3: sound -- a chord unmixed, the cochlea, and the live spectrum of the soundtrack."""
import math

import cairo
import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, VIOLET, W, WHITE, background, circle, clamp, draw_text, ease_out,
                    glow_dot, hsv, lerp, mix, polyline, rounded_rect, set_rgba, smooth, stroke_poly, window)
from ..data import beat_pulse, spec_at, spectrum
from .base import Scene

NOTE_COLS = [CYAN, GOLD, MAGENTA]


class Chord(Scene):
    start, end = 80.0, 88.2
    fade_in = 0.4
    fade_out = 0.5
    bloom = 0.95
    X0, X1 = 300.0, 1620.0

    def env(self, t, i):
        return clamp((t - 80.3 - 0.4 * i) / 0.8) ** 2

    def comp(self, t, i, xs):
        f = S.SOUND_CHORD[i][1]
        tau = (xs - self.X0) / (self.X1 - self.X0) * 0.030 + (t - 80) * 0.004
        return np.sin(2 * np.pi * f * tau)

    def keyboard(self, ctx, t, a):
        # C3..B4 : 14 white keys
        kw, x0, y0, kh = 84.0, 960 - 7 * 84.0, 770.0, 110.0
        whites = ["C", "D", "E", "F", "G", "A", "B"]
        lit = {"A3": 0, "C4": 1, "E4": 2}
        centers = {}
        for o in range(2):
            for j, n in enumerate(whites):
                x = x0 + (o * 7 + j) * kw
                name = f"{n}{3 + o}"
                centers[name] = x + kw / 2
                li = lit.get(name)
                glow = 0.0
                if li is not None:
                    glow = smooth((t - 86.0 - 0.25 * li) / 0.4)
                rounded_rect(ctx, x + 2, y0, kw - 4, kh, 5)
                base = mix((0.75, 0.8, 0.9), NOTE_COLS[li] if li is not None else WHITE, glow)
                set_rgba(ctx, base, a * (0.14 + 0.75 * glow))
                ctx.fill()
        blacks = {0: "C#", 1: "D#", 3: "F#", 4: "G#", 5: "A#"}
        for o in range(2):
            for j in blacks:
                x = x0 + (o * 7 + j + 1) * kw - 22
                rounded_rect(ctx, x, y0, 44, kh * 0.6, 4)
                set_rgba(ctx, (0.02, 0.02, 0.04), a)
                ctx.fill()
                rounded_rect(ctx, x, y0, 44, kh * 0.6, 4)
                set_rgba(ctx, (0.5, 0.6, 0.8), a * 0.3)
                ctx.set_line_width(1)
                ctx.stroke()
        return centers, y0

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        xs = np.linspace(self.X0, self.X1, 900)
        env = [self.env(t, i) for i in range(3)]
        comps = [self.comp(t, i, xs) for i in range(3)]
        # combined wave
        y_top = 270.0
        tot = sum(e * c for e, c in zip(env, comps))
        stroke_poly(ctx, np.stack([xs, y_top - 42 * tot], 1), WHITE, 2.4, 0.95, glow=1.0)
        draw_text(ctx, "混合的声音", self.X0 - 30, y_top - 8, size=24, font="sans", weight=500, color=WHITE,
                  alpha=0.8 * smooth(lt / 1.0), anchor="right")
        draw_text(ctx, "MIXTURE", self.X0 - 30, y_top + 22, size=12, font="latin", weight=500, color=WHITE,
                  alpha=0.5 * smooth(lt / 1.0), anchor="right", tracking=0.3)
        # split into components
        lanes = [430.0, 540.0, 650.0]
        if t > 83.8:
            for i in range(3):
                spi = ease_out((t - 84.0 - 0.15 * i) / 1.2, 3)
                y = lerp(y_top, lanes[i], spi)
                a = smooth((t - 83.8) / 0.4)
                stroke_poly(ctx, np.stack([xs, y - 42 * comps[i]], 1), NOTE_COLS[i], 2.2, 0.9 * a, glow=0.8)
                nm, f = S.SOUND_CHORD[i]
                draw_text(ctx, f"{nm}", self.X0 - 30, y - 8, size=26, font="latin", weight=500, color=NOTE_COLS[i],
                          alpha=a * spi, anchor="right")
                draw_text(ctx, f"{f:.0f} Hz", self.X0 - 30, y + 22, size=14, font="mono", weight=400, color=WHITE,
                          alpha=0.6 * a * spi, anchor="right")
        # keyboard + beams
        ka = smooth((t - 85.3) / 0.6)
        if ka > 0:
            centers, ky = self.keyboard(ctx, t, ka)
            for i, (nm, f) in enumerate(S.SOUND_CHORD):
                b = smooth((t - 86.0 - 0.25 * i) / 0.4)
                if b <= 0:
                    continue
                x = centers[nm]
                g = cairo.LinearGradient(0, lanes[i], 0, ky)
                c = NOTE_COLS[i]
                g.add_color_stop_rgba(0, c[0], c[1], c[2], 0.0)
                g.add_color_stop_rgba(1, c[0], c[1], c[2], 0.55 * b)
                ctx.set_source(g)
                ctx.rectangle(x - 3, lanes[i] + 40, 6, ky - lanes[i] - 40)
                ctx.fill()
                glow_dot(ctx, x, ky - 6, 6, c, b, 6)


class Ear(Scene):
    start, end = 88.0, 96.0
    fade_in = 0.5
    fade_out = 0.3
    bloom = 1.0
    CX, CY = 960.0, 470.0

    def prepare(self):
        n = 700
        u = np.linspace(0, 1, n)          # 0 = apex (low), 1 = base (high)
        turns = 2.6
        phi = u * turns * 2 * np.pi
        r = 40 + 300 * (u ** 0.85)
        self.u = u
        self.px = self.CX + r * np.cos(-phi + 0.6)
        self.py = self.CY + r * np.sin(-phi + 0.6) * 0.92
        # outward normals
        dx, dy = np.gradient(self.px), np.gradient(self.py)
        L = np.hypot(dx, dy) + 1e-9
        self.nx, self.ny = dy / L, -dx / L
        self.width = 10 + 26 * u

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6, hue=(0.03, 0.04, 0.10))
        reveal = ease_out(lt / 1.6, 2)
        n = len(self.u)
        k0 = int((1 - reveal) * n)
        spec = spec_at(t)
        nb = len(spec)
        # tube
        for i in range(max(k0, 1), n - 1, 2):
            polyline(ctx, [(self.px[i], self.py[i]), (self.px[i + 2 if i + 2 < n else i + 1], self.py[i + 2 if i + 2 < n else i + 1])])
            set_rgba(ctx, (0.25, 0.3, 0.55), 0.35)
            ctx.set_line_width(self.width[i])
            ctx.stroke()
        # activation
        for i in range(max(k0, 1), n - 1):
            b = int(self.u[i] * (nb - 1))
            v = float(spec[b]) ** 1.3
            if v < 0.02:
                continue
            col = hsv(0.0 + 0.78 * self.u[i], 0.8, 1.0)
            polyline(ctx, [(self.px[i], self.py[i]), (self.px[i + 1], self.py[i + 1])])
            set_rgba(ctx, col, min(1.0, v * 1.4))
            ctx.set_line_width(self.width[i] * 0.8)
            ctx.stroke()
            # hair-cell ticks
            if i % 3 == 0:
                L = 70 * v
                ctx.move_to(self.px[i] + self.nx[i] * self.width[i] * 0.5, self.py[i] + self.ny[i] * self.width[i] * 0.5)
                ctx.line_to(self.px[i] + self.nx[i] * (self.width[i] * 0.5 + L), self.py[i] + self.ny[i] * (self.width[i] * 0.5 + L))
                set_rgba(ctx, col, 0.7 * v)
                ctx.set_line_width(1.5)
                ctx.stroke()
        a = smooth((lt - 1.0) / 0.8)
        draw_text(ctx, "低音", self.CX + 5, self.CY + 5, size=22, font="sans", weight=500, color=(1, 0.5, 0.4),
                  alpha=0.9 * a)
        draw_text(ctx, "LOW", self.CX + 5, self.CY + 32, size=12, font="latin", weight=500, color=WHITE, alpha=0.5 * a,
                  tracking=0.3)
        ex, ey = self.px[-1], self.py[-1]
        draw_text(ctx, "高音", ex + 40, ey - 10, size=22, font="sans", weight=500, color=(0.7, 0.5, 1.0),
                  alpha=0.9 * a, anchor="left")
        draw_text(ctx, "HIGH", ex + 40, ey + 18, size=12, font="latin", weight=500, color=WHITE, alpha=0.5 * a,
                  anchor="left", tracking=0.3)
        # side panel: live spectrum bars
        xs0, xs1, yb = 1380.0, 1780.0, 700.0
        if a > 0:
            nbar = 64
            bw = (xs1 - xs0) / nbar
            for j in range(nbar):
                v = float(spec[int(j / nbar * nb)]) ** 1.1
                col = hsv(0.78 * j / nbar, 0.8, 1.0)
                set_rgba(ctx, col, 0.85 * a)
                ctx.rectangle(xs0 + j * bw, yb - 180 * v, bw * 0.7, 180 * v)
                ctx.fill()
            draw_text(ctx, "频谱 · 实时", xs0, yb + 30, size=18, font="sans", weight=500, color=WHITE, alpha=0.7 * a,
                      anchor="left")
        # small 'ear' label
        draw_text(ctx, "耳蜗", 240, 300, size=64, font="serif", weight=700, color=WHITE, alpha=0.9 * a, anchor="left",
                  tracking=0.2, glow=8, glow_alpha=0.3)
        draw_text(ctx, "COCHLEA", 244, 360, size=16, font="latin", weight=500, color=GOLD, alpha=0.7 * a,
                  anchor="left", tracking=0.5)


class Drop(Scene):
    start, end = 96.0, 111.6
    fade_in = 0.1
    fade_out = 0.6
    bloom = 1.0

    def prepare(self):
        rng = np.random.default_rng(96)
        self.pang = rng.uniform(0, 2 * np.pi, 600)
        self.pspd = rng.uniform(200, 700, 600)
        self.pbirth = rng.uniform(0, 1, 600)

    def ridges(self, ctx, t, a):
        step = 0.1
        nr = 46
        base_i = math.floor(t / step)
        frac = t / step - base_i
        nb = spectrum().shape[1]
        xs = np.linspace(300, 1620, nb)
        center = np.exp(-((xs - 960) / 520) ** 2)
        for j in range(nr - 1, -1, -1):
            pos = j + frac
            tj = (base_i - j) * step
            if tj < 95.5:
                continue
            sp = spec_at(tj) ** 1.2
            depth = pos / nr
            y0 = 900 - pos * 14.0
            amp = 170 * (1 - 0.35 * depth)
            ys = y0 - amp * sp * (0.35 + 0.65 * center)
            pts = np.stack([xs, ys], 1)
            # occlusion fill
            polyline(ctx, pts)
            ctx.line_to(xs[-1], y0 + 40)
            ctx.line_to(xs[0], y0 + 40)
            ctx.close_path()
            ctx.set_source_rgb(0.006, 0.008, 0.02)
            ctx.fill()
            col = mix(mix(MAGENTA, GOLD, 0.3), CYAN, depth ** 0.8)
            fade_edge = min(1.0, (nr - pos) / 4.0)
            stroke_poly(ctx, pts, col, 2.0 if j < 3 else 1.5, a * (1 - 0.75 * depth) * fade_edge)

    def radial(self, ctx, t, a):
        spec = spec_at(t)
        nb = len(spec)
        bp = beat_pulse(t)
        cx, cy = 960.0, 480.0
        R = 190 + 22 * bp
        nbar = 160
        for j in range(nbar):
            u = j / nbar
            k = int((1 - abs(2 * u - 1)) * (nb * 0.8))
            v = float(spec[k]) ** 1.1
            ang = -math.pi / 2 + u * 2 * math.pi + 0.15 * t
            L = 20 + 260 * v
            col = hsv(0.55 + 0.45 * (1 - abs(2 * u - 1)), 0.75, 1.0)
            ctx.move_to(cx + R * math.cos(ang), cy + R * math.sin(ang))
            ctx.line_to(cx + (R + L) * math.cos(ang), cy + (R + L) * math.sin(ang))
            set_rgba(ctx, col, a * (0.5 + 0.5 * v))
            ctx.set_line_width(5)
            ctx.stroke()
        circle(ctx, cx, cy, R - 12, WHITE, a * (0.35 + 0.5 * bp), width=2)
        circle(ctx, cx, cy, R - 30, CYAN, a * 0.2, width=1)
        # particles on beats
        for ang, spd, b in zip(self.pang, self.pspd, self.pbirth):
            age = (t * 0.8 + b) % 1.0
            d = R + spd * age
            x, y = cx + d * math.cos(ang), cy + d * math.sin(ang)
            ctx.arc(x, y, 1.6, 0, 2 * math.pi)
            set_rgba(ctx, mix(CYAN, MAGENTA, b), a * (1 - age) * 0.7)
            ctx.fill()
        # the inner core shows the waveform (sum of the loudest bins as sines)
        top = np.argsort(spec)[-6:]
        th = np.linspace(0, 2 * np.pi, 240)
        rr = R - 60 + 0 * th
        for k in top:
            rr = rr + 7 * float(spec[k]) ** 2 * np.sin(th * (3 + k % 6) + t * (2 + k % 5))
        stroke_poly(ctx, np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1), GOLD, 2.0, a * 0.8, close=True)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.8, strength=1.0 + 0.6 * beat_pulse(t))
        a1 = 1 - smooth((t - 103.4) / 0.6)
        a2 = smooth((t - 103.6) / 0.6)
        if a1 > 0:
            self.ridges(ctx, t, a1)
        if a2 > 0:
            self.radial(ctx, t, a2)
        draw_text(ctx, "LIVE", 1800, 70, size=14, font="latin", weight=600, color=(1, 0.3, 0.35),
                  alpha=0.8 * (0.6 + 0.4 * math.sin(t * 6)), anchor="right", tracking=0.4)

    def effects(self, t, lt):
        return {"flash": 0.3 * math.exp(-max(0.0, t - 96.0) / 0.2), "chroma": 1.5 * beat_pulse(t, 0.1)}
