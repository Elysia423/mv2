"""Chapter 2: drawing with circles -- convergence panels, a 3D butterfly, quick draws, Ptolemy's epicycles."""
import math

import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, ORANGE, RED, VIOLET, W, WHITE, background, circle, clamp, draw_math,
                    draw_text, ease_out, glow_dot, hsv, lerp, mix, polyline, set_rgba, smooth, smoother, stroke_poly,
                    window)
from ..epicycles import glyph_epicycles, pen_runs
from ..three import Camera, line3
from .base import Scene

CLEF = "\U0001D11E"
BUTTERFLY = "🦋"
QUICK = ["🎻", "🧠", "🌍"]


class Panels(Scene):
    start, end = S.T_PANELS, S.T_BUTTERFLY
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.95
    PANELS = [8, 24, 80, 400]

    def prepare(self):
        self.clef = glyph_epicycles(CLEF, "NotoMusic.ttf", 300.0, 2048)
        for m in self.PANELS:
            self.clef.curve(m)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6)
        s_prog = clamp((lt - 0.4) / 4.3)
        E = self.clef
        xs = [300, 740, 1180, 1620]
        circ_a = 1 - smooth((lt - 4.8) / 0.7)
        for p, (m, px) in enumerate(zip(self.PANELS, xs)):
            py = 470.0
            sc = 0.82
            ap = smooth((lt - 0.1 * p) / 0.4)
            set_rgba(ctx, (0.4, 0.6, 1.0), 0.10 * ap)
            ctx.set_line_width(1)
            ctx.rectangle(px - 200, py - 300, 400, 600)
            ctx.stroke()
            z = E.curve(m)
            nseg = int(s_prog * E.n)
            Z = px + sc * z.real + 1j * (py + sc * z.imag)
            col = mix(GOLD, WHITE, 0.25 * (p / 3))
            for a, b in pen_runs(E.pen[:nseg + 1]):
                b = min(b, nseg + 1)
                if b - a > 1:
                    stroke_poly(ctx, np.stack([Z.real[a:b], Z.imag[a:b]], 1), col, 2.2, 0.95 * ap, glow=0.8)
            if circ_a > 0.01:
                ch = E.chain(s_prog % 1.0, m)
                chx, chy = px + sc * ch.real, py + sc * ch.imag
                r = sc * E.radii(m)
                show = min(m, 120)
                for i in range(show):
                    if r[i] < 0.4:
                        continue
                    u = i / show
                    circle(ctx, chx[i], chy[i], r[i], mix(CYAN, MAGENTA, u), 0.35 * ap * circ_a * (1 - 0.7 * u), width=1.0)
                polyline(ctx, np.stack([chx[:show + 1], chy[:show + 1]], 1))
                set_rgba(ctx, WHITE, 0.6 * ap * circ_a)
                ctx.set_line_width(1.0)
                ctx.stroke()
                glow_dot(ctx, chx[-1], chy[-1], 3, WHITE, ap * circ_a, 4)
            draw_text(ctx, f"{m}", px - 8, py + 340, size=40, font="latin", weight=300, color=WHITE, alpha=ap,
                      anchor="right")
            draw_text(ctx, "个圆", px + 4, py + 342, size=30, font="sans", weight=500, color=GOLD, alpha=ap,
                      anchor="left")


class Butterfly3D(Scene):
    """600 epicycles; the pen trail is extruded along time, the camera orbits, then it flattens to 2D."""
    start, end = S.T_BUTTERFLY, S.T_QUICK
    fade_in = 0.2
    fade_out = 0.3
    bloom = 1.0
    M = 600
    SC = 1 / 64.0

    def prepare(self):
        self.E = glyph_epicycles(BUTTERFLY, "NotoEmoji.ttf", 300.0, 4096, 0.3)
        self.curve = self.E.curve(self.M)
        n = self.E.n
        self.hue = np.array([hsv(0.5 + 0.5 * i / n, 0.7, 1.0) for i in range(n)])

    def s_of(self, lt):
        return clamp((lt - 0.3) / 7.4) ** 1.1

    def depth(self, lt):
        return 9.0 * (1 - smoother((lt - 5.2) / 2.2))

    def cam(self, lt):
        u1 = smoother((lt - 1.6) / 3.4)
        u2 = smoother((lt - 5.0) / 2.4)
        yaw = lerp(0.95, -0.75, u1)
        yaw = lerp(yaw, 0.0, u2)
        pitch = lerp(0.42, 0.18, u1)
        pitch = lerp(pitch, 0.0, u2)
        dist = lerp(10.5, 12.0, u1)
        dist = lerp(dist, 10.0, u2)
        tz = lerp(-2.5, -3.5, u1) * (1 - u2)
        return Camera(yaw, pitch, dist, target=(0, -0.3, tz), fov=40)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6)
        E = self.E
        cam = self.cam(lt)
        s = self.s_of(lt)
        D = self.depth(lt)
        n = E.n
        k = max(2, int(s * n))
        idx = np.arange(k)
        z = self.curve[:k]
        P = np.stack([z.real * self.SC, -z.imag * self.SC, -(s - idx / n) * D], 1)
        sx, sy, sz = cam.project(P)
        for a, b in pen_runs(E.pen[:k]):
            for c0 in range(a, b - 1, 24):
                c1 = min(b, c0 + 25)
                col = self.hue[c0]
                polyline(ctx, np.stack([sx[c0:c1], sy[c0:c1]], 1))
                set_rgba(ctx, col, 0.95)
                ctx.set_line_width(2.4)
                ctx.stroke()
        # the chain lives in the plane z = 0
        chain_a = 1 - smooth((lt - 7.9) / 0.6)
        if chain_a > 0.01:
            ch = E.chain(s % 1.0, self.M)
            C = np.stack([ch.real * self.SC, -ch.imag * self.SC, np.zeros(len(ch))], 1)
            cx, cy, _ = cam.project(C)
            r = E.radii(self.M) * self.SC
            ang = np.linspace(0, 2 * np.pi, 40)
            for i in range(min(160, self.M)):
                if r[i] < 0.004:
                    continue
                ring = np.stack([C[i, 0] + r[i] * np.cos(ang), C[i, 1] + r[i] * np.sin(ang), np.zeros_like(ang)], 1)
                u = i / 160
                line3(ctx, cam, ring, mix(CYAN, VIOLET, u), 1.0, chain_a * (0.4 - 0.3 * u))
            polyline(ctx, np.stack([cx, cy], 1))
            set_rgba(ctx, WHITE, 0.55 * chain_a)
            ctx.set_line_width(1.0)
            ctx.stroke()
            glow_dot(ctx, cx[-1], cy[-1], 4, WHITE, chain_a, 5)
        # time axis hint while extruded
        ax = smooth((lt - 1.0) / 0.6) * (1 - smooth((lt - 5.0) / 1.0))
        if ax > 0:
            line3(ctx, cam, np.array([[3.2, -2.8, 0], [3.2, -2.8, -D]]), WHITE, 1.2, 0.4 * ax)
            px, py, _ = cam.project(np.array([[3.3, -2.9, -D * 0.6]]))
            draw_text(ctx, "时间 →", px[0] + 10, py[0], size=28, font="sans", weight=500, color=WHITE, alpha=0.7 * ax,
                      anchor="left")
        a = window(lt, 0.2, 9.8, 0.4, 0.3)
        draw_text(ctx, f"N = {self.M}", 1830, 960 - 70, size=36, font="latin", weight=300, color=WHITE, alpha=0.75 * a,
                  anchor="right", tracking=0.1)
        draw_text(ctx, f"{s * 100:5.1f}%", 1830, 960 - 30, size=28, font="mono", weight=300, color=GOLD,
                  alpha=0.7 * a, anchor="right")

    def effects(self, t, lt):
        return {"flash": 0.25 * math.exp(-max(0.0, t - 50.0) / 0.25) if t >= 50.0 else 0.0,
                "chroma": 2.0 * math.exp(-max(0.0, t - 50.0) / 0.4) if t >= 50.0 else 0.0}


class QuickDraws(Scene):
    start, end = S.T_QUICK, S.T_PTOLEMY
    fade_in = 0.2
    fade_out = 0.3
    bloom = 1.0
    M = 400
    XS = [380, 960, 1540]

    def prepare(self):
        self.Es = [glyph_epicycles(ch, "NotoEmoji.ttf", 300.0, 2048, 0.3) for ch in QUICK]
        for E in self.Es:
            E.curve(self.M)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6)
        sc = 1.3
        cols = [GOLD, MAGENTA, (0.45, 1.0, 0.6)]
        for i, (E, px) in enumerate(zip(self.Es, self.XS)):
            t0 = 0.1 + i * 1.2
            s = clamp((lt - t0) / 1.15)
            if lt < t0:
                continue
            py = 400.0
            z = E.curve(self.M)
            k = int(s * E.n)
            X, Y = px + sc * z.real, py + sc * z.imag
            for a, b in pen_runs(E.pen[:k + 1]):
                b = min(b, k + 1)
                if b - a > 1:
                    stroke_poly(ctx, np.stack([X[a:b], Y[a:b]], 1), cols[i], 2.0, 0.95, glow=0.7)
            if s < 1:
                ch = E.chain(s, self.M)
                cx, cy = px + sc * ch.real, py + sc * ch.imag
                r = sc * E.radii(self.M)
                for j in range(60):
                    if r[j] > 0.5:
                        circle(ctx, cx[j], cy[j], r[j], WHITE, 0.25 * (1 - j / 60), width=1.0)
                polyline(ctx, np.stack([cx[:120], cy[:120]], 1))
                set_rgba(ctx, WHITE, 0.5)
                ctx.set_line_width(1)
                ctx.stroke()
                glow_dot(ctx, cx[-1], cy[-1], 3.5, WHITE, 1.0, 5)
        a = smooth((lt - 0.4) / 0.5)
        draw_math(ctx, r"$X_k=\sum_{n=0}^{N-1} x_n\, e^{-2\pi i\,kn/N}$", W / 2, 760, size=46, color=WHITE, alpha=a,
                  glow=8)
        draw_text(ctx, "每个 X_k 就是一个圆", W / 2, 858, size=32, font="sans", weight=500, color=GOLD,
                  alpha=0.9 * smooth((lt - 0.9) / 0.5), tracking=0.04)


class Ptolemy(Scene):
    """Geocentric Mars = deferent (Mars around the Sun) + epicycle (Sun around the Earth)."""
    start, end = S.T_PTOLEMY, S.T_CHORD
    fade_in = 0.25
    fade_out = 0.3
    bloom = 1.0
    RM, PM = 1.524, 1.881

    def pos(self, yr):
        m = self.RM * np.exp(2j * np.pi * yr / self.PM + 0.6j)
        e = np.exp(2j * np.pi * yr + 0.2j)
        return m, e

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.8, hue=(0.03, 0.03, 0.08))
        yrs = max(0.0, (lt - 0.2) * 2.6)
        cam = Camera(-0.35 + 0.12 * lt, 0.62 - 0.03 * lt, 7.0, target=(0.1, 0, 0.35), fov=40, center=(W / 2, 470))
        ys = np.linspace(0, yrs, max(2, int(yrs * 120)))
        m, e = self.pos(ys)
        geo = m - e
        P = np.stack([geo.real, np.zeros(len(ys)), geo.imag], 1)
        sx, sy, _ = cam.project(P)
        # colour retrograde stretches (where the geocentric longitude decreases)
        lon = np.unwrap(np.angle(geo))
        retro = np.diff(lon, prepend=lon[0]) < 0
        for i in range(0, len(ys) - 1, 6):
            j = min(len(ys), i + 7)
            col = RED if retro[i] else GOLD
            polyline(ctx, np.stack([sx[i:j], sy[i:j]], 1))
            set_rgba(ctx, col, 0.95)
            ctx.set_line_width(2.6)
            ctx.stroke()
        # deferent + epicycle chain from the Earth
        mm, ee = self.pos(np.array([yrs]))
        d_vec = mm[0]
        ang = np.linspace(0, 2 * np.pi, 100)
        line3(ctx, cam, np.stack([self.RM * np.cos(ang), 0 * ang, self.RM * np.sin(ang)], 1), CYAN, 1.3, 0.45)
        cen = d_vec
        line3(ctx, cam, np.stack([cen.real + np.cos(ang), 0 * ang, cen.imag + np.sin(ang)], 1), VIOLET, 1.3, 0.55)
        pts = np.array([[0, 0, 0], [d_vec.real, 0, d_vec.imag], [geo[-1].real, 0, geo[-1].imag]])
        px, py, _ = cam.project(pts)
        polyline(ctx, np.stack([px, py], 1))
        set_rgba(ctx, WHITE, 0.85)
        ctx.set_line_width(1.8)
        ctx.stroke()
        glow_dot(ctx, px[0], py[0], 7, (0.4, 0.7, 1.0), 1.0, 5)
        glow_dot(ctx, px[2], py[2], 6, ORANGE, 1.0, 5)
        draw_text(ctx, "地球", px[0], py[0] + 34, size=28, font="sans", weight=500, color=(0.6, 0.8, 1.0))
        draw_text(ctx, "火星", px[2] + 16, py[2] - 18, size=28, font="sans", weight=500, color=ORANGE, anchor="left")
        lx, ly, _ = cam.project(np.array([[self.RM * 0.72, 0, -self.RM * 0.72], [cen.real + 0.72, 0, cen.imag - 0.72]]))
        draw_text(ctx, "均轮", lx[0], ly[0] - 14, size=30, font="sans", weight=500, color=CYAN, alpha=0.85)
        draw_text(ctx, "本轮", lx[1], ly[1] - 14, size=30, font="sans", weight=500, color=VIOLET, alpha=0.85)
        a = smooth((lt - 0.8) / 0.5)
        draw_text(ctx, "逆行", 1700, 200, size=40, font="serif", weight=700, color=RED, alpha=a, anchor="right",
                  glow=8, glow_alpha=0.4)
        draw_text(ctx, "RETROGRADE", 1700, 245, size=22, font="latin", weight=600, color=WHITE, alpha=0.6 * a,
                  anchor="right", tracking=0.18)
        draw_text(ctx, "约公元 150 年", 1700, 285, size=30, font="sans", weight=400, color=GOLD,
                  alpha=0.8 * a, anchor="right")


def drawing_scenes():
    return [Panels(), Butterfly3D(), QuickDraws(), Ptolemy()]
