"""Chapter 1 (circles -> waves) and chapter 2 (drawing with circles)."""
import math

import cairo
import numpy as np
from scipy.interpolate import PchipInterpolator

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, RED, VIOLET, W, WHITE, background, circle, clamp, draw_text, ease_out,
                    glow_dot, hsv, lerp, mix, polyline, remap, set_rgba, smooth, stroke_poly, window)
from ..data import beat_pulse
from ..epicycles import glyph_contours, glyph_epicycles, pen_runs
from .base import Scene


# ====================================================================== chapter 1
def _turn_on_times():
    """Time at which each harmonic's circle appears (odd ones follow the count keys; even ones join for the saw)."""
    ton = np.full(S.KMAX, S.C1_SAW - 0.3)
    for m in range(1, S.KMAX // 2 + 1):
        for tk, c in S.C1_COUNT_KEYS:
            if c >= m:
                ton[2 * m - 2] = tk
                break
    return ton


class Circles(Scene):
    start, end = 24.0, 48.0
    fade_in = 0.6
    fade_out = 0.5
    bloom = 0.9
    CX, CY = 560.0, 500.0
    X0 = 1150.0
    V = 180.0

    def prepare(self):
        self.ton = _turn_on_times()

    def gains(self, tt):
        """(n_t, KMAX) growth factor for each harmonic at times tt."""
        tt = np.atleast_1d(tt)[:, None]
        return np.clip((tt - self.ton[None, :]) / 0.3, 0, 1) ** 2 * (3 - 2 * np.clip((tt - self.ton[None, :]) / 0.3, 0, 1))

    def scale(self, t):
        we = S.c1_shape_weights(t)[2]
        return lerp(140.0, 210.0, we)

    def coeffs(self, tt):
        return np.array([S.c1_coeffs(x) for x in np.atleast_1d(tt)]) * self.gains(tt)

    def tip_y(self, tt):
        th = np.array([S.c1_theta(x) for x in tt])
        c = self.coeffs(tt)
        y = np.sum(np.imag(c * np.exp(1j * np.outer(th, S.C1_K))), axis=1)
        we = np.array([S.c1_shape_weights(x)[2] for x in tt])
        return y + we * S.C1_ECG_DC

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6)
        ws, wa, we = S.c1_shape_weights(t)
        A = self.scale(t)
        cy = self.CY + 60 * we
        # ecg paper grid
        if we > 0:
            set_rgba(ctx, (1, 0.3, 0.35), 0.08 * we)
            ctx.set_line_width(1)
            for x in np.arange(self.X0, W, 40):
                ctx.move_to(x, 200)
                ctx.line_to(x, 820)
            for y in np.arange(200, 821, 40):
                ctx.move_to(self.X0, y)
                ctx.line_to(W - 40, y)
            ctx.stroke()
        # axis
        set_rgba(ctx, (0.5, 0.7, 1.0), 0.18)
        ctx.set_line_width(1)
        ctx.move_to(self.X0, cy)
        ctx.line_to(W - 40, cy)
        ctx.stroke()
        # chain
        th = S.c1_theta(t)
        c = self.coeffs([t])[0]
        v = c * np.exp(1j * S.C1_K * th)
        pts = np.concatenate([[0], np.cumsum(v)])
        dc = we * S.C1_ECG_DC
        X = self.CX + A * pts.real
        Y = cy - A * (pts.imag + dc)
        rad = A * np.abs(c)
        order = np.where(rad > 0.3)[0]
        for j, i in enumerate(order):
            u = j / max(1, len(order) - 1)
            col = mix(CYAN, MAGENTA, u ** 0.7) if we < 0.5 else mix(MAGENTA, RED, u)
            circle(ctx, X[i], Y[i], rad[i], col, 0.55 * (1 - 0.6 * u), width=1.8 if j < 3 else 1.1)
        # radius vectors
        polyline(ctx, np.stack([X[[0] + list(order + 1)], Y[[0] + list(order + 1)]], 1))
        set_rgba(ctx, WHITE, 0.85)
        ctx.set_line_width(1.6)
        ctx.stroke()
        tx, ty = X[-1], Y[-1]
        # projection line
        ctx.set_dash([6, 8])
        set_rgba(ctx, WHITE, 0.35)
        ctx.set_line_width(1.2)
        ctx.move_to(tx, ty)
        ctx.line_to(self.X0, ty)
        ctx.stroke()
        ctx.set_dash([])
        # target ghost (ideal square / saw / ecg)
        span = (W - 40 - self.X0) / self.V
        ts = np.linspace(max(24.0, t - span), t, 700)
        ys = self.tip_y(ts)
        xs = self.X0 + (t - ts) * self.V
        As = np.array([self.scale(x) for x in ts])
        cys = np.array([self.CY + 60 * S.c1_shape_weights(x)[2] for x in ts])
        ghost_a = window(t, 31.0, 43.5, 1.0, 0.8)
        if ghost_a > 0:
            thg = np.array([S.c1_theta(x) for x in ts])
            ideal = np.where(np.sin(thg) >= 0, 1.0, -1.0)
            ideal_saw = ((thg / np.pi + 1) % 2) - 1
            wsg = np.array([S.c1_shape_weights(x)[0] for x in ts])
            ig = wsg * ideal + (1 - wsg) * ideal_saw
            ctx.set_dash([4, 7])
            stroke_poly(ctx, np.stack([xs, cys - As * ig], 1), WHITE, 1.2, 0.22 * ghost_a)
            ctx.set_dash([])
        col = mix(mix(GOLD, (1.0, 0.85, 0.4), wa), (1.0, 0.25, 0.3), we)
        bp = beat_pulse(t)
        hb = 0.0
        for tb in S.heart_beats():
            hb = max(hb, math.exp(-max(0.0, t - tb) / 0.18) if t >= tb else 0.0)
        stroke_poly(ctx, np.stack([xs, cys - As * ys], 1), col, 3.0 + 1.5 * hb, 0.95, glow=1.0 + hb)
        glow_dot(ctx, self.X0, ty, 5 + 3 * hb, col, 1.0, 5)
        glow_dot(ctx, tx, ty, 4 + 2 * bp, WHITE, 1.0, 5)
        # counter
        n = S.c1_count(t)
        a_cnt = window(t, 24.6, 42.8, 0.6, 0.5)
        if a_cnt > 0:
            chg = max([math.exp(-(t - tk) / 0.25) for tk, _ in S.C1_COUNT_KEYS if t >= tk] + [0])
            draw_text(ctx, str(n), 1650, 170, size=110 + 14 * chg, font="latin", weight=200, color=WHITE,
                      alpha=a_cnt, anchor="right", glow=6, glow_alpha=0.5 * chg)
            draw_text(ctx, "个圆", 1668, 150, size=26, font="sans", weight=500, color=GOLD, alpha=a_cnt, anchor="left")
            draw_text(ctx, "CIRCLES", 1668, 186, size=14, font="latin", weight=500, color=(0.7, 0.8, 1.0),
                      alpha=a_cnt * 0.7, anchor="left", tracking=0.3)
        # shape label
        for (a, b, zh, en, cc) in [(24.8, 29.8, "正弦波", "SINE", GOLD), (36.4, 40.2, "方波", "SQUARE", GOLD),
                                   (40.4, 43.2, "锯齿波", "SAWTOOTH", (1.0, 0.85, 0.4)),
                                   (43.5, 48.0, "心跳", "HEARTBEAT", (1.0, 0.35, 0.4))]:
            al = window(t, a, b, 0.4, 0.3)
            if al > 0:
                draw_text(ctx, zh, self.X0 + 10, 250, size=40, font="serif", weight=700, color=cc, alpha=al,
                          anchor="left", tracking=0.2, glow=8, glow_alpha=0.4)
                draw_text(ctx, en, self.X0 + 12, 300, size=15, font="latin", weight=500, color=WHITE,
                          alpha=al * 0.6, anchor="left", tracking=0.4)

    def effects(self, t, lt):
        hb = 0.0
        for tb in S.heart_beats():
            if t >= tb:
                hb = max(hb, math.exp(-(t - tb) / 0.12))
        return {"flash": 0.03 * hb}


# ====================================================================== chapter 2
CLEF = "\U0001D11E"
NAME = "傅里叶"


class Drawing(Scene):
    start, end = 48.0, 79.4
    fade_out = 0.6
    bloom = 0.95
    PANELS = [8, 24, 80, 400]
    NAME_N = 800

    def prepare(self):
        self.clef = glyph_epicycles(CLEF, "NotoMusic.ttf", 300.0, 2048)
        self.name = glyph_epicycles(NAME, "MaShanZheng.ttf", 300.0, 4096, 0.3)
        for m in self.PANELS:
            self.clef.curve(m)
        self.name.curve(self.NAME_N)
        self.name_contours = glyph_contours(NAME, "MaShanZheng.ttf", 300.0)
        zc = self.name.z
        self.name_center = complex((zc.real.min() + zc.real.max()) / 2, (zc.imag.min() + zc.imag.max()) / 2)
        # name drawing progress keyframes
        kt = [60.6, 64.0, 69.5, 75.0]
        ks = [0.0, 0.08, 0.24, 1.0]
        self.s_of_t = PchipInterpolator(kt, ks)
        zt = [63.7, 64.35, 69.2, 72.2]
        zz = [1.0, 3.4, 3.4, 1.0]
        self.z_of_t = PchipInterpolator(zt, zz)
        # glyph offset: center of drawing in screen space at zoom 1
        self.SCALE = 1.55

    # ---------------------------------------------------------------- panels
    def draw_panels(self, ctx, t):
        s_prog = clamp((t - 49.6) / 8.0)
        a_all = window(t, 49.0, 60.2, 0.8, 0.8)
        if a_all <= 0:
            return
        E = self.clef
        xs = [300, 740, 1180, 1620]
        circ_a = 1 - smooth((t - 57.8) / 1.0)
        for p, (m, px) in enumerate(zip(self.PANELS, xs)):
            py = 470.0
            sc = 0.82
            ap = a_all * smooth((t - 49.0 - 0.15 * p) / 0.6)
            # panel frame
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
                    pts = np.stack([Z.real[a:b], Z.imag[a:b]], 1)
                    stroke_poly(ctx, pts, col, 2.2, 0.95 * ap, glow=0.8)
            # chain
            if circ_a > 0.01:
                ch = E.chain(s_prog % 1.0, m)
                chx = px + sc * ch.real
                chy = py + sc * ch.imag
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
            # label
            draw_text(ctx, f"{m}", px - 8, py + 340, size=40, font="latin", weight=300, color=WHITE, alpha=ap,
                      anchor="right")
            draw_text(ctx, "个圆", px + 4, py + 342, size=22, font="sans", weight=500, color=GOLD, alpha=ap,
                      anchor="left")

    # ---------------------------------------------------------------- the name
    def camera(self, t):
        z = float(self.z_of_t(clamp(t, 63.7, 72.2)))
        f = (z - 1.0) / 2.4
        # smoothed tip position
        ss = [float(self.s_of_t(clamp(t - d, 60.6, 75.0))) for d in (0.0, 0.15, 0.3, 0.45)]
        tips = [self.name.chain(s % 1.0, self.NAME_N)[-1] for s in ss]
        tip = sum(tips) / len(tips)
        cam = f * tip + (1 - f) * self.name_center
        return z, cam

    def to_screen(self, zc, z, cam):
        p = (zc - cam) * self.SCALE * z
        return 960 + p.real, 470 + p.imag

    def draw_name(self, ctx, t):
        a_all = smooth((t - 60.2) / 0.8)
        if a_all <= 0:
            return
        E = self.name
        m = self.NAME_N
        s = float(self.s_of_t(clamp(t, 60.6, 75.0)))
        z, cam = self.camera(t)
        curve = E.curve(m)
        nseg = int(clamp(s) * E.n)
        X, Y = self.to_screen(curve, z, cam)
        done = smooth((t - 75.0) / 1.2)
        # ink fill after completion
        if done > 0:
            ctx.save()
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            for c in self.name_contours:
                zc = c[:, 0] + 1j * c[:, 1]
                cx, cy = self.to_screen(zc - self._glyph_offset(), z, cam)
                polyline(ctx, np.stack([cx, cy], 1), close=True)
            g = cairo.LinearGradient(300, 300, 1600, 650)
            g.add_color_stop_rgba(0, 1.0, 0.85, 0.5, 0.85 * done)
            g.add_color_stop_rgba(0.5, 1.0, 0.95, 0.8, 0.95 * done)
            g.add_color_stop_rgba(1, 1.0, 0.7, 0.4, 0.85 * done)
            ctx.set_source(g)
            ctx.fill()
            ctx.restore()
        runs = pen_runs(E.pen[:nseg + 1])
        for a, b in runs:
            b = min(b, nseg + 1)
            if b - a > 1:
                pts = np.stack([X[a:b], Y[a:b]], 1)
                stroke_poly(ctx, pts, GOLD, 2.6, 0.95 * a_all, glow=1.0)
        # chain
        circ_a = a_all * (1 - smooth((t - 75.0) / 0.8)) * smooth((t - 60.4) / 1.2)
        if circ_a > 0.01:
            grow = ease_out((t - 60.4) / 1.6)
            ch = E.chain(s % 1.0, m)
            cx, cy = self.to_screen(ch, z, cam)
            r = E.radii(m) * self.SCALE * z
            k = int(m * grow)
            for i in range(k):
                if r[i] < 0.35:
                    continue
                u = i / m
                col = hsv(0.52 + 0.4 * u ** 0.6, 0.75, 1.0)
                circle(ctx, cx[i], cy[i], r[i], col, circ_a * (0.42 - 0.3 * u ** 0.5), width=1.1 if i > 10 else 1.6)
            polyline(ctx, np.stack([cx[:k + 1], cy[:k + 1]], 1))
            set_rgba(ctx, WHITE, 0.55 * circ_a)
            ctx.set_line_width(1.0)
            ctx.stroke()
            glow_dot(ctx, cx[k], cy[k], 4, WHITE, circ_a, 5)
        # counter
        a_c = window(t, 61.0, 75.5, 0.6, 0.6)
        if a_c > 0:
            draw_text(ctx, "N = 800", 1830, 1010 - 60, size=30, font="latin", weight=300, color=WHITE, alpha=0.8 * a_c,
                      anchor="right", tracking=0.1)
            draw_text(ctx, f"{s * 100:5.1f}%", 1830, 1010 - 20, size=18, font="mono", weight=300, color=GOLD,
                      alpha=0.7 * a_c, anchor="right")

    def _glyph_offset(self):
        if not hasattr(self, "_goff"):
            cs = self.name_contours
            allp = np.vstack(cs)
            self._goff = complex((allp[:, 0].min() + allp[:, 0].max()) / 2, (allp[:, 1].min() + allp[:, 1].max()) / 2)
        return self._goff

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.7)
        self.draw_panels(ctx, t)
        self.draw_name(ctx, t)

    def effects(self, t, lt):
        return {"flash": 0.25 * math.exp(-max(0.0, t - 64.0) / 0.25) if t >= 64.0 else 0.0,
                "chroma": 2.0 * math.exp(-max(0.0, t - 64.0) / 0.4) if t >= 64.0 else 0.0}
