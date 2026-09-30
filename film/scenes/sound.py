"""Chapter 3: sound -- a chord unmixed, the winding machine, the cochlea, song fingerprints, 3D spectrogram,
noise cancelling and MP3 masking."""
import math

import cairo
import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, RED, VIOLET, W, WHITE, background, circle, clamp, draw_math, draw_text,
                    ease_out, glow_dot, hsv, lerp, mix, polyline, rounded_rect, set_rgba, smooth, smoother,
                    stroke_poly, window)
from ..data import beat_pulse, spec_at, spectrum
from ..three import Camera, solid_surface
from .base import Scene

NOTE_COLS = [CYAN, GOLD, MAGENTA]


class Chord(Scene):
    start, end = S.T_CHORD, S.T_WIND
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.95
    X0, X1 = 300.0, 1620.0

    def env(self, t, i):
        return clamp((t - self.start - 0.3 - 0.4 * i) / 0.8) ** 2

    def comp(self, t, i, xs):
        f = S.SOUND_CHORD[i][1]
        tau = (xs - self.X0) / (self.X1 - self.X0) * 0.030 + (t - self.start) * 0.004
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
                    glow = smooth((t - self.start - 4.0 - 0.2 * li) / 0.35)
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
        draw_text(ctx, "混合的声音", self.X0 - 30, y_top - 8, size=32, font="sans", weight=500, color=WHITE,
                  alpha=0.8 * smooth(lt / 1.0), anchor="right")
        draw_text(ctx, "MIXTURE", self.X0 - 30, y_top + 22, size=22, font="latin", weight=500, color=WHITE,
                  alpha=0.5 * smooth(lt / 1.0), anchor="right", tracking=0.18)
        # split into components
        lanes = [430.0, 540.0, 650.0]
        if t > self.start + 2.4:
            for i in range(3):
                spi = ease_out((t - self.start - 2.6 - 0.12 * i) / 1.0, 3)
                y = lerp(y_top, lanes[i], spi)
                a = smooth((t - self.start - 2.4) / 0.4)
                stroke_poly(ctx, np.stack([xs, y - 42 * comps[i]], 1), NOTE_COLS[i], 2.2, 0.9 * a, glow=0.8)
                nm, f = S.SOUND_CHORD[i]
                draw_text(ctx, f"{nm}", self.X0 - 30, y - 8, size=34, font="latin", weight=500, color=NOTE_COLS[i],
                          alpha=a * spi, anchor="right")
                draw_text(ctx, f"{f:.0f} Hz", self.X0 - 30, y + 22, size=22, font="mono", weight=400, color=WHITE,
                          alpha=0.6 * a * spi, anchor="right")
        # keyboard + beams
        ka = smooth((t - self.start - 3.4) / 0.5)
        if ka > 0:
            centers, ky = self.keyboard(ctx, t, ka)
            for i, (nm, f) in enumerate(S.SOUND_CHORD):
                b = smooth((t - self.start - 4.0 - 0.2 * i) / 0.35)
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
    start, end = S.T_EAR, S.T_SHAZAM
    fade_in = 0.3
    fade_out = 0.3
    bloom = 1.0
    CX, CY = 800.0, 470.0

    def prepare(self):
        n = 700
        u = np.linspace(0, 1, n)          # 0 = apex (low), 1 = base (high)
        turns = 2.6
        phi = u * turns * 2 * np.pi
        r = 48 + 372 * (u ** 0.85)
        self.u = u
        self.px = self.CX + r * np.cos(-phi + 0.6)
        self.py = self.CY + r * np.sin(-phi + 0.6) * 0.92
        # outward normals
        dx, dy = np.gradient(self.px), np.gradient(self.py)
        L = np.hypot(dx, dy) + 1e-9
        self.nx, self.ny = dy / L, -dx / L
        self.width = 12 + 32 * u

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.6, hue=(0.03, 0.04, 0.10))
        reveal = ease_out(lt / 1.1, 2)
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
        a = smooth((lt - 0.6) / 0.6)
        draw_text(ctx, "低音", self.CX + 5, self.CY + 5, size=30, font="sans", weight=500, color=(1, 0.5, 0.4),
                  alpha=0.9 * a)
        draw_text(ctx, "LOW", self.CX + 5, self.CY + 32, size=22, font="latin", weight=500, color=WHITE, alpha=0.5 * a,
                  tracking=0.18)
        ex, ey = self.px[-1], self.py[-1]
        draw_text(ctx, "高音", ex - 36, ey - 10, size=30, font="sans", weight=500, color=(0.7, 0.5, 1.0),
                  alpha=0.9 * a, anchor="right")
        draw_text(ctx, "HIGH", ex - 36, ey + 18, size=22, font="latin", weight=500, color=WHITE, alpha=0.5 * a,
                  anchor="right", tracking=0.18)
        # side panel: live spectrum bars
        xs0, xs1, yb = 1330.0, 1840.0, 760.0
        if a > 0:
            nbar = 64
            bw = (xs1 - xs0) / nbar
            for j in range(nbar):
                v = float(spec[int(j / nbar * nb)]) ** 1.1
                col = hsv(0.78 * j / nbar, 0.8, 1.0)
                set_rgba(ctx, col, 0.85 * a)
                ctx.rectangle(xs0 + j * bw, yb - 280 * v, bw * 0.7, 280 * v)
                ctx.fill()
            draw_text(ctx, "频谱 · 实时", xs0, yb + 30, size=28, font="sans", weight=500, color=WHITE, alpha=0.7 * a,
                      anchor="left")
        # small 'ear' label
        draw_text(ctx, "耳蜗", 120, 200, size=64, font="serif", weight=700, color=WHITE, alpha=0.9 * a, anchor="left",
                  tracking=0.2, glow=8, glow_alpha=0.3)
        draw_text(ctx, "COCHLEA", 124, 260, size=24, font="latin", weight=500, color=GOLD, alpha=0.7 * a,
                  anchor="left", tracking=0.18)


class Winding(Scene):
    """Wrap the chord around a circle at a sweeping frequency; the centre of mass reveals the notes."""
    start, end = S.T_WIND, S.T_EAR
    fade_in = 0.3
    fade_out = 0.3
    bloom = 1.0
    T = 0.06
    F0, F1 = 150.0, 400.0

    def prepare(self):
        self.ts = np.linspace(0, self.T, 2400)
        g = sum(np.sin(2 * np.pi * f * self.ts) for _, f in S.SOUND_CHORD) / 3
        self.g = g
        fs = np.linspace(self.F0, self.F1, 700)
        E = np.exp(-2j * np.pi * np.outer(fs, self.ts))
        self.fs = fs
        self.com = (E @ (g + 1.2)) / len(self.ts)

    def fw(self, lt):
        return lerp(self.F0, self.F1, clamp((lt - 0.5) / 6.8))

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        f = self.fw(lt)
        cx, cy, R = 600.0, 520.0, 150.0
        z = (self.g + 1.2) * np.exp(-2j * np.pi * f * self.ts)
        # axes
        set_rgba(ctx, WHITE, 0.15)
        ctx.set_line_width(1)
        ctx.move_to(cx - 330, cy)
        ctx.line_to(cx + 330, cy)
        ctx.move_to(cx, cy - 330)
        ctx.line_to(cx, cy + 330)
        ctx.stroke()
        pts = np.stack([cx + R * z.real, cy - R * z.imag], 1)
        stroke_poly(ctx, pts, CYAN, 1.2, 0.55)
        com = np.interp(f, self.fs, self.com.real) + 1j * np.interp(f, self.fs, self.com.imag)
        mx, my = cx + R * com.real * 4, cy - R * com.imag * 4
        ctx.move_to(cx, cy)
        ctx.line_to(mx, my)
        set_rgba(ctx, RED, 0.9)
        ctx.set_line_width(2.5)
        ctx.stroke()
        glow_dot(ctx, mx, my, 8, RED, 1.0, 5)
        draw_text(ctx, f"缠绕频率  {f:5.1f} Hz", cx, cy + 370, size=34, font="sans", weight=500, color=WHITE)
        draw_text(ctx, "重心 ×4", mx + 18, my - 18, size=28, font="sans", weight=500, color=RED, anchor="left")
        # |centre of mass| vs frequency
        gx0, gx1, gy0, gy1 = 1080.0, 1800.0, 700.0, 330.0
        set_rgba(ctx, WHITE, 0.3)
        ctx.move_to(gx0, gy0)
        ctx.line_to(gx1, gy0)
        ctx.stroke()
        k = np.searchsorted(self.fs, f)
        mag = np.abs(self.com)
        mmax = mag.max()
        X = gx0 + (self.fs[:k] - self.F0) / (self.F1 - self.F0) * (gx1 - gx0)
        Y = gy0 - mag[:k] / mmax * (gy0 - gy1)
        if k > 1:
            stroke_poly(ctx, np.stack([X, Y], 1), GOLD, 2.6, 1.0, glow=1.0)
        for i, (nm, fn) in enumerate(S.SOUND_CHORD):
            if f >= fn:
                xx = gx0 + (fn - self.F0) / (self.F1 - self.F0) * (gx1 - gx0)
                a = smooth((f - fn) / 12)
                draw_text(ctx, nm, xx, gy1 - 30, size=34, font="latin", weight=600, color=NOTE_COLS[i], alpha=a)
                draw_text(ctx, f"{fn:.0f} Hz", xx, gy0 + 30, size=24, font="mono", weight=400, color=WHITE,
                          alpha=0.6 * a)
        draw_text(ctx, "|重心|", gx0, gy1 - 60, size=28, font="sans", weight=500, color=GOLD, alpha=0.8, anchor="left")
        draw_math(ctx, r"$\hat{g}(f)=\int g(t)\,e^{-2\pi i f t}\,dt$", 1440, 190, size=44, color=WHITE,
                  alpha=smooth((lt - 0.3) / 0.5), glow=8)


class Shazam(Scene):
    """Scrolling spectrogram of the soundtrack; peaks become a constellation fingerprint."""
    start, end = S.T_SHAZAM, S.T_DROP
    fade_in = 0.3
    fade_out = 0.1
    bloom = 0.9

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.3)
        spec = spectrum()
        fps = 30
        ncol = 150  # 5 s of history
        i1 = int(t * fps)
        cols = np.arange(i1 - ncol + 1, i1 + 1)
        cols = np.clip(cols, 0, len(spec) - 1)
        img = spec[cols][:, 8:200].T[::-1]  # (freq, time), high freqs on top
        from ..core import LUT_ICE, apply_lut, paint_image, surface_from_array
        vis = apply_lut(np.clip(img * 1.25, 0, 1) ** 1.1 * 0.95, LUT_ICE)
        x0, y0, w, h = 140.0, 170.0, 1180.0, 640.0
        paint_image(ctx, surface_from_array(vis), x0, y0, w, h, 1.0, anchor="topleft")
        set_rgba(ctx, CYAN, 0.35)
        ctx.set_line_width(1)
        ctx.rectangle(x0, y0, w, h)
        ctx.stroke()
        # peaks: local maxima in 2D neighbourhoods
        from scipy.ndimage import maximum_filter
        mx = maximum_filter(img, size=(15, 13))
        pk = np.argwhere((img == mx) & (img > 0.55))
        a_pk = smooth((lt - 0.6) / 0.5)
        pts = []
        for fy, tx in pk:
            px = x0 + (tx + 0.5) / ncol * w
            py = y0 + (fy + 0.5) / img.shape[0] * h
            pts.append((px, py, fy, tx))
        pts.sort(key=lambda p: p[0])
        for i, (px, py, fy, tx) in enumerate(pts):
            for (qx, qy, _, _) in pts[i + 1:i + 4]:
                if 0 < qx - px < 220:
                    ctx.move_to(px, py)
                    ctx.line_to(qx, qy)
                    set_rgba(ctx, GOLD, 0.35 * a_pk)
                    ctx.set_line_width(1.2)
                    ctx.stroke()
        for px, py, _, _ in pts:
            glow_dot(ctx, px, py, 3.5, WHITE, a_pk, 4)
        draw_text(ctx, "频率 ↑", x0 - 16, y0 + 20, size=28, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="right")
        draw_text(ctx, "时间 →", x0 + w, y0 + h + 28, size=28, font="sans", weight=500, color=WHITE, alpha=0.7,
                  anchor="right")
        # hash list
        hx = 1400.0
        draw_text(ctx, "指纹 HASH", hx, 200, size=32, font="sans", weight=600, color=GOLD, alpha=a_pk, anchor="left")
        for j, (px, py, fy, tx) in enumerate(pts[-9:]):
            a = a_pk * smooth((lt - 0.8 - 0.1 * j) / 0.3)
            f_hz = 40 * (12000 / 40) ** ((8 + (img.shape[0] - fy)) / 256)
            draw_text(ctx, f"{f_hz:6.0f} Hz · Δt {((ncol - tx) / fps):4.2f}s", hx, 250 + j * 40, size=28, font="mono",
                      weight=400, color=WHITE, alpha=0.75 * a, anchor="left")
        draw_text(ctx, "Shazam · 2003", hx, 250 + 9 * 40 + 20, size=28, font="latin", weight=500, color=CYAN,
                  alpha=0.8 * a_pk, anchor="left")


class Drop3D(Scene):
    """3D terrain of the soundtrack's spectrogram, flown over live."""
    start, end = S.T_DROP, S.T_NOISE
    fade_in = 0.1
    fade_out = 0.3
    bloom = 1.0
    NB = 72
    NR = 44
    STEP = 0.1

    def color(self, v):
        v = clamp(v / 2.2)
        if v < 0.35:
            return mix((0.1, 0.25, 0.9), CYAN, v / 0.35)
        if v < 0.7:
            return mix(CYAN, MAGENTA, (v - 0.35) / 0.35)
        return mix(MAGENTA, GOLD, (v - 0.7) / 0.3)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5, strength=1.0 + 0.5 * beat_pulse(t))
        spec = spectrum()
        nb = spec.shape[1]
        bins = np.linspace(0, nb - 1, self.NB).astype(int)
        base = math.floor(t / self.STEP)
        frac = t / self.STEP - base
        rows = []
        for j in range(self.NR):
            tj = (base - j) * self.STEP
            rows.append(spec_at(max(0.0, tj))[bins] if tj >= S.T_DROP - 4.4 else np.zeros(self.NB))
        Hm = np.array(rows)  # (NR, NB) newest first
        xs = np.linspace(-6.0, 6.0, self.NB)
        zs = -(np.arange(self.NR) + frac) * 0.42 + 1.5
        X, Z = np.meshgrid(xs, zs)
        Y = 2.2 * Hm ** 1.3
        u = lt / (self.end - self.start)
        cam = Camera(0.25 - 0.5 * u, 0.5, 14.5 - 1.5 * u, target=(0, 0.2, -6.5), fov=46)
        solid_surface(ctx, cam, X, Y, Z, self.color, edge_alpha=0.75, fill_alpha=0.92, width=1.0)
        draw_text(ctx, "低频", 150, 820, size=30, font="sans", weight=600, color=WHITE, alpha=0.8, anchor="left")
        draw_text(ctx, "高频", 1770, 820, size=30, font="sans", weight=600, color=WHITE, alpha=0.8, anchor="right")
        draw_text(ctx, "LIVE", 1800, 70, size=26, font="latin", weight=700, color=(1, 0.3, 0.35),
                  alpha=0.8 * (0.6 + 0.4 * math.sin(t * 6)), anchor="right", tracking=0.18)

    def effects(self, t, lt):
        return {"flash": 0.3 * math.exp(-max(0.0, lt) / 0.2), "chroma": 1.5 * beat_pulse(t, 0.1)}


class NoiseMP3(Scene):
    start, end = S.T_NOISE, S.T_WAVES2D
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.9

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        # ---- left: noise cancelling
        x0, x1 = 120.0, 860.0
        xs = np.linspace(x0, x1, 500)
        u = (xs - x0) / (x1 - x0) * 6 + t * 1.5
        noise = (0.55 * np.sin(2.1 * u) + 0.3 * np.sin(5.3 * u + 1) + 0.18 * np.sin(11.7 * u + 2)
                 + 0.1 * np.sin(23.1 * u + 0.3))
        mix_a = smooth((lt - 0.8) / 0.8)
        rows = [(300.0, noise, (0.85, 0.85, 0.9), "噪声"), (500.0, -noise, CYAN, "反相波"),
                (720.0, noise * (1 - mix_a), GOLD, "相加")]
        for i, (y, sig, col, lab) in enumerate(rows):
            a = smooth((lt - 0.2 * i) / 0.4)
            stroke_poly(ctx, np.stack([xs, y - 70 * sig], 1), col, 2.4, a, glow=0.6)
            draw_text(ctx, lab, x0 - 10, y, size=28, font="sans", weight=500, color=col, alpha=a, anchor="right")
        draw_text(ctx, "+", (x0 + x1) / 2, 405, size=44, font="latin", weight=300, color=WHITE, alpha=0.7)
        draw_text(ctx, "=", (x0 + x1) / 2, 615, size=44, font="latin", weight=300, color=WHITE, alpha=0.7)
        draw_text(ctx, "降噪耳机", (x0 + x1) / 2, 170, size=34, font="serif", weight=700, color=WHITE, glow=6,
                  glow_alpha=0.3)
        # ---- right: MP3 masking
        spec = spec_at(t)
        gx0, gx1, gb, gh = 1060.0, 1800.0, 760.0, 420.0
        nbar = 64
        idx = np.linspace(0, len(spec) - 1, nbar).astype(int)
        v = spec[idx]
        # masking threshold: spread each band's level to its neighbours (roughly a bark-scale spreading function)
        d = np.abs(np.arange(nbar)[:, None] - np.arange(nbar)[None, :])
        thr = np.max(v[None, :] - 0.06 * d - 0.12, axis=1)
        thr = np.maximum(thr, 0.12)
        bw = (gx1 - gx0) / nbar
        cut = smooth((lt - 2.8) / 0.8)
        for i in range(nbar):
            hgt = gh * v[i]
            masked = v[i] < thr[i]
            if masked:
                col, al = (0.5, 0.5, 0.55), 0.8 * (1 - cut)
            else:
                col, al = hsv(0.55 + 0.35 * i / nbar, 0.7, 1.0), 0.9
            set_rgba(ctx, col, al)
            ctx.rectangle(gx0 + i * bw + 1, gb - hgt, bw - 2, hgt)
            ctx.fill()
        tx = gx0 + (np.arange(nbar) + 0.5) * bw
        ctx.set_dash([5, 5])
        stroke_poly(ctx, np.stack([tx, gb - gh * thr], 1), RED, 2.0, 0.9)
        ctx.set_dash([])
        draw_text(ctx, "掩蔽阈值", gx1, gb - gh - 20, size=28, font="sans", weight=500, color=RED, anchor="right")
        draw_text(ctx, "灰色 = 听不见，删掉", (gx0 + gx1) / 2, gb + 40, size=28, font="sans", weight=400,
                  color=WHITE, alpha=0.75)
        draw_text(ctx, "MP3 / AAC", (gx0 + gx1) / 2, 170, size=34, font="latin", weight=700, color=WHITE, glow=6,
                  glow_alpha=0.3, tracking=0.1)


def sound_scenes():
    return [Chord(), Winding(), Ear(), Shazam(), Drop3D(), NoiseMP3()]
