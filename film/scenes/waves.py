"""Chapter 1: circles and waves -- Euler's helix, the audible square wave, time <-> frequency in 3D."""
import math

import numpy as np

from .. import story as S
from ..core import (CYAN, GOLD, H, MAGENTA, RED, VIOLET, W, WHITE, background, circle, clamp, draw_math, draw_text,
                    ease_out, glow_dot, hsv, lerp, mix, polyline, set_rgba, smooth, smoother, stroke_poly, window)
from ..data import beat_pulse
from ..three import Camera, axis3, line3, line3_depth
from .base import Scene


class Euler3D(Scene):
    """e^{iwt} as a helix along the time axis; its shadows are cos and sin."""
    start, end = S.T_EULER, S.T_SQUARE
    fade_in = 0.3
    fade_out = 0.3
    bloom = 1.0
    L = 1.6        # world units per turn along x
    TURNS = 4.0

    def cam(self, t):
        u = smoother((t - (self.start + 1.4)) / 2.2)
        yaw = lerp(-0.62, -1.5, u)
        pitch = lerp(0.36, 0.05, u)
        dist = lerp(11.0, 9.5, u)
        return Camera(yaw, pitch, dist, target=(self.L * self.TURNS / 2 * (1 - 0.35 * u), 0, 0), fov=38)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        cam = self.cam(t)
        head = clamp(1.2 + lt * 0.95, 0, self.TURNS)
        tau = np.linspace(0, head, max(2, int(head * 160)))
        x = tau * self.L
        c, s = np.cos(2 * np.pi * tau), np.sin(2 * np.pi * tau)
        R = 1.0
        side = smooth((lt - 0.6) / 0.6) * (1 - smooth((lt - 3.0) / 0.6))
        # axis
        axis3(ctx, cam, (-0.3, 0, 0), (self.L * self.TURNS + 0.6, 0, 0), WHITE, 0.35, 1.2, head=10)
        # shadows: real part on the floor, imaginary part on the back wall
        if side > 0:
            fl, wl = -1.7, -1.7
            line3(ctx, cam, np.stack([x, np.full_like(x, fl), c * R], 1), CYAN, 2.0, 0.8 * side)
            line3(ctx, cam, np.stack([x, s * R, np.full_like(x, wl)], 1), MAGENTA, 2.0, 0.8 * side)
            for k in range(0, len(x), 12):
                line3(ctx, cam, np.array([[x[k], s[k], c[k]], [x[k], fl, c[k]]]), CYAN, 1.0, 0.12 * side)
                line3(ctx, cam, np.array([[x[k], s[k], c[k]], [x[k], s[k], wl]]), MAGENTA, 1.0, 0.12 * side)
            lx, ly, _ = cam.project(np.array([[x[-1] + 0.3, fl, c[-1]], [x[-1] + 0.3, s[-1], wl]]))
            draw_text(ctx, "cos ωt  实部", lx[0] + 10, ly[0], size=22, font="sans", weight=500, color=CYAN,
                      alpha=side, anchor="left")
            draw_text(ctx, "sin ωt  虚部", lx[1] + 10, ly[1], size=22, font="sans", weight=500, color=MAGENTA,
                      alpha=side, anchor="left")
        # the helix itself
        line3_depth(ctx, cam, np.stack([x, s * R, c * R], 1), GOLD, 3.0, 1.0, far_alpha=0.35)
        hx, hy, _ = cam.project(np.array([[x[-1], 0, 0], [x[-1], s[-1] * R, c[-1] * R]]))
        ctx.move_to(hx[0], hy[0])
        ctx.line_to(hx[1], hy[1])
        set_rgba(ctx, WHITE, 0.9)
        ctx.set_line_width(2)
        ctx.stroke()
        glow_dot(ctx, hx[1], hy[1], 6, WHITE, 1.0, 5)
        # ring at the head
        ang = np.linspace(0, 2 * np.pi, 120)
        line3(ctx, cam, np.stack([np.full_like(ang, x[-1]), np.sin(ang), np.cos(ang)], 1), CYAN, 1.2, 0.35)
        draw_math(ctx, r"$e^{i\omega t}=\cos\omega t+i\,\sin\omega t$", W / 2, 150, size=46, color=WHITE,
                  alpha=smooth((lt - 0.3) / 0.5), glow=8)
        draw_text(ctx, "欧拉公式 · EULER", W / 2, 215, size=18, font="sans", weight=500, color=GOLD,
                  alpha=0.8 * smooth((lt - 0.6) / 0.5), tracking=0.2)



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
    """1D epicycle chain -> waveform. Visible 22-30 (square) and 36-42 (saw, heartbeat)."""
    start, end = S.T_SQUARE, S.T_PANELS
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.9
    CX, CY = 560.0, 500.0
    X0 = 1150.0
    V = 180.0

    def prepare(self):
        self.ton = _turn_on_times()

    def weight(self, t):
        return super().weight(t) * (1 - window(t, S.T_TF3D - 0.15, S.C1_SAW + 0.15, 0.3, 0.3))

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
        ts = np.linspace(max(S.T_SQUARE - 1.0, t - span), t, 700)
        ys = self.tip_y(ts)
        xs = self.X0 + (t - ts) * self.V
        As = np.array([self.scale(x) for x in ts])
        cys = np.array([self.CY + 60 * S.c1_shape_weights(x)[2] for x in ts])
        ghost_a = window(t, 25.0, S.C1_ECG + 0.4, 0.8, 0.5)
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
        self.gibbs(ctx, t)
        # counter
        n = S.c1_count(t)
        a_cnt = window(t, S.T_SQUARE + 0.2, S.C1_ECG - 0.1, 0.4, 0.4)
        if a_cnt > 0:
            chg = max([math.exp(-(t - tk) / 0.25) for tk, _ in S.C1_COUNT_KEYS if t >= tk] + [0])
            draw_text(ctx, str(n), 1650, 170, size=110 + 14 * chg, font="latin", weight=200, color=WHITE,
                      alpha=a_cnt, anchor="right", glow=6, glow_alpha=0.5 * chg)
            draw_text(ctx, "个圆", 1668, 150, size=26, font="sans", weight=500, color=GOLD, alpha=a_cnt, anchor="left")
            draw_text(ctx, "CIRCLES", 1668, 186, size=14, font="latin", weight=500, color=(0.7, 0.8, 1.0),
                      alpha=a_cnt * 0.7, anchor="left", tracking=0.3)
        # shape label
        for (a, b, zh, en, cc) in [(22.2, 23.6, "正弦波", "SINE", GOLD), (27.0, 30.0, "方波", "SQUARE", GOLD),
                                   (36.2, 38.1, "锯齿波", "SAWTOOTH", (1.0, 0.85, 0.4)),
                                   (38.3, 42.0, "心电图", "ECG", (1.0, 0.35, 0.4))]:
            al = window(t, a, b, 0.4, 0.3)
            if al > 0:
                draw_text(ctx, zh, self.X0 + 10, 250, size=40, font="serif", weight=700, color=cc, alpha=al,
                          anchor="left", tracking=0.2, glow=8, glow_alpha=0.4)
                draw_text(ctx, en, self.X0 + 12, 300, size=15, font="latin", weight=500, color=WHITE,
                          alpha=al * 0.6, anchor="left", tracking=0.4)

    def gibbs(self, ctx, t):
        """Magnified view of the overshoot next to a jump."""
        a = window(t, 28.0, 30.2, 0.35, 0.3)
        if a <= 0:
            return
        x0, y0, w, h = 110.0, 150.0, 460.0, 300.0
        set_rgba(ctx, (0.02, 0.03, 0.07), 0.85 * a)
        ctx.rectangle(x0, y0, w, h)
        ctx.fill()
        set_rgba(ctx, GOLD, 0.6 * a)
        ctx.set_line_width(1.2)
        ctx.rectangle(x0, y0, w, h)
        ctx.stroke()
        m = S.c1_count(t)
        th = np.linspace(-0.12, 0.55, 500)
        k = 2 * np.arange(1, m + 1) - 1
        y = (np.sin(np.outer(th, k)) @ (4 / (np.pi * k)))
        lo, hi = -1.2, 1.3
        X = x0 + 20 + (th + 0.12) / 0.67 * (w - 40)
        Y = y0 + h - 25 - (y - lo) / (hi - lo) * (h - 60)
        yi = np.where(th >= 0, 1.0, -1.0)
        ctx.set_dash([4, 6])
        stroke_poly(ctx, np.stack([X, y0 + h - 25 - (yi - lo) / (hi - lo) * (h - 60)], 1), WHITE, 1.2, 0.45 * a)
        ctx.set_dash([])
        stroke_poly(ctx, np.stack([X, Y], 1), GOLD, 2.4, a, glow=0.8)
        i = int(np.argmax(y))
        y1 = y0 + h - 25 - (1.0 - lo) / (hi - lo) * (h - 60)
        set_rgba(ctx, (1.0, 0.4, 0.4), 0.95 * a)
        ctx.set_line_width(2)
        ctx.move_to(X[i] + 16, Y[i])
        ctx.line_to(X[i] + 16, y1)
        ctx.stroke()
        draw_text(ctx, "≈ 9% 过冲", X[i] + 28, (Y[i] + y1) / 2 + 30, size=24, font="sans", weight=600, color=(1.0, 0.5, 0.5),
                  alpha=a, anchor="left")
        draw_text(ctx, "吉布斯现象 · GIBBS", x0 + 14, y0 + 22, size=16, font="sans", weight=500, color=WHITE,
                  alpha=0.75 * a, anchor="left", tracking=0.08)

    def effects(self, t, lt):
        hb = 0.0
        for tb in S.heart_beats():
            if t >= tb:
                hb = max(hb, math.exp(-(t - tb) / 0.12))
        return {"flash": 0.03 * hb}




class TimeFreq3D(Scene):
    """The square wave's odd harmonics laid out along a frequency axis; rotate 90 degrees to see the spectrum."""
    start, end = S.T_TF3D, S.C1_SAW
    fade_in = 0.3
    fade_out = 0.3
    bloom = 1.0
    KS = [1, 3, 5, 7, 9, 11, 13, 15]
    ZSUM = -1.6
    DZ = 1.0

    def cam(self, lt):
        u1 = smoother((lt - 0.6) / 1.8)
        u2 = smoother((lt - 3.0) / 1.6)
        yaw = lerp(0.0, -0.75, u1)
        yaw = lerp(yaw, -math.pi / 2, u2)
        pitch = lerp(0.0, 0.32, u1) * (1 - u2) + 0.06 * u2
        dist = lerp(13.5, 15.0, u1)
        tz = lerp(self.ZSUM, 3.2, max(u1, u2))
        return Camera(yaw, pitch, dist, target=(0, 0.2, tz), fov=36)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        cam = self.cam(lt)
        spread = smoother((lt - 0.4) / 1.6)
        spec = smooth((lt - 3.6) / 1.0)
        th0 = S.c1_theta(t)
        x = np.linspace(-4.2, 4.2, 360)
        th = x * np.pi / 2 + th0
        total = np.zeros_like(x)
        for i, k in enumerate(self.KS):
            amp = 4 / (np.pi * k)
            y = amp * np.sin(k * th)
            total += y
            z = lerp(self.ZSUM, i * self.DZ, spread)
            col = hsv(0.55 + 0.4 * i / len(self.KS), 0.75, 1.0)
            a_curve = 0.9 * (1 - 0.8 * spec)
            line3(ctx, cam, np.stack([x, y, np.full_like(x, z)], 1), col, 2.0, a_curve)
            # spectrum bar at the near end of the time axis
            if spec > 0:
                xb = -4.6
                line3(ctx, cam, np.array([[xb, 0, z], [xb, amp * spec, z]]), col, 9.0, 0.95 * spec)
                px, py, _ = cam.project(np.array([[xb, amp * spec + 0.25, z], [xb, -0.35, z]]))
                draw_text(ctx, f"{amp:.2f}", px[0], py[0], size=20, font="mono", weight=400, color=WHITE,
                          alpha=0.85 * spec)
                draw_text(ctx, f"{k}f", px[1], py[1], size=24, font="latin", weight=600, color=col, alpha=spec)
        # the sum (the square wave) in front
        line3(ctx, cam, np.stack([x, total, np.full_like(x, self.ZSUM)], 1), GOLD, 3.2, 1.0 * (1 - 0.7 * spec))
        # axes
        axis3(ctx, cam, (-4.4, 0, self.ZSUM), (4.6, 0, self.ZSUM), WHITE, 0.35, 1.2, head=10)
        axis3(ctx, cam, (-4.6, 0, self.ZSUM), (-4.6, 0, len(self.KS) * self.DZ + 0.4), WHITE, 0.35 * spread, 1.2,
              head=10)
        tx, ty, _ = cam.project(np.array([[4.6, -0.5, self.ZSUM], [-4.6, -0.9, len(self.KS) * self.DZ]]))
        draw_text(ctx, "时间 t", tx[0], ty[0], size=22, font="sans", weight=500, color=WHITE, alpha=0.8 * (1 - spec))
        draw_text(ctx, "频率 f", tx[1], ty[1], size=22, font="sans", weight=500, color=WHITE, alpha=0.8 * spread)
        a1 = window(lt, 0.0, 3.6, 0.3, 0.4)
        a2 = smooth((lt - 3.9) / 0.5)
        draw_text(ctx, "时域  TIME DOMAIN", W / 2, 130, size=30, font="sans", weight=600, color=GOLD, alpha=a1,
                  tracking=0.15)
        draw_text(ctx, "频域  FREQUENCY DOMAIN", W / 2, 130, size=30, font="sans", weight=600, color=CYAN, alpha=a2,
                  tracking=0.15)
