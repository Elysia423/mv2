"""Chapter 4: images are waves too -- 2D ripples in 3D, reconstruction, the spectrum in 3D, filters, JPEG."""
import math

import cairo
import numpy as np

from .. import story as S
from ..three import Camera, solid_surface, wire_surface
from ..core import (CYAN, GOLD, H, LUT_ICE, MAGENTA, VIOLET, W, WHITE, smoother, apply_lut, background, clamp, draw_text, ease_out,
                    lerp, mix, paint_image, text_width, rounded_rect, set_rgba, smooth, surface_from_array, window)
from ..data import image
from .base import Scene

IH, IW = 300, 450  # working resolution (chelsea is 300x451)


def _to_bgra(rgb):
    return np.clip(rgb, 0, 1)


class ImageBase(Scene):
    """Shared data for the image shots (computed once per process)."""
    fade_in = 0.3
    fade_out = 0.3
    bloom = 0.22
    _shared = None

    def prepare(self):
        if ImageBase._shared is not None:
            self.__dict__.update(ImageBase._shared)
            return
        self._prepare()
        ImageBase._shared = dict(self.__dict__)

    def _prepare(self):
        img = image("chelsea", (IH, IW), gray=False)[..., :3]
        self.img = img
        self.F = np.fft.fft2(img, axes=(0, 1))
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        Fl = np.fft.fft2(lum)
        self.Fl = Fl
        mag = np.abs(Fl)
        self.order = np.argsort(-mag.ravel())
        self.ntot = mag.size
        ky = np.fft.fftfreq(IH)[:, None]
        kx = np.fft.fftfreq(IW)[None, :]
        self.kr = np.sqrt(ky ** 2 + kx ** 2)  # radial frequency (cycles/px)
        spec = np.log1p(np.fft.fftshift(mag))
        lo, hi = np.percentile(spec, 35), np.percentile(spec, 99.95)
        self.spec_img = np.clip((spec - lo) / (hi - lo), 0, 1) ** 1.3
        # 8x8 DCT
        from scipy.fft import dctn, idctn
        self.dctn, self.idctn = dctn, idctn
        by, bx = IH // 8, IW // 8
        crop = img[: by * 8, : bx * 8]
        blocks = crop.reshape(by, 8, bx, 8, 3).transpose(0, 2, 1, 3, 4)
        self.dct_blocks = dctn(blocks, axes=(2, 3), norm="ortho")
        self.by, self.bx = by, bx
        zz = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 else p[0]))
        self.zigzag = zz
        self._cache = {}

    # ------------------------------------------------------------------ helpers
    def K_of_t(self, t):
        t0 = S.T_IMGBUILD
        if t < t0 + 2.0:
            return int(1 + (t - t0) / 2.0 * 7)  # 1..8, slowly
        one = int(self.ntot * 0.01)
        if t < t0 + 4.2:
            u = (t - t0 - 2.0) / 2.2
            return int(8 * (one / 8) ** u)
        if t < t0 + 6.0:
            return one
        u = clamp((t - t0 - 6.0) / 1.6)
        return int(one * (self.ntot / one) ** (u ** 1.2))

    def recon(self, K):
        K = max(1, min(self.ntot, K))
        key = ("r", K)
        if key in self._cache:
            return self._cache[key]
        m = np.zeros(self.ntot, bool)
        m[self.order[:K]] = True
        m = m.reshape(IH, IW)
        r = np.real(np.fft.ifft2(self.F * m[..., None], axes=(0, 1)))
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = r
        return r

    def grating(self, idx, t, size=240):
        iy, ix = divmod(int(idx), IW)
        fy = np.fft.fftfreq(IH)[iy] * IH
        fx = np.fft.fftfreq(IW)[ix] * IW
        yy, xx = np.mgrid[0:size, 0:size] / size
        ph = np.angle(self.Fl.ravel()[idx])
        g = 0.5 + 0.5 * np.cos(2 * np.pi * (fx * xx * 1.5 + fy * yy) + ph + t * 2)
        return g

    def draw_rgb(self, ctx, rgb, x, y, w, h, alpha=1.0, nearest=False):
        surf = surface_from_array(_to_bgra(rgb))
        paint_image(ctx, surf, x, y, w, h, alpha, filt=cairo.FILTER_NEAREST if nearest else cairo.FILTER_BEST)

    def frame(self, ctx, x, y, w, h, a, col=(0.5, 0.7, 1.0)):
        set_rgba(ctx, col, 0.35 * a)
        ctx.set_line_width(1.2)
        ctx.rectangle(x - w / 2 - 6, y - h / 2 - 6, w + 12, h + 12)
        ctx.stroke()

    # ------------------------------------------------------------------ parts
    def part_build(self, ctx, t, a):
        K = self.K_of_t(t)
        r = self.recon(K)
        # accumulated image
        x, y, w, h = 1160.0, 470.0, 810.0, 540.0
        rr = np.clip(r, 0, 1)
        self.draw_rgb(ctx, rr, x, y, w, h, a)
        self.frame(ctx, x, y, w, h, a)
        # current wave tile
        idx = self.order[min(K - 1, self.ntot - 1)]
        g = self.grating(idx, t)
        tile = apply_lut(g * 0.9, LUT_ICE)
        tx, ty, ts = 380.0, 440.0, 300.0
        self.draw_rgb(ctx, tile, tx, ty, ts, ts, a)
        self.frame(ctx, tx, ty, ts, ts, a, CYAN)
        draw_text(ctx, "+", 640, 440, size=90, font="latin", weight=200, color=WHITE, alpha=0.8 * a)
        draw_text(ctx, "当前叠加的波", tx, ty + 190, size=22, font="sans", weight=500, color=CYAN, alpha=0.9 * a)
        draw_text(ctx, "THE WAVE BEING ADDED", tx, ty + 222, size=12, font="latin", weight=500, color=WHITE,
                  alpha=0.5 * a, tracking=0.3)
        # counters
        pct = K / self.ntot * 100
        draw_text(ctx, f"{K:,}", x - w / 2, y - h / 2 - 60, size=52, font="latin", weight=300, color=WHITE, alpha=a,
                  anchor="left")
        draw_text(ctx, "个波", x - w / 2 + 14 + text_width(f"{K:,}", 52, "latin", 300), y - h / 2 - 56, size=22, font="sans",
                  weight=500, color=GOLD, alpha=a, anchor="left")
        draw_text(ctx, f"{pct:.3f} %" if pct < 1 else f"{pct:.1f} %", x + w / 2, y - h / 2 - 58, size=30, font="mono", weight=300,
                  color=GOLD if S.T_IMGBUILD + 4.1 < t < S.T_IMGBUILD + 6.1 else WHITE, alpha=a, anchor="right")

    def part_filter(self, ctx, t, a):
        t0 = S.T_FILTER
        lp = t < t0 + 2.9
        if lp:
            u = ease_out((t - t0 - 0.2) / 2.4, 2)
            rad = lerp(0.5, 0.012, u)
        else:
            u = ease_out((t - t0 - 3.0) / 2.4, 2)
            rad = lerp(0.0, 0.06, u)
        soft = max(rad * 0.25, 0.003)
        if lp:
            m = 1 / (1 + np.exp((self.kr - rad) / soft))
        else:
            m = 1 / (1 + np.exp(-(self.kr - rad) / soft))
        r = np.real(np.fft.ifft2(self.F * m[..., None], axes=(0, 1)))
        if not lp:
            r = np.clip(np.abs(r.mean(-1, keepdims=True)) * 4.0, 0, 1) * np.array([0.7, 0.9, 1.0])
        x, y, w, h = 560.0, 470.0, 720.0, 480.0
        self.draw_rgb(ctx, np.clip(r, 0, 1), x, y, w, h, a)
        self.frame(ctx, x, y, w, h, a)
        # spectrum with the mask
        sx, sy, sw, sh = 1380.0, 470.0, 480.0, 480.0
        sm = np.fft.fftshift(m)
        sp = self.spec_img
        vis = apply_lut(sp * (0.25 + 0.75 * sm), LUT_ICE)
        self.draw_rgb(ctx, vis, sx, sy, sw, sh, a)
        self.frame(ctx, sx, sy, sw, sh, a, CYAN)
        # circle marker (aspect: kr uses x normalised to IW)
        ctx.save()
        ctx.translate(sx, sy)
        ctx.scale(sw, sh)
        ctx.arc(0, 0, max(rad, 0.002), 0, 2 * math.pi)
        ctx.restore()
        set_rgba(ctx, GOLD, 0.9 * a)
        ctx.set_line_width(2)
        ctx.stroke()
        draw_text(ctx, "图像", x, y - h / 2 - 40, size=24, font="sans", weight=500, color=WHITE, alpha=0.9 * a)
        draw_text(ctx, "频谱", sx, sy - sh / 2 - 40, size=24, font="sans", weight=500, color=CYAN, alpha=0.9 * a)
        lab = "低通：只留低频" if lp else "高通：只留高频"
        draw_text(ctx, lab, sx, sy + sh / 2 + 44, size=22, font="sans", weight=500, color=GOLD, alpha=0.9 * a)
        draw_text(ctx, "中心 = 低频，边缘 = 高频", sx, sy + sh / 2 + 80, size=16, font="sans", weight=400,
                  color=WHITE, alpha=0.55 * a)

    def part_jpeg(self, ctx, t, a):
        # 8x8 DCT basis grid
        gx, gy, cell = 470.0, 470.0, 58.0
        t0 = S.T_JPEG
        n_show = int(clamp((t - t0 - 0.2) / 1.8) * 64)
        for q, (i, j) in enumerate(self.zigzag):
            if q >= n_show:
                break
            pop = ease_out(clamp((t - t0 - 0.2 - q * 1.8 / 64) / 0.3), 3)
            yy, xx = np.mgrid[0:16, 0:16] + 0.5
            b = np.cos(np.pi * i * yy / 16) * np.cos(np.pi * j * xx / 16)
            tile = apply_lut((b * 0.5 + 0.5), LUT_ICE)
            cx = gx + (j - 3.5) * (cell + 4)
            cy = gy + (i - 3.5) * (cell + 4)
            s = cell * (0.6 + 0.4 * pop)
            self.draw_rgb(ctx, tile, cx, cy, s, s, a * pop, nearest=False)
        draw_text(ctx, "64 种波纹", gx, gy + 4.4 * (cell + 4) + 20, size=24, font="sans", weight=500, color=CYAN,
                  alpha=a * smooth((t - t0 - 0.8) / 0.5))
        # blockwise reconstruction with the first n zigzag coefficients
        ncoef = 1 + int(round(ease_out((t - t0 - 2.2) / 2.4, 2) * 9))
        mask = np.zeros((8, 8))
        for (i, j) in self.zigzag[:ncoef]:
            mask[i, j] = 1
        rec = self.idctn(self.dct_blocks * mask[None, None, :, :, None], axes=(2, 3), norm="ortho")
        rec = rec.transpose(0, 2, 1, 3, 4).reshape(self.by * 8, self.bx * 8, 3)
        x, y, w, h = 1310.0, 470.0, 750.0, 750.0 * rec.shape[0] / rec.shape[1]
        self.draw_rgb(ctx, np.clip(rec, 0, 1), x, y, w, h, a, nearest=ncoef < 3)
        self.frame(ctx, x, y, w, h, a)
        draw_text(ctx, f"每个 8×8 方块：{ncoef} / 64 种波纹", x, y + h / 2 + 44, size=22, font="sans", weight=500,
                  color=GOLD, alpha=a * smooth((t - t0 - 2.0) / 0.4))
        draw_text(ctx, "JPEG", x - w / 2, y - h / 2 - 44, size=34, font="latin", weight=700, color=WHITE, alpha=a,
                  anchor="left", tracking=0.2)



class Waves3D(ImageBase):
    """A few 2D plane waves as 3D surfaces, then their sum."""
    start, end = S.T_WAVES2D, S.T_IMGBUILD
    bloom = 0.9
    WAVES = [((1.0, 0.0), 1.0), ((0.6, 1.4), 0.6), ((-1.7, 0.9), 0.4)]

    def surf(self, X, Z, i, t):
        (kx, kz), a = self.WAVES[i]
        return 0.55 * a * np.cos(kx * X + kz * Z + t * 1.2 + i)

    def color(self, v):
        u = clamp((v + 0.9) / 1.8)
        return mix((0.1, 0.3, 1.0), mix(CYAN, GOLD, u), u)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.5)
        n = 34
        g = np.linspace(-2.6, 2.6, n)
        X, Z = np.meshgrid(g, g)
        merge = smoother((lt - 1.7) / 1.3)
        cam = Camera(0.55 - 0.25 * lt / 4, 0.6, 16.5, target=(0, 0, 0), fov=40)
        offs = [-6.2, 0.0, 6.2]
        if merge < 1:
            for i in range(3):
                Y = self.surf(X, Z, i, t)
                cam_i = Camera(0.55 - 0.25 * lt / 4, 0.6, 16.5, target=(-offs[i] * (1 - merge), 0, 0), fov=40)
                col = [CYAN, VIOLET, MAGENTA][i]
                wire_surface(ctx, cam_i, X, Y, Z, lambda v, c=col: c, 1.0, 0.8 * (1 - merge), every=1)
        if merge > 0:
            Y = sum(self.surf(X, Z, i, t) for i in range(3))
            solid_surface(ctx, cam, X, Y * merge, Z, self.color, edge_alpha=0.8 * merge, fill_alpha=0.85 * merge,
                          width=1.0)
        a = 1 - smooth((lt - 1.5) / 0.4)
        for i, x in enumerate([420, 960, 1500]):
            draw_text(ctx, ["低频 · 横向", "中频 · 斜向", "高频 · 斜向"][i], x, 820, size=22, font="sans", weight=500,
                      color=[CYAN, VIOLET, MAGENTA][i], alpha=a)
        draw_text(ctx, "叠加 = 一张“图像”", W / 2, 150, size=30, font="sans", weight=600, color=GOLD,
                  alpha=smooth((lt - 2.4) / 0.5), tracking=0.1)


class ImageBuild(ImageBase):
    start, end = S.T_IMGBUILD, S.T_SPEC3D

    def draw(self, cv, t, lt):
        background(cv.ctx, t, dust=0.5)
        self.part_build(cv.ctx, t, 1.0)


class Spectrum3D(ImageBase):
    """The photo's log-magnitude spectrum as a 3D landscape."""
    start, end = S.T_SPEC3D, S.T_FILTER
    bloom = 0.9

    def prepare(self):
        super().prepare()
        sp = self.spec_img
        h, w = sp.shape
        c = sp[h // 2 - 120:h // 2 + 120, w // 2 - 120:w // 2 + 120]
        # max-pool so the thin bright cross survives the downsampling
        self.grid = c.reshape(60, 4, 60, 4).max(axis=(1, 3))

    def color(self, v):
        u = clamp(v / 3.0)
        if u < 0.4:
            return mix((0.05, 0.15, 0.6), CYAN, u / 0.4)
        return mix(CYAN, WHITE, (u - 0.4) / 0.6)

    def draw(self, cv, t, lt):
        ctx = cv.ctx
        background(ctx, t, dust=0.4)
        g = np.linspace(-4.0, 4.0, 60)
        X, Z = np.meshgrid(g, g)
        grow = ease_out(lt / 1.2, 3)
        Y = 3.0 * self.grid ** 2.2 * grow
        cam = Camera(0.4 + 0.35 * lt / 6, 0.55 - 0.02 * lt, 13.0, target=(0, 0.6, 0), fov=40)
        solid_surface(ctx, cam, X, Y, Z, self.color, edge_alpha=0.7, fill_alpha=0.9, width=0.9)
        # the photo in the corner
        self.draw_rgb(ctx, self.img, 1650, 230, 300, 200, smooth(lt / 0.5))
        self.frame(ctx, 1650, 230, 300, 200, 1.0)
        a = smooth((lt - 0.8) / 0.5)
        px, py, _ = cam.project(np.array([[0, 3.3, 0], [3.6, 0.3, 0]]))
        draw_text(ctx, "低频（中心）", px[0], py[0] - 20, size=22, font="sans", weight=500, color=GOLD, alpha=a)
        draw_text(ctx, "高频", px[1] + 10, py[1], size=20, font="sans", weight=500, color=CYAN, alpha=a, anchor="left")


class Filter(ImageBase):
    start, end = S.T_FILTER, S.T_JPEG

    def draw(self, cv, t, lt):
        background(cv.ctx, t, dust=0.5)
        self.part_filter(cv.ctx, t, 1.0)


class JPEG(ImageBase):
    start, end = S.T_JPEG, S.T_BEYOND - 0.2

    def draw(self, cv, t, lt):
        background(cv.ctx, t, dust=0.5)
        self.part_jpeg(cv.ctx, t, 1.0)


def image_scenes():
    return [Waves3D(), ImageBuild(), Spectrum3D(), Filter(), JPEG()]
