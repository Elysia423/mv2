"""Core graphics engine: canvas, text, easing, colour and post-processing."""
import math
import os
from functools import lru_cache

import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
BPM = 120
RS = 1.0  # render scale (1 = 1080p, 2 = 4K); text, maths and glow are rasterised at this density
BEAT = 60.0 / BPM
BAR = 4 * BEAT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
FONTS = {
    "serif": ("NotoSerifSC.ttf", True),
    "sans": ("NotoSansSC.ttf", True),
    "brush": ("MaShanZheng.ttf", False),
    "latin": ("Montserrat.ttf", True),
    "mono": ("JetBrainsMono.ttf", True),
    "emoji": ("NotoEmoji.ttf", True),
    "garamond": ("Cormorant.ttf", True),
    "music": ("NotoMusic.ttf", False),
}

# ----------------------------------------------------------------- palette
CYAN = (0.25, 0.85, 1.0)
BLUE = (0.30, 0.50, 1.0)
MAGENTA = (1.0, 0.28, 0.62)
GOLD = (1.0, 0.76, 0.32)
VIOLET = (0.62, 0.45, 1.0)
WHITE = (1.0, 1.0, 1.0)
TEAL = (0.25, 1.0, 0.75)
ORANGE = (1.0, 0.50, 0.20)
RED = (1.0, 0.25, 0.25)


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p, q, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t_, p), (q, v, p), (p, v, t_), (p, q, v), (t_, p, v), (v, p, q)][i % 6]


def mix(c1, c2, a):
    return tuple(x + (y - x) * a for x, y in zip(c1, c2))


def spectrum_color(u):
    """u in [0,1] -> red..violet like a rainbow (u=0 red)."""
    return hsv(0.0 + 0.78 * clamp(u), 0.85, 1.0)


# ----------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def remap(t, a, b):
    """Map t from [a,b] to [0,1], clamped."""
    if b == a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3):
    t = clamp(t)
    return t ** p


def ease_io(t, p=3):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2 if p == 3 else smoother(t)


def ease_out_expo(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def window(t, a, b, fin=0.4, fout=0.4):
    """1 inside [a,b] with smooth fades of length fin/fout."""
    if t < a or t > b:
        return 0.0
    v = 1.0
    if fin > 0:
        v = min(v, smooth((t - a) / fin))
    if fout > 0:
        v = min(v, smooth((b - t) / fout))
    return v


def pulse(t, t0, decay=0.25):
    if t < t0:
        return 0.0
    return math.exp(-(t - t0) / decay)


# ----------------------------------------------------------------- canvas
def set_render_scale(scale):
    global RS
    RS = float(scale)


class Canvas:
    def __init__(self, scale=1.0):
        set_render_scale(scale)
        self.scale = scale
        self.w = int(round(W * scale))
        self.h = int(round(H * scale))
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w, self.h)
        self.ctx = cairo.Context(self.surf)
        self.ctx.scale(scale, scale)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    def array(self):
        self.surf.flush()
        return np.ndarray((self.h, self.w, 4), np.uint8, self.surf.get_data(), strides=(self.surf.get_stride(), 4, 1))

    def clear(self, rgb=(0, 0, 0), a=1.0):
        c = self.ctx
        c.save()
        c.identity_matrix()
        c.set_operator(cairo.OPERATOR_SOURCE)
        c.set_source_rgba(*rgb, a)
        c.paint()
        c.restore()


def surface_from_array(arr):
    """arr: HxWx4 uint8 premultiplied BGRA or HxWx3 float RGB in [0,1] -> cairo surface (keeps buffer ref)."""
    if arr.dtype != np.uint8:
        a = np.clip(arr, 0, 1)
        rgb = (a * 255 + 0.5).astype(np.uint8)
        bgra = np.empty(rgb.shape[:2] + (4,), np.uint8)
        bgra[..., 0] = rgb[..., 2]
        bgra[..., 1] = rgb[..., 1]
        bgra[..., 2] = rgb[..., 0]
        bgra[..., 3] = 255
        arr = bgra
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    if stride != w * 4:
        buf = np.zeros((h, stride), np.uint8)
        buf[:, : w * 4] = arr.reshape(h, w * 4)
        arr = buf
    surf = cairo.ImageSurface.create_for_data(memoryview(arr), cairo.FORMAT_ARGB32, w, h, stride)
    return surf


def paint_image(ctx, surf, x, y, w=None, h=None, alpha=1.0, anchor="center", filt=cairo.FILTER_GOOD):
    sw, sh = surf.get_width(), surf.get_height()
    w = w if w is not None else sw
    h = h if h is not None else sh
    if anchor == "center":
        x, y = x - w / 2, y - h / 2
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(w / sw, h / sh)
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(filt)
    ctx.rectangle(0, 0, sw, sh)
    ctx.clip()
    ctx.paint_with_alpha(alpha)
    ctx.restore()


def colormap_surface(values, cmap_fn):
    """values HxW in [0,1] -> cairo surface using cmap_fn(values)->HxWx3."""
    return surface_from_array(cmap_fn(np.clip(values, 0, 1)))


# ----------------------------------------------------------------- colormaps
def _make_lut(stops):
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=np.float32)
    u = np.linspace(0, 1, 1024)
    lut = np.stack([np.interp(u, xs, cs[:, i]) for i in range(3)], -1).astype(np.float32)
    return lut


LUT_FIRE = _make_lut([(0, (0, 0, 0)), (0.25, (0.35, 0.02, 0.25)), (0.5, (0.85, 0.15, 0.2)),
                      (0.72, (1.0, 0.55, 0.1)), (0.9, (1.0, 0.9, 0.45)), (1.0, (1, 1, 1))])
LUT_ICE = _make_lut([(0, (0, 0, 0)), (0.3, (0.02, 0.12, 0.35)), (0.6, (0.1, 0.5, 0.9)),
                     (0.85, (0.5, 0.9, 1.0)), (1.0, (1, 1, 1))])
LUT_GOLD = _make_lut([(0, (0, 0, 0)), (0.4, (0.35, 0.18, 0.03)), (0.75, (0.95, 0.65, 0.25)), (1.0, (1, 0.97, 0.85))])
LUT_DIVERGE = _make_lut([(0, (0.1, 0.6, 1.0)), (0.35, (0.02, 0.12, 0.3)), (0.5, (0.01, 0.01, 0.03)),
                         (0.65, (0.35, 0.05, 0.2)), (1.0, (1.0, 0.35, 0.6))])
LUT_GRAY = _make_lut([(0, (0, 0, 0)), (1, (1, 1, 1))])


def apply_lut(v, lut):
    idx = np.clip((v * 1023).astype(np.int32), 0, 1023)
    return lut[idx]


# ----------------------------------------------------------------- text
@lru_cache(maxsize=256)
def get_font(name, size, weight=400):
    fname, variable = FONTS[name]
    f = ImageFont.truetype(os.path.join(FONT_DIR, fname), int(size))
    if variable:
        axes = f.get_variation_axes()
        lo, hi = axes[0]["minimum"], axes[0]["maximum"]
        f.set_variation_by_axes([clamp(weight, lo, hi)])
    return f


class TextLayer:
    def __init__(self, surf, w, h, ox, mid, tw, glow=None):
        # ox: x of the text start inside the layer; mid: y of the visual middle line
        self.surf, self.w, self.h, self.ox, self.mid, self.tw, self.glow = surf, w, h, ox, mid, tw, glow


def _mask_to_surface(mask, color):
    m = np.asarray(mask, dtype=np.float32) / 255.0
    h, w = m.shape
    arr = np.empty((h, w, 4), np.uint8)
    arr[..., 0] = (m * color[2] * 255).astype(np.uint8)
    arr[..., 1] = (m * color[1] * 255).astype(np.uint8)
    arr[..., 2] = (m * color[0] * 255).astype(np.uint8)
    arr[..., 3] = (m * 255).astype(np.uint8)
    return surface_from_array(arr)


_REF = {"latin": "H", "mono": "H", "garamond": "H", "music": "H"}


@lru_cache(maxsize=4096)
def text_layer(s, font="sans", size=40, weight=400, color=WHITE, tracking=0.0, glow=0):
    """Render a single line to a cairo surface.

    tracking: extra spacing in em units. glow: blur radius (px) for a separate soft glow surface.
    """
    f = get_font(font, size, weight)
    pad = int(size * 0.6) + glow * 3
    adv = [f.getlength(ch) for ch in s]
    total = sum(adv) + tracking * size * max(0, len(s) - 1)
    asc, desc = f.getmetrics()
    w = int(math.ceil(total)) + 2 * pad
    h = asc + desc + 2 * pad
    img = Image.new("L", (max(w, 1), max(h, 1)), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for ch, a in zip(s, adv):
        d.text((x, pad), ch, font=f, fill=255)
        x += a + tracking * size
    bb = f.getbbox(_REF.get(font, "中"))
    mid = pad + (bb[1] + bb[3]) / 2
    surf = _mask_to_surface(img, color)
    gsurf = _mask_to_surface(img.filter(ImageFilter.GaussianBlur(glow)), color) if glow else None
    return TextLayer(surf, w, h, pad, mid, total, gsurf)


def draw_text(ctx, s, x, y, size=40, font="sans", weight=400, color=WHITE, alpha=1.0, anchor="center",
              tracking=0.0, glow=0, glow_alpha=0.8, scale=1.0, reveal=None, reveal_soft=0.15):
    """Draw text. anchor: 'center' (center of ink box, vertical middle of em),
    'left', 'right' (vertical middle). y is the vertical middle of the cap height box."""
    if alpha <= 0.002 or not s:
        return
    L = text_layer(s, font, int(round(size * RS)), weight, tuple(color), tracking, int(round(glow * RS)))
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale / RS, scale / RS)
    mid = L.mid
    if anchor == "center":
        ox = L.ox + L.tw / 2
    elif anchor == "left":
        ox = L.ox
    else:
        ox = L.ox + L.tw
    ctx.translate(-ox, -mid)
    if reveal is not None and reveal < 1.0:
        # soft left->right wipe
        x0 = L.ox + (L.tw + L.tw * reveal_soft) * reveal - L.tw * reveal_soft
        grad = cairo.LinearGradient(x0, 0, x0 + L.tw * reveal_soft + 1, 0)
        grad.add_color_stop_rgba(0, 0, 0, 0, alpha)
        grad.add_color_stop_rgba(1, 0, 0, 0, 0)
        if L.glow is not None:
            ctx.set_source_surface(L.glow, 0, 0)
            ctx.mask(grad)
        ctx.set_source_surface(L.surf, 0, 0)
        ctx.mask(grad)
    else:
        if L.glow is not None:
            ctx.set_source_surface(L.glow, 0, 0)
            ctx.paint_with_alpha(alpha * glow_alpha)
        ctx.set_source_surface(L.surf, 0, 0)
        ctx.paint_with_alpha(alpha)
    ctx.restore()


def text_width(s, size=40, font="sans", weight=400, tracking=0.0):
    L = text_layer(s, font, int(round(size * RS)), weight, WHITE, tracking, 0)
    return L.tw / RS


def draw_text_chars(ctx, s, x, y, t, size=40, font="sans", weight=400, color=WHITE, alpha=1.0,
                    anchor="center", tracking=0.0, stagger=0.05, dur=0.5, rise=18, glow=0, glow_alpha=0.8,
                    t_out=None, out_dur=0.4):
    """Per-character animated reveal. t: time since the reveal started."""
    f = get_font(font, int(round(size * RS)), weight)
    advs = [f.getlength(ch) / RS for ch in s]
    total = sum(advs) + tracking * size * (len(s) - 1)
    if anchor == "center":
        cx = x - total / 2
    elif anchor == "left":
        cx = x
    else:
        cx = x - total
    for i, (ch, a) in enumerate(zip(s, advs)):
        ti = t - i * stagger
        p = ease_out(ti / dur, 3) if dur > 0 else 1.0
        if ti < 0:
            p = 0.0
        al = alpha * clamp(ti / dur if dur > 0 else 1.0)
        if t_out is not None:
            to = t - t_out - (i * stagger * 0.5)
            al *= 1 - clamp(to / out_dur)
        if al > 0.002 and not ch.isspace():
            draw_text(ctx, ch, cx + a / 2, y + rise * (1 - p), size, font, weight, color, al, "center",
                      0.0, glow, glow_alpha)
        cx += a + tracking * size


# ----------------------------------------------------------------- math text via matplotlib
@lru_cache(maxsize=64)
def math_layer(tex, size=40, color=WHITE, glow=0):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    import io
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    buf = io.BytesIO()
    prop = FontProperties(size=size)
    mathtext.math_to_image(tex, buf, prop=prop, dpi=72 * 1.0, format="png", color="black")
    buf.seek(0)
    from PIL import ImageOps
    a = ImageOps.invert(Image.open(buf).convert("L"))
    pad = 20 + glow * 3
    big = Image.new("L", (a.width + 2 * pad, a.height + 2 * pad), 0)
    big.paste(a, (pad, pad))
    surf = _mask_to_surface(big, color)
    g = _mask_to_surface(big.filter(ImageFilter.GaussianBlur(glow)), color) if glow else None
    return surf, big.width, big.height, g


def draw_math(ctx, tex, x, y, size=40, color=WHITE, alpha=1.0, glow=0, glow_alpha=0.9, scale=1.0, reveal=None):
    surf, w, h, g = math_layer(tex, int(round(size * RS)), tuple(color), int(round(glow * RS)))
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale / RS, scale / RS)
    ctx.translate(-w / 2, -h / 2)
    if reveal is not None and reveal < 1:
        soft = 0.15 * w
        x0 = (w + soft) * reveal - soft
        grad = cairo.LinearGradient(x0, 0, x0 + soft, 0)
        grad.add_color_stop_rgba(0, 0, 0, 0, alpha)
        grad.add_color_stop_rgba(1, 0, 0, 0, 0)
        if g is not None:
            ctx.set_source_surface(g, 0, 0)
            ctx.mask(grad)
        ctx.set_source_surface(surf, 0, 0)
        ctx.mask(grad)
    else:
        if g is not None:
            ctx.set_source_surface(g, 0, 0)
            ctx.paint_with_alpha(alpha * glow_alpha)
        ctx.set_source_surface(surf, 0, 0)
        ctx.paint_with_alpha(alpha)
    ctx.restore()


# ----------------------------------------------------------------- drawing helpers
def set_rgba(ctx, c, a=1.0):
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def polyline(ctx, pts, close=False):
    if len(pts) == 0:
        return
    ctx.move_to(float(pts[0][0]), float(pts[0][1]))
    for p in pts[1:]:
        ctx.line_to(float(p[0]), float(p[1]))
    if close:
        ctx.close_path()


def stroke_poly(ctx, pts, color, width=2.0, alpha=1.0, close=False, glow=0.0):
    """Stroke a polyline; optional cheap glow via wide low-alpha under-strokes."""
    if len(pts) < 2:
        return
    polyline(ctx, pts, close)
    path = ctx.copy_path()
    if glow > 0:
        for k, (wm, am) in enumerate([(6.0, 0.06), (3.0, 0.12)]):
            ctx.new_path()
            ctx.append_path(path)
            set_rgba(ctx, color, alpha * am * glow)
            ctx.set_line_width(width * wm)
            ctx.stroke()
    ctx.new_path()
    ctx.append_path(path)
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.stroke()


def stroke_fading(ctx, pts, color, width=2.0, alpha=1.0, segs=24, head_bright=1.0, tail_alpha=0.0):
    """Polyline whose alpha fades from tail (start) to head (end)."""
    n = len(pts)
    if n < 2:
        return
    step = max(1, n // segs)
    for i in range(0, n - 1, step):
        j = min(n, i + step + 1)
        u = (i + j) / 2 / n
        a = alpha * (tail_alpha + (1 - tail_alpha) * u ** 1.5) * head_bright
        polyline(ctx, pts[i:j])
        set_rgba(ctx, color, a)
        ctx.set_line_width(width)
        ctx.stroke()


def circle(ctx, x, y, r, color, alpha=1.0, fill=False, width=1.5):
    ctx.new_sub_path()
    ctx.arc(float(x), float(y), max(float(r), 0.01), 0, 2 * math.pi)
    set_rgba(ctx, color, alpha)
    if fill:
        ctx.fill()
    else:
        ctx.set_line_width(width)
        ctx.stroke()


def glow_dot(ctx, x, y, r, color, alpha=1.0, halo=4.0):
    g = cairo.RadialGradient(x, y, 0, x, y, r * halo)
    g.add_color_stop_rgba(0, color[0], color[1], color[2], alpha)
    g.add_color_stop_rgba(0.25, color[0], color[1], color[2], alpha * 0.35)
    g.add_color_stop_rgba(1, color[0], color[1], color[2], 0)
    ctx.set_source(g)
    ctx.arc(x, y, r * halo, 0, 2 * math.pi)
    ctx.fill()
    circle(ctx, x, y, r, mix(color, WHITE, 0.6), alpha, fill=True)


def arrow(ctx, x0, y0, x1, y1, color, alpha=1.0, width=2.0, head=10):
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.stroke()
    ang = math.atan2(y1 - y0, x1 - x0)
    L = math.hypot(x1 - x0, y1 - y0)
    hd = min(head, L * 0.4)
    ctx.move_to(x1, y1)
    ctx.line_to(x1 - hd * math.cos(ang - 0.4), y1 - hd * math.sin(ang - 0.4))
    ctx.line_to(x1 - hd * math.cos(ang + 0.4), y1 - hd * math.sin(ang + 0.4))
    ctx.close_path()
    ctx.fill()


def rounded_rect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


# ----------------------------------------------------------------- background
_rng = np.random.default_rng(7)
DUST = {
    "x": _rng.uniform(0, W, 420),
    "y": _rng.uniform(0, H, 420),
    "z": _rng.uniform(0.2, 1.0, 420) ** 2,
    "ph": _rng.uniform(0, 2 * np.pi, 420),
    "vx": _rng.normal(0, 1, 420),
    "vy": _rng.normal(0, 1, 420),
}


def background(ctx, t, strength=1.0, hue=(0.03, 0.05, 0.10), dust=1.0, drift=(8.0, -3.0)):
    """Deep-space gradient with drifting dust."""
    g = cairo.RadialGradient(W * 0.5, H * 0.45, 50, W * 0.5, H * 0.5, W * 0.75)
    g.add_color_stop_rgb(0, hue[0] * strength, hue[1] * strength, hue[2] * strength)
    g.add_color_stop_rgb(1, 0.004, 0.005, 0.012)
    ctx.set_source(g)
    ctx.paint()
    if dust <= 0:
        return
    D = DUST
    z = D["z"]
    xs = (D["x"] + (drift[0] + D["vx"] * 3) * t * (0.3 + z)) % W
    ys = (D["y"] + (drift[1] + D["vy"] * 3) * t * (0.3 + z)) % H
    tw = 0.55 + 0.45 * np.sin(D["ph"] + t * (0.7 + z * 2))
    for x, y, zz, a in zip(xs, ys, z, tw):
        r = 0.6 + 1.6 * zz
        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.set_source_rgba(0.65, 0.8, 1.0, dust * a * (0.08 + 0.35 * zz))
        ctx.fill()


def grid(ctx, spacing=60, color=(0.3, 0.5, 1.0), alpha=0.06, ox=0, oy=0):
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(1)
    x = ox % spacing
    while x < W:
        ctx.move_to(x, 0)
        ctx.line_to(x, H)
        x += spacing
    y = oy % spacing
    while y < H:
        ctx.move_to(0, y)
        ctx.line_to(W, y)
        y += spacing
    ctx.stroke()


# ----------------------------------------------------------------- post processing
_VIGNETTE = {}
_GRAIN = None


def _vignette(h, w):
    key = (h, w)
    if key not in _VIGNETTE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - w / 2) / (w / 2)
        ny = (yy - h / 2) / (h / 2)
        r = np.sqrt(nx * nx * 0.9 + ny * ny * 1.1)
        v = 1.0 - 0.42 * np.clip(r - 0.35, 0, None) ** 1.6
        _VIGNETTE[key] = np.clip(v, 0.35, 1)[..., None].astype(np.float32)
    return _VIGNETTE[key]


def _grain(h, w, frame):
    global _GRAIN
    if _GRAIN is None or _GRAIN.shape[1:3] != (h, w):
        r = np.random.default_rng(3)
        _GRAIN = r.normal(0, 1, (6, h, w, 1)).astype(np.float32)
    return _GRAIN[frame % 6]


def bloom(img, strength=0.9, threshold=0.12):
    """img: float32 HxWx3 (BGR). Multi-scale bloom."""
    h, w = img.shape[:2]
    src = img if threshold <= 0 else np.clip(img - threshold, 0, None) * (1.0 / (1.0 - threshold))
    acc = np.zeros_like(img)
    cur = src
    weights = [0.4, 0.34, 0.28, 0.24]
    sig = [2.0 * RS, 3.0 * RS, 4.0 * RS, 6.0 * RS]
    for k in range(4):
        ch, cw = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        cur = cv2.resize(cur, (cw, ch), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(cur, (0, 0), sig[k])
        acc += weights[k] * cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return img + strength * acc


def postprocess(bgra, frame_idx, bloom_amt=0.9, overlay=None, fade=1.0, flash=0.0, grain=0.004,
                chroma=0.0, exposure=1.0):
    """bgra: uint8 HxWx4 from cairo (opaque). overlay: uint8 premultiplied BGRA to composite after bloom.
    Returns uint8 HxWx4 BGRA."""
    img = bgra[..., :3].astype(np.float32) * (1.0 / 255.0)
    if bloom_amt > 0:
        img = bloom(img, bloom_amt)
    if exposure != 1.0:
        img *= exposure
    if chroma > 0:
        h, w = img.shape[:2]
        s = 1 + chroma * 0.004
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        img[..., 2] = cv2.warpAffine(img[..., 2], M, (w, h), borderMode=cv2.BORDER_REFLECT)
        s = 1 - chroma * 0.004
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        img[..., 0] = cv2.warpAffine(img[..., 0], M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if overlay is not None:
        ov = overlay.astype(np.float32) * (1.0 / 255.0)
        a = ov[..., 3:4]
        img = img * (1 - a) + ov[..., :3]
    # filmic-ish shoulder
    img = img / (1.0 + 0.18 * np.clip(img - 0.8, 0, None))
    h, w = img.shape[:2]
    img *= _vignette(h, w)
    if flash > 0:
        img = img + flash
    if grain > 0:
        img += grain * _grain(h, w, frame_idx)
    if fade < 1.0:
        img *= fade
    out = np.empty((h, w, 4), np.uint8)
    np.clip(img * 255.0 + 0.5, 0, 255, out=img)
    out[..., :3] = img.astype(np.uint8)
    out[..., 3] = 255
    return out
