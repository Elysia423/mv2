"""A tiny perspective camera and 3D drawing helpers on top of cairo."""
import math

import numpy as np

from .core import H, W, mix, polyline, set_rgba


class Camera:
    """Orbit camera: looks at `target` from angles yaw (around +y) and pitch (elevation), distance `dist`.

    World axes: x right, y up, z towards the viewer at yaw = 0.
    """

    def __init__(self, yaw=0.0, pitch=0.0, dist=10.0, target=(0.0, 0.0, 0.0), fov=40.0, center=(W / 2, H / 2),
                 roll=0.0):
        self.target = np.asarray(target, float)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.pos = self.target + dist * np.array([cp * sy, sp, cp * cy])
        f = self.target - self.pos
        f /= np.linalg.norm(f)
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll:
            cr, sr = math.cos(roll), math.sin(roll)
            r, u = r * cr + u * sr, -r * sr + u * cr
        self.r, self.u, self.f = r, u, f
        self.focal = (H / 2) / math.tan(math.radians(fov) / 2)
        self.cx, self.cy = center

    def project(self, P):
        """P (...,3) -> (x, y, depth)."""
        P = np.asarray(P, float)
        d = P - self.pos
        xc = d @ self.r
        yc = d @ self.u
        zc = np.maximum(d @ self.f, 1e-3)
        return self.cx + self.focal * xc / zc, self.cy - self.focal * yc / zc, zc

    def scale_at(self, depth):
        return self.focal / depth


def line3(ctx, cam, P, color, width=2.0, alpha=1.0, close=False):
    x, y, _ = cam.project(P)
    polyline(ctx, np.stack([x, y], 1), close)
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.stroke()


def line3_depth(ctx, cam, P, color, width=2.0, alpha=1.0, near=None, far=None, segs=40, far_alpha=0.25):
    """Polyline whose alpha fades with depth (split into chunks)."""
    x, y, z = cam.project(P)
    n = len(x)
    if n < 2:
        return
    near = z.min() if near is None else near
    far = z.max() if far is None else far
    step = max(1, n // segs)
    for i in range(0, n - 1, step):
        j = min(n, i + step + 1)
        dz = (z[i:j].mean() - near) / max(far - near, 1e-6)
        a = alpha * (1 - (1 - far_alpha) * np.clip(dz, 0, 1))
        polyline(ctx, np.stack([x[i:j], y[i:j]], 1))
        set_rgba(ctx, color, a)
        ctx.set_line_width(width)
        ctx.stroke()


def wire_surface(ctx, cam, X, Y, Z, color_fn, width=1.0, alpha=1.0, every=1):
    """Wireframe of a height field sampled on a grid (X,Y,Z all (nu,nv)); lines along both directions.

    color_fn(value_array) -> list of colours per vertex row (uses Z for colour).
    """
    sx, sy, sz = cam.project(np.stack([X, Y, Z], -1))
    nu, nv = X.shape
    for i in range(0, nu, every):
        _stroke_colored(ctx, sx[i], sy[i], Y[i], color_fn, width, alpha)
    for j in range(0, nv, every):
        _stroke_colored(ctx, sx[:, j], sy[:, j], Y[:, j], color_fn, width, alpha)


def _stroke_colored(ctx, xs, ys, vals, color_fn, width, alpha, chunk=6):
    n = len(xs)
    for i in range(0, n - 1, chunk):
        j = min(n, i + chunk + 1)
        c = color_fn(float(np.mean(vals[i:j])))
        polyline(ctx, np.stack([xs[i:j], ys[i:j]], 1))
        set_rgba(ctx, c[:3], alpha * (c[3] if len(c) > 3 else 1.0))
        ctx.set_line_width(width)
        ctx.stroke()


def solid_surface(ctx, cam, X, Y, Z, color_fn, edge_alpha=0.5, fill_alpha=0.9, width=0.8, shade=True):
    """Painter's-algorithm quads of a surface grid; colour from the mean height (Y)."""
    P = np.stack([X, Y, Z], -1)
    sx, sy, sz = cam.project(P)
    nu, nv = X.shape
    quads = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            d = (sz[i, j] + sz[i + 1, j] + sz[i, j + 1] + sz[i + 1, j + 1]) * 0.25
            quads.append((d, i, j))
    quads.sort(key=lambda q: -q[0])
    for d, i, j in quads:
        v = (Y[i, j] + Y[i + 1, j] + Y[i, j + 1] + Y[i + 1, j + 1]) * 0.25
        c = color_fn(v)
        ctx.move_to(sx[i, j], sy[i, j])
        ctx.line_to(sx[i + 1, j], sy[i + 1, j])
        ctx.line_to(sx[i + 1, j + 1], sy[i + 1, j + 1])
        ctx.line_to(sx[i, j + 1], sy[i, j + 1])
        ctx.close_path()
        k = 0.25 if shade else 1.0
        ctx.set_source_rgba(c[0] * k, c[1] * k, c[2] * k, fill_alpha)
        ctx.fill_preserve()
        set_rgba(ctx, c[:3], edge_alpha)
        ctx.set_line_width(width)
        ctx.stroke()


def axis3(ctx, cam, p0, p1, color, alpha=0.6, width=1.5, head=0.0):
    x, y, _ = cam.project(np.array([p0, p1]))
    ctx.move_to(x[0], y[0])
    ctx.line_to(x[1], y[1])
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.stroke()
    if head > 0:
        ang = math.atan2(y[1] - y[0], x[1] - x[0])
        ctx.move_to(x[1], y[1])
        ctx.line_to(x[1] - head * math.cos(ang - 0.4), y[1] - head * math.sin(ang - 0.4))
        ctx.line_to(x[1] - head * math.cos(ang + 0.4), y[1] - head * math.sin(ang + 0.4))
        ctx.close_path()
        ctx.fill()
    return x[1], y[1]


def height_color(lo_col, hi_col, lo=0.0, hi=1.0):
    def f(v):
        u = min(1.0, max(0.0, (v - lo) / (hi - lo)))
        return mix(lo_col, hi_col, u)
    return f
