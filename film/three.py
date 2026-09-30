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


def wire_surface(ctx, cam, X, Y, Z, color_fn, width=1.0, alpha=1.0, every=1, fog=0.6):
    """Wireframe of a height field sampled on a grid (X,Y,Z all (nu,nv)); lines along both directions,
    dimmer and thinner with distance."""
    sx, sy, sz = cam.project(np.stack([X, Y, Z], -1))
    near, far = sz.min(), sz.max()
    fk = 1 - fog * np.clip((sz - near) / max(far - near, 1e-6), 0, 1)
    nu, nv = X.shape
    for i in range(0, nu, every):
        _stroke_colored(ctx, sx[i], sy[i], Y[i], color_fn, width, alpha, fade=fk[i])
    for j in range(0, nv, every):
        _stroke_colored(ctx, sx[:, j], sy[:, j], Y[:, j], color_fn, width, alpha, fade=fk[:, j])


def _stroke_colored(ctx, xs, ys, vals, color_fn, width, alpha, chunk=6, fade=None):
    n = len(xs)
    for i in range(0, n - 1, chunk):
        j = min(n, i + chunk + 1)
        c = color_fn(float(np.mean(vals[i:j])))
        f = 1.0 if fade is None else float(np.mean(fade[i:j]))
        polyline(ctx, np.stack([xs[i:j], ys[i:j]], 1))
        set_rgba(ctx, c[:3], alpha * f * (c[3] if len(c) > 3 else 1.0))
        ctx.set_line_width(width * (0.5 + 0.5 * f))
        ctx.stroke()


LIGHT = np.array([-0.45, 0.8, 0.4]) / np.linalg.norm([-0.45, 0.8, 0.4])


def solid_surface(ctx, cam, X, Y, Z, color_fn, edge_alpha=0.5, fill_alpha=0.9, width=0.8, shade=True, fog=0.65,
                  light=LIGHT):
    """Painter's-algorithm quads of a surface grid.

    Colour comes from the mean height (Y); each quad is lit by a directional light (two-sided Lambert plus a
    small specular glint on the edges) and fades into distance fog.
    """
    P = np.stack([X, Y, Z], -1)
    sx, sy, sz = cam.project(P)
    # per-quad normals from the two diagonals
    d1 = P[1:, 1:] - P[:-1, :-1]
    d2 = P[1:, :-1] - P[:-1, 1:]
    nrm = np.cross(d1, d2)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    diff = np.abs(nrm @ light)
    view = cam.f
    half = (light - view) / np.linalg.norm(light - view)
    spec = np.abs(nrm @ half) ** 24
    depth = (sz[1:, 1:] + sz[:-1, :-1] + sz[1:, :-1] + sz[:-1, 1:]) * 0.25
    near, far = depth.min(), depth.max()
    fogk = np.clip((depth - near) / max(far - near, 1e-6), 0, 1) * fog
    hv = (Y[1:, 1:] + Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:]) * 0.25
    order = np.argsort(-depth, axis=None)
    nv1 = X.shape[1] - 1
    for q in order:
        i, j = divmod(int(q), nv1)
        c = color_fn(float(hv[i, j]))
        lit = (0.16 + 0.84 * diff[i, j]) if shade else 1.0
        fk = 1 - fogk[i, j]
        ctx.move_to(sx[i, j], sy[i, j])
        ctx.line_to(sx[i + 1, j], sy[i + 1, j])
        ctx.line_to(sx[i + 1, j + 1], sy[i + 1, j + 1])
        ctx.line_to(sx[i, j + 1], sy[i, j + 1])
        ctx.close_path()
        k = 0.42 * lit * fk
        ctx.set_source_rgba(c[0] * k, c[1] * k, c[2] * k, fill_alpha)
        ctx.fill_preserve()
        g = spec[i, j] * 0.6
        ec = (min(1.0, c[0] * (0.55 + 0.45 * lit) + g), min(1.0, c[1] * (0.55 + 0.45 * lit) + g),
              min(1.0, c[2] * (0.55 + 0.45 * lit) + g))
        set_rgba(ctx, ec, edge_alpha * (0.35 + 0.65 * fk))
        ctx.set_line_width(width * (0.6 + 0.6 * fk))
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
