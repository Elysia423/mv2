"""Frame compositor and video/audio encoder.

Usage:
    python -m film.render audio                 # synthesize soundtrack + spectrum
    python -m film.render still 12.5 30 ...     # PNG stills at given times (build/still_*.png)
    python -m film.render video [--scale 0.5] [--from 0] [--to 232] [--out build/film.mp4]
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

from . import story as S
from . import core
from .core import H, W, Canvas, postprocess
from .data import BUILD, SPEC_RATE


_SCENES = None
_OVERLAYS = None


def _setup():
    global _SCENES, _OVERLAYS
    if _SCENES is None:
        from .scenes import all_scenes, overlays
        _SCENES = all_scenes()
        _OVERLAYS = overlays()
        for sc in _SCENES:
            sc.prepare()
        for i, sc in enumerate(_SCENES):
            if getattr(sc, "trans", None) and i > 0:
                sc.fade_in = 0.0
                _prev_scene(sc).fade_out = 0.0


def _prev_scene(sc):
    """The scene on screen just before `sc` starts."""
    cands = [p for p in _SCENES if p is not sc and p.start < sc.start <= p.end + p.fade_out + 0.5]
    return max(cands, key=lambda p: p.start)


def scene_array(sc, t, scale, clamp_t=None):
    """Render one scene to float32 HxWx3 in 0..255, averaging sub-frames for motion blur."""
    # BLUR counts are tuned for 30 fps; at higher rates each frame covers less motion and needs fewer samples
    n = max(1, int(round(getattr(sc, "blur", 1) * 30 / core.FPS)))
    shutter = 0.5 / core.FPS  # 180-degree shutter
    offs = [0.0] if n == 1 else [((k + 0.5) / n - 0.5) * shutter for k in range(n)]
    acc = None
    for o in offs:
        tt = t + o
        cv = Canvas(scale)
        cv.clear((0, 0, 0))
        sc.draw(cv, tt, tt - sc.start)
        arr = cv.array()[..., :3].astype(np.float32)
        acc = arr if acc is None else acc + arr
    return acc / len(offs)


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _circle_mask(h, w, r, scale):
    import cv2
    m = np.zeros((h, w), np.float32)
    if r > 0:
        cv2.circle(m, (w // 2, h // 2), int(r * scale), 1.0, -1, lineType=cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 3 * scale)
    return m[..., None]


def transition_frame(prev, sc, t, scale):
    """Composite the hand-over from `prev` to `sc` (sc.trans = (kind, pre, post, arg))."""
    import cv2
    kind, pre, post, arg = sc.trans
    h, w = int(round(H * scale)), int(round(W * scale))
    extra = Canvas(scale)
    extra.clear((0, 0, 0))
    if kind == "iris":
        rmax = math.hypot(W, H) / 2 + 40
        t0 = sc.start
        if t < t0:
            u = _smoothstep((t - (t0 - pre)) / (pre - 0.25))
            r = rmax * (1 - u) ** 1.6
            A = scene_array(prev, t, scale)
            out = A * _circle_mask(h, w, r, scale)
        else:
            u = _smoothstep((t - (t0 + 0.3)) / (post - 0.3))
            r = rmax * u ** 1.6
            out = scene_array(sc, t, scale) * _circle_mask(h, w, r, scale) if r > 0 else np.zeros((h, w, 3), np.float32)
        ctx = extra.ctx
        from .core import CYAN, GOLD, WHITE, circle, draw_text, glow_dot
        if 1 < r < rmax - 5:
            circle(ctx, W / 2, H / 2, r, CYAN, 0.85, width=3)
            circle(ctx, W / 2, H / 2, r + 8, CYAN, 0.25, width=8)
        if r < 12:
            glow_dot(ctx, W / 2, H / 2, 6, WHITE, 1.0, 6)
        if arg:
            qa = _smoothstep((t - (t0 - 0.45)) / 0.2) * (1 - _smoothstep((t - (t0 + 0.5)) / 0.3))
            if qa > 0:
                draw_text(ctx, arg, W / 2, H / 2 + 110, size=72, font="serif", weight=700, color=WHITE, alpha=qa,
                          tracking=0.12, glow=12, glow_alpha=0.45)
        return out + extra.array()[..., :3].astype(np.float32), 0.9
    t0 = sc.start
    u = _smoothstep((t - (t0 - pre)) / (pre + post))
    A = scene_array(prev, t, scale)
    B = scene_array(sc, t, scale)
    if kind in ("push_l", "push_r"):
        off = int(round(w * u))
        out = np.zeros_like(A)
        if kind == "push_l":
            out[:, :w - off] = A[:, off:]
            out[:, w - off:] = B[:, :off]
        else:
            out[:, off:] = A[:, :w - off]
            out[:, :off] = B[:, w - off:]
        speed = 4 * u * (1 - u)  # 0..1, peaks mid-way
        k = int(1 + 90 * scale * speed)
        if k > 1:
            out = cv2.blur(out, (k, 1))
        return out, 0.9
    if kind == "zoom":
        def zoom(img, s):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        out = zoom(A, 1 + 0.35 * u) * (1 - u) + zoom(B, 0.8 + 0.2 * u) * u
        return out, 0.9
    return A * (1 - u) + B * u, 0.9


def render_frame(fi, scale=1.0):
    _setup()
    t = fi / core.FPS
    fx = {"flash": 0.0, "chroma": 0.0, "shake": 0.0, "fade": 1.0}
    acc = None
    bloom_amt = 0.0
    tr = None
    for i, sc in enumerate(_SCENES):
        if getattr(sc, "trans", None) and i > 0:
            kind, pre, post, _ = sc.trans
            if sc.start - pre <= t < sc.start + post:
                tr = (_prev_scene(sc), sc)
    if tr is not None:
        prev, sc = tr
        acc, bloom_amt = transition_frame(prev, sc, t, scale)
        for s_ in (prev, sc):
            for k, v in s_.effects(t, t - s_.start).items():
                fx[k] = max(fx[k], v) if k != "fade" else min(fx[k], v)
    else:
        active = [sc for sc in _SCENES if sc.start - sc.fade_in <= t < sc.end + sc.fade_out]
        wsum = 0.0
        for sc in active:
            w = sc.weight(t)
            if w <= 0.001:
                continue
            arr = scene_array(sc, t, scale)
            acc = arr * w if acc is None else acc + arr * w
            bloom_amt += sc.bloom * w
            wsum += w
            for k, v in sc.effects(t, t - sc.start).items():
                fx[k] = max(fx[k], v) if k != "fade" else min(fx[k], v)
        if acc is not None:
            bloom_amt /= max(wsum, 1e-6)
    if acc is None:
        acc = np.zeros((int(round(H * scale)), int(round(W * scale)), 3), np.float32)
    base = np.empty(acc.shape[:2] + (4,), np.uint8)
    base[..., :3] = np.clip(acc, 0, 255).astype(np.uint8)
    base[..., 3] = 255
    ov = Canvas(scale)
    ov.clear((0, 0, 0), 0.0)
    for o in _OVERLAYS:
        o.draw(ov, t)
    out = postprocess(base, fi, bloom_amt, ov.array(), fx["fade"], fx["flash"], chroma=fx["chroma"])
    return out


def _worker(args):
    fi, scale = args
    return fi, render_frame(fi, scale).tobytes()


def cmd_audio():
    from . import audio
    os.makedirs(BUILD, exist_ok=True)
    t0 = time.time()
    out, _ = audio.render_soundtrack(os.path.join(BUILD, "soundtrack.wav"))
    print(f"soundtrack: {time.time() - t0:.1f}s")
    spec = audio.spectrum_frames(out, SPEC_RATE)
    np.save(os.path.join(BUILD, "spectrum.npy"), spec)
    print(f"spectrum: {spec.shape}")


def cmd_still(times, scale):
    import cv2
    for tt in times:
        fi = int(round(float(tt) * core.FPS))
        t0 = time.time()
        img = render_frame(fi, scale)
        p = os.path.join(BUILD, f"still_{float(tt):07.2f}.png")
        cv2.imwrite(p, img[..., :3])
        print(p, f"{time.time() - t0:.2f}s")


def cmd_video(scale, t_from, t_to, out, workers, crf, preset="medium", abr="256k"):
    import multiprocessing as mp
    import imageio_ffmpeg
    _setup()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = int(round(W * scale)), int(round(H * scale))
    FPS = core.FPS
    f0, f1 = int(round(t_from * FPS)), int(round(t_to * FPS))
    wav = os.environ.get("FILM_WAV", os.path.join(BUILD, "soundtrack.wav"))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-"]
    if os.path.exists(wav):
        cmd += ["-ss", f"{t_from:.3f}", "-t", f"{(f1 - f0) / FPS:.3f}", "-i", wav]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-tune", "film", "-movflags", "+faststart"]
    if os.path.exists(wav):
        cmd += ["-c:a", "aac", "-b:a", abr, "-shortest"]
    cmd += [out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for k, (fi, buf) in enumerate(pool.imap(_worker, [(i, scale) for i in range(f0, f1)], chunksize=2)):
            proc.stdin.write(buf)
            if k % 150 == 0:
                el = time.time() - t0
                print(f"frame {fi} ({fi / FPS:.1f}s)  {k / max(el, 1e-6):.2f} fps", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"done in {time.time() - t0:.1f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("times", nargs="*")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=S.DURATION)
    ap.add_argument("--out", default=os.path.join(BUILD, "fourier.mp4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--abr", default="256k", help="AAC audio bitrate")
    ap.add_argument("--fps", type=int, default=None, help="frame rate (default 60, or $FILM_FPS)")
    a = ap.parse_args()
    if a.fps:
        core.FPS = a.fps
    if a.cmd == "audio":
        cmd_audio()
    elif a.cmd == "still":
        cmd_still(a.times, a.scale)
    elif a.cmd == "video":
        cmd_video(a.scale, a.t_from, a.t_to, a.out, a.workers, a.crf, a.preset, a.abr)
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
