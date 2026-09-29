"""Frame compositor and video/audio encoder.

Usage:
    python -m film.render audio                 # synthesize soundtrack + spectrum
    python -m film.render still 12.5 30 ...     # PNG stills at given times (build/still_*.png)
    python -m film.render video [--scale 0.5] [--from 0] [--to 232] [--out build/film.mp4]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

from . import story as S
from .core import FPS, H, W, Canvas, postprocess
from .data import BUILD


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


def render_frame(fi, scale=1.0):
    _setup()
    t = fi / FPS
    active = [sc for sc in _SCENES if sc.start - sc.fade_in <= t < sc.end + sc.fade_out]
    acc = None
    bloom_amt = 0.0
    wsum = 0.0
    fx = {"flash": 0.0, "chroma": 0.0, "shake": 0.0, "fade": 1.0}
    for sc in active:
        w = sc.weight(t)
        if w <= 0.001:
            continue
        cv = Canvas(scale)
        cv.clear((0, 0, 0))
        sc.draw(cv, t, t - sc.start)
        arr = cv.array()[..., :3].astype(np.float32)
        acc = arr * w if acc is None else acc + arr * w
        bloom_amt += sc.bloom * w
        wsum += w
        for k, v in sc.effects(t, t - sc.start).items():
            fx[k] = max(fx[k], v) if k != "fade" else min(fx[k], v)
    if acc is None:
        acc = np.zeros((int(round(H * scale)), int(round(W * scale)), 3), np.float32)
    else:
        bloom_amt /= max(wsum, 1e-6)
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
    spec = audio.spectrum_frames(out, FPS)
    np.save(os.path.join(BUILD, "spectrum.npy"), spec)
    print(f"spectrum: {spec.shape}")


def cmd_still(times, scale):
    import cv2
    for tt in times:
        fi = int(round(float(tt) * FPS))
        t0 = time.time()
        img = render_frame(fi, scale)
        p = os.path.join(BUILD, f"still_{float(tt):07.2f}.png")
        cv2.imwrite(p, img[..., :3])
        print(p, f"{time.time() - t0:.2f}s")


def cmd_video(scale, t_from, t_to, out, workers, crf):
    import multiprocessing as mp
    import imageio_ffmpeg
    _setup()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = int(round(W * scale)), int(round(H * scale))
    f0, f1 = int(round(t_from * FPS)), int(round(t_to * FPS))
    wav = os.environ.get("FILM_WAV", os.path.join(BUILD, "soundtrack.wav"))
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-"]
    if os.path.exists(wav):
        cmd += ["-ss", f"{t_from:.3f}", "-t", f"{(f1 - f0) / FPS:.3f}", "-i", wav]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-tune", "film", "-movflags", "+faststart"]
    if os.path.exists(wav):
        cmd += ["-c:a", "aac", "-b:a", "256k", "-shortest"]
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
    a = ap.parse_args()
    if a.cmd == "audio":
        cmd_audio()
    elif a.cmd == "still":
        cmd_still(a.times, a.scale)
    elif a.cmd == "video":
        cmd_video(a.scale, a.t_from, a.t_to, a.out, a.workers, a.crf)
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
