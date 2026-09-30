"""Re-assemble the max-bitrate film from its parts and verify the checksum.

    python output/hq/join.py            # writes output/hq/fourier_1080p_crf10.mp4

(Or: cat fourier_1080p_crf10.mp4.part* > fourier_1080p_crf10.mp4 on macOS/Linux,
 copy /b fourier_1080p_crf10.mp4.part00+fourier_1080p_crf10.mp4.part01+... out.mp4 on Windows.)
"""
import glob
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "fourier_1080p_crf10.mp4"

parts = sorted(glob.glob(os.path.join(HERE, NAME + ".part*")))
out = os.path.join(HERE, NAME)
h = hashlib.sha256()
with open(out, "wb") as f:
    for p in parts:
        with open(p, "rb") as g:
            data = g.read()
        f.write(data)
        h.update(data)
want = open(os.path.join(HERE, "SHA256SUMS")).read().split()[0]
print(f"{len(parts)} parts -> {out}")
print("checksum OK" if h.hexdigest() == want else "CHECKSUM MISMATCH")
