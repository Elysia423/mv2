"""Soundtrack synthesiser.

Every sound in the film is built by adding sine waves together (noise included: it is made with an
inverse FFT of random-phase spectra). The score follows the shared timeline in story.py.
"""
import math
import os

import numpy as np
from scipy.signal import fftconvolve, lfilter

from . import story as S

SR = 44100
TAIL = 3.0
N = int((S.DURATION + TAIL) * SR)
RNG = np.random.default_rng(1807)


# ------------------------------------------------------------------ primitives
def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def sine_noise(n, lo=20.0, hi=20000.0, tilt=0.0, seed=None):
    """Noise synthesised as a sum of sines with random phases (inverse FFT)."""
    r = RNG if seed is None else np.random.default_rng(seed)
    m = 1 << int(math.ceil(math.log2(max(n, 2))))
    f = np.fft.rfftfreq(m, 1 / SR)
    mag = ((f >= lo) & (f <= hi)).astype(float)
    # soft edges
    mag *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4) * 1 / (1 + (f / hi) ** 4)
    if tilt:
        mag *= (np.maximum(f, 20) / 1000.0) ** tilt
    ph = r.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(mag * np.exp(1j * ph), m)[:n]
    return x / (np.std(x) + 1e-9) * 0.3


def wavetable(harm_amps, size=4096, seed=0):
    r = np.random.default_rng(seed)
    x = np.arange(size) / size
    tab = np.zeros(size)
    for k, a in enumerate(harm_amps, start=1):
        if a != 0:
            tab += a * np.sin(2 * np.pi * k * x + r.uniform(0, 2 * np.pi))
    return tab / (np.max(np.abs(tab)) + 1e-9)


def play_table(tab, f_inst):
    """f_inst: per-sample frequency array."""
    ph = np.cumsum(f_inst) / SR
    ph = ph - np.floor(ph)
    idx = ph * len(tab)
    i0 = idx.astype(np.int64) % len(tab)
    i1 = (i0 + 1) % len(tab)
    fr = idx - np.floor(idx)
    return tab[i0] * (1 - fr) + tab[i1] * fr


def env_ar(n, attack, release, total=None, curve=3.0):
    """Attack (linear->smooth) + sustain + exponential-ish release over n samples."""
    t = np.arange(n) / SR
    dur = n / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    a = a * a * (3 - 2 * a)
    r = np.clip((dur - t) / max(release, 1e-4), 0, 1) ** (curve / 2)
    return a * r


def pan_st(x, pan=0.0):
    """Constant power pan, pan in [-1,1]."""
    a = (pan + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)])


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, t0, sig, gain=1.0, pan=0.0):
        if sig.ndim == 1:
            sig = pan_st(sig, pan)
        i0 = int(round(t0 * SR))
        if i0 < 0:
            sig = sig[:, -i0:]
            i0 = 0
        n = min(sig.shape[1], N - i0)
        if n > 0:
            self.x[:, i0:i0 + n] += gain * sig[:, :n]


# ------------------------------------------------------------------ instruments
PAD_TABS = [wavetable([1 / k * math.exp(-((k - 1) / 7.0) ** 2) for k in range(1, 24)], seed=s) for s in range(4)]
LEAD_TAB = wavetable([1 / k ** 1.1 * math.exp(-((k - 1) / 9.0) ** 2) for k in range(1, 30)], seed=11)
WARM_TAB = wavetable([1, 0.35, 0.12, 0.05, 0.02], seed=3)


def pad_note(f, dur, attack=1.2, release=2.0, bright=1.0, detune_c=9.0):
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    vib = 1 + 0.0015 * np.sin(2 * np.pi * 0.23 * t + RNG.uniform(0, 6))
    for ch in range(2):
        for v, dc in enumerate([-detune_c, detune_c * 0.4] if ch == 0 else [detune_c, -detune_c * 0.4]):
            ff = f * 2 ** (dc / 1200) * vib
            tab = PAD_TABS[(2 * ch + v) % 4]
            out[ch] += play_table(tab, ff)
    e = env_ar(n, attack, release)
    # gentle brightness swell
    return out * e * 0.5


def pluck(f, dur=1.6, bright=1.0, n_h=10):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, n_h + 1):
        if k * f > 12000:
            break
        amp = (1 / k ** 1.3) * (bright if k > 2 else 1)
        tau = 0.9 / (1 + 0.75 * (k - 1))
        x += amp * np.sin(2 * np.pi * k * f * t + 0.3 * k) * np.exp(-t / tau)
    att = np.clip(t / 0.003, 0, 1)
    return x * att * 0.35


def bell(f, dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for ratio, amp, tau in [(1, 1, 1.4), (2.0, 0.45, 0.9), (3.01, 0.25, 0.6), (4.2, 0.18, 0.4), (5.43, 0.1, 0.25)]:
        if ratio * f < 14000:
            x += amp * np.sin(2 * np.pi * ratio * f * t) * np.exp(-t / tau)
    return x * np.clip(t / 0.002, 0, 1) * 0.25


def lead_note(f, dur, release=0.25):
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    vib_amt = np.clip((t - 0.25) / 0.4, 0, 1) * 0.006
    ff = f * (1 + vib_amt * np.sin(2 * np.pi * 5.3 * t))
    x = play_table(LEAD_TAB, ff) * 0.7 + 0.35 * np.sin(2 * np.pi * np.cumsum(ff / 2) / SR)
    x2 = play_table(LEAD_TAB, ff * 2 ** (7 / 1200))
    e = env_ar(n, 0.02, release)
    hold = np.clip((dur + release - t) / release, 0, 1)
    return np.stack([(x + 0.5 * x2) * e * hold, (x + 0.5 * play_table(LEAD_TAB, ff * 2 ** (-7 / 1200))) * e * hold]) * 0.28


def bass_note(f, dur, release=0.08):
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * t
    x = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph) + 0.04 * np.sin(4 * ph)
    e = env_ar(n, 0.006, release)
    return x * e * 0.45


def kick(gain=1.0):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    fr = 44 + 120 * np.exp(-t / 0.03) + 30 * np.exp(-t / 0.2)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    x = np.sin(ph) * np.exp(-t / 0.32) * np.clip(t / 0.001, 0, 1)
    x += 0.25 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.004)
    return np.tanh(1.6 * x) * 0.6 * gain


def clap(gain=1.0):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    nz = sine_noise(n, 900, 9000)
    e = np.zeros(n)
    for d in (0.0, 0.011, 0.022):
        e += (t >= d) * np.exp(-np.clip(t - d, 0, None) / 0.008)
    e += (t >= 0.03) * np.exp(-np.clip(t - 0.03, 0, None) / 0.13)
    body = 0.4 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05)
    return (nz * e * 1.4 + body) * gain


def hat(gain=1.0, open_=False):
    n = int((0.35 if open_ else 0.08) * SR)
    t = np.arange(n) / SR
    nz = sine_noise(n, 7000, 18000)
    return nz * np.exp(-t / (0.12 if open_ else 0.022)) * gain * 1.2


def snare(gain=1.0):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    nz = sine_noise(n, 1500, 12000)
    return (nz * np.exp(-t / 0.07) * 1.3 + 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.04)) * gain


def impact(dur=5.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    fr = 28 + 60 * np.exp(-t / 0.25)
    sub = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 1.4)
    boom = sine_noise(n, 30, 2500, tilt=-0.8) * np.exp(-t / 0.5) * 2.2
    crack = sine_noise(n, 2000, 14000) * np.exp(-t / 0.05) * 0.8
    x = np.tanh(1.3 * (sub * 1.1 + boom + crack)) * np.clip(t / 0.002, 0, 1)
    return x


def riser(dur, lo=200.0, hi=3000.0, n_sines=160, seed=5, power=2.2):
    """Rising cloud of sines (a 'whoosh' made of pure tones)."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    x = np.zeros(n)
    base = np.exp(r.uniform(np.log(lo), np.log(hi), n_sines))
    for b in base:
        fr = b * 2 ** (2.5 * u ** 1.5)
        fr = np.minimum(fr, 16000)
        x += np.sin(2 * np.pi * np.cumsum(fr) / SR + r.uniform(0, 6)) / math.sqrt(b / lo)
    x /= np.std(x) + 1e-9
    e = u ** power * np.clip((dur - t) / 0.02, 0, 1)
    return x * e * 0.22


def downlifter(dur, seed=9):
    x = riser(dur, 400, 5000, 100, seed, 1)[::-1].copy()
    return x


def heartbeat():
    n = int(0.7 * SR)
    t = np.arange(n) / SR

    def thump(t0, a):
        tt = np.clip(t - t0, 0, None)
        fr = 38 + 45 * np.exp(-tt / 0.03)
        return a * (t >= t0) * np.sin(2 * np.pi * np.cumsum(fr * (t >= t0)) / SR) * np.exp(-tt / 0.09)

    return np.tanh(2.0 * (thump(0.0, 1.0) + thump(0.24, 0.7))) * 0.9


def gw_chirp(dur=3.2, t_merge=2.6):
    """Sonified gravitational-wave chirp (frequency rises as the orbit shrinks, then rings down)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    tau = np.clip(t_merge - t, 1e-3, None)
    f = np.where(t < t_merge, 55 * (tau / t_merge) ** (-3 / 8), 0)
    f = np.minimum(f, 420)
    f_rd = 420.0
    f = np.where(t >= t_merge, f_rd, f)
    ph = 2 * np.pi * np.cumsum(f) / SR
    amp = np.where(t < t_merge, (f / 420) ** 0.9, np.exp(-(t - t_merge) / 0.06))
    x = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * amp
    return x * np.clip(t / 0.3, 0, 1) * 0.6


# ------------------------------------------------------------------ effects
def reverb_ir(dur=3.2, decay=0.55, seed=2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    irs = []
    for ch in range(2):
        bright = sine_noise(n, 200, 12000, seed=seed + ch) * np.exp(-t / (decay * 0.45))
        dark = sine_noise(n, 80, 3000, seed=seed + 10 + ch) * np.exp(-t / decay)
        ir = bright * 0.6 + dark
        pre = int(0.022 * SR)
        ir = np.concatenate([np.zeros(pre), ir])
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return irs


def reverb(x, wet=0.35, dur=3.2, decay=0.55):
    irs = reverb_ir(dur, decay)
    y = np.zeros_like(x)
    for ch in range(2):
        y[ch] = fftconvolve(x[ch], irs[ch], mode="full")[: x.shape[1]]
    return y * wet


def pingpong(x, delay=0.375, fb=0.38, wet=0.3):
    """Ping-pong echo; every repeat is a little darker."""
    d = int(delay * SR)
    n = x.shape[1]
    y = np.zeros_like(x)
    cur = x.mean(0)
    for i in range(1, 10):
        cur = onepole_lp(cur, 4500) * (1.0 if i == 1 else fb)
        sh = d * i
        if sh >= n:
            break
        y[i % 2, sh:] += cur[: n - sh]
    return y * wet


def onepole_lp(x, fc):
    a = math.exp(-2 * math.pi * fc / SR)
    return lfilter([1 - a], [1, -a], x, axis=-1)


# ------------------------------------------------------------------ score
class Score:
    def __init__(self):
        self.b = {k: Bus() for k in ["drums", "kick", "bass", "pad", "arp", "lead", "fx", "tone", "chord", "bell"]}
        self.kicks = []
        self.claps = []

    # ---- helpers
    def beats(self, a, b, step=S.BEAT, offset=0.0):
        t = a + offset
        while t < b - 1e-6:
            yield t
            t += step

    def kick_at(self, t, g=1.0):
        self.b["kick"].add(t, kick(g))
        self.kicks.append((t, g))

    def clap_at(self, t, g=0.6):
        self.b["drums"].add(t, clap(g), pan=0.05)
        self.claps.append(t)

    # ---- sections
    def pads(self):
        for a, b, c in S.chord_events():
            if 80.0 <= a < 88.0:
                continue
            notes = S.CHORDS[c]["pad"]
            g = 0.16
            att, rel = 1.0, 2.2
            if a < 16:
                att, rel, g = 5.0, 0.9, 0.10
                b = 14.6
            if 16 <= a < 24:
                g = 0.2
            if 196 <= a < 204:
                g = 0.14
            if a >= 228:
                att, rel, g = 0.8, 4.0, 0.18
            for nn in notes:
                self.b["pad"].add(a, pad_note(S.freq(nn), b - a, att, rel), g)

    def intro(self):
        # sparse bell/plucks answering the first circle
        seq = [(2.0, "E5"), (4.0, "A4"), (5.0, "C5"), (7.0, "E5"), (8.0, "B4"), (9.0, "A4"), (10.5, "C5"),
               (11.0, "E5"), (12.0, "A5")]
        for t, nn in seq:
            self.b["bell"].add(t, bell(S.freq(nn), 4.0), 0.5, pan=RNG.uniform(-0.5, 0.5))
        # the 'sound of a circle' -- a pure 220 Hz sine, then three circles = three harmonics
        dur = 12.0
        n = int(dur * SR)
        t = np.arange(n) / SR + 3.0
        e = np.clip((t - 3.0) / 1.5, 0, 1) * np.clip((15.0 - t) / 0.6, 0, 1)
        x = np.sin(2 * np.pi * 220 * t)
        h2 = np.clip((t - 8.0) / 1.0, 0, 1)
        x += h2 * (0.5 * np.sin(2 * np.pi * 440 * t + 0.4) + 0.33 * np.sin(2 * np.pi * 660 * t + 1.1))
        self.b["tone"].add(3.0, x * e * 0.16)
        # rising tension then silence before the title
        self.b["fx"].add(11.0, riser(4.0, 150, 2500), 0.8)
        self.b["fx"].add(12.0, sine_noise(int(3.0 * SR), 40, 400) * np.linspace(0, 1, int(3.0 * SR)) ** 2, 0.35)

    def impacts(self):
        for t in S.IMPACTS:
            self.b["fx"].add(t, impact(), 0.9 if t != 16 else 1.1)
            self.b["fx"].add(t, sine_noise(int(0.25 * SR), 3000, 16000) * np.exp(-np.arange(int(0.25 * SR)) / SR / 0.05), 0.3)
        # softer section hits
        for t in [48.0, 80.0, 112.0, 196.0]:
            self.b["fx"].add(t, impact(3.0), 0.35)

    def title(self):
        # shimmering high arpeggio
        tones = ["A5", "E6", "B5", "C6", "E6", "A6", "B5", "E6"]
        i = 0
        for t in self.beats(16.5, 23.5, S.BEAT / 2):
            self.b["bell"].add(t, bell(S.freq(tones[i % len(tones)]), 2.5), 0.35 * (0.6 + 0.4 * (i % 2)),
                               pan=0.6 * math.sin(i * 1.3))
            i += 1
        self.b["fx"].add(21.0, riser(3.0, 300, 3000), 0.4)

    def fourier_tone(self):
        """Chapter 1: the audible Fourier series. Harmonic count follows the visual circle count."""
        a, b = 24.0, 43.6
        n = int((b - a) * SR)
        t = a + np.arange(n) / SR
        # pitch follows the chord root, eighth-note pattern
        pat = [0, 0, 12, 0, 7, 0, 12, 7]
        f_inst = np.zeros(n)
        gate = np.zeros(n)
        for k, tt in enumerate(self.beats(a, b, S.BEAT / 2)):
            root = S.CHORDS[S.chord_at(tt)]["root"]
            m = S.midi(root + "3")
            if m > S.midi("B3"):
                m -= 12
            fr = S.freq(m + pat[k % 8])
            i0, i1 = int((tt - a) * SR), min(n, int((tt - a + S.BEAT / 2) * SR))
            f_inst[i0:i1] = fr
            L = i1 - i0
            tl = np.arange(L) / SR
            g = np.clip(tl / 0.004, 0, 1) * (0.55 + 0.45 * np.exp(-tl / 0.12)) * np.clip((L / SR - tl) / 0.015, 0, 1)
            gate[i0:i1] = g * (1.0 if k % 2 == 0 else 0.8)
        f_inst[f_inst == 0] = 220.0
        # during the first single-circle phase, a steady pure sine (no pulsing) so it reads as 'one circle'
        steady = np.clip((29.5 - t) / 0.4, 0, 1)
        gate = gate * (1 - steady) + steady * np.clip((t - 24.0) / 0.3, 0, 1) * 0.85
        f_inst = f_inst * (1 - steady) + 220.0 * steady
        ph = np.cumsum(f_inst) / SR
        # control-rate harmonic amplitudes
        ctrl = np.arange(a, b, 0.005)
        amps = np.zeros((S.KMAX, len(ctrl)))
        for j, tc in enumerate(ctrl):
            c = np.abs(S.c1_coeffs(tc))
            ka = S.c1_kactive(tc)
            amps[:ka, j] = c[:ka]
        # smooth the switching so harmonics fade in over ~25 ms
        from scipy.ndimage import uniform_filter1d
        amps = uniform_filter1d(amps, 5, axis=1)
        x = np.zeros(n)
        for k in range(1, S.KMAX + 1):
            ak = np.interp(t, ctrl, amps[k - 1])
            if ak.max() < 1e-4:
                continue
            fk = f_inst * k
            ak = ak * (fk < 11000) * np.clip((11000 - fk) / 2000, 0, 1)
            x += ak * np.sin(2 * np.pi * k * ph)
        fade = np.clip((43.4 - t) / 0.5, 0, 1)
        x = onepole_lp(x, 9000) * gate * fade
        self.b["tone"].add(a, x * 0.3)
        # heartbeats
        for tb in S.heart_beats():
            self.b["kick"].add(tb - 0.02, heartbeat(), 0.9)

    def chord_section(self):
        """Exactly three pure sines: A3, C4, E4."""
        a, b = 80.0, 88.6
        n = int((b - a) * SR)
        t = np.arange(n) / SR
        x = np.zeros(n)
        for i, (_, f) in enumerate(S.SOUND_CHORD):
            e = np.clip((t - 0.3 - 0.4 * i) / 0.8, 0, 1) ** 2
            x += np.sin(2 * np.pi * f * t) * e
        x *= np.clip((b - a - t) / 1.2, 0, 1)
        self.b["chord"].add(a, x * 0.14)
        self.b["fx"].add(93.0, riser(3.0, 200, 3000), 0.6)

    def arps(self, a, b, g=0.5, octave=4, pattern=(0, 1, 2, 3, 2, 1, 3, 2), step=S.BEAT / 4, bright=1.0):
        i = 0
        for t in self.beats(a, b, step):
            c = S.CHORDS[S.chord_at(t)]
            tones = c["tones"]
            seq = []
            for o in (0, 1):
                for tn in tones[:3]:
                    seq.append(S.midi(tn + str(octave + o)))
            seq.sort()
            nn = seq[pattern[i % len(pattern)] % len(seq)]
            acc = 1.0 if i % 4 == 0 else 0.65
            self.b["arp"].add(t, pluck(S.freq(nn), 1.2, bright), g * acc, pan=0.35 * math.sin(i * 0.9))
            i += 1

    def bassline(self, a, b, style="pump", g=1.0):
        if style == "hold":
            for ca, cb, c in S.chord_events():
                s, e = max(a, ca), min(b, cb)
                if e > s:
                    self.b["bass"].add(s, bass_note(S.freq(S.CHORDS[c]["bass"]), e - s - 0.05, 0.3), g * 0.9)
            return
        for i, t in enumerate(self.beats(a, b, S.BEAT / 2)):
            c = S.CHORDS[S.chord_at(t)]
            f = S.freq(S.CHORDS[S.chord_at(t)]["bass"])
            if style == "pump":
                if i % 2 == 1:
                    self.b["bass"].add(t, bass_note(f * (2 if i % 8 == 7 else 1), S.BEAT / 2 - 0.03), g)
            else:  # drive: every eighth
                self.b["bass"].add(t, bass_note(f * (2 if i % 4 == 3 else 1), S.BEAT / 2 - 0.03), g * (1 if i % 2 else 0.85))

    def groove(self, a, b, kick_every=1, clap_on=True, hats="16", open_hats=True, g=1.0, kick_g=1.0):
        for i, t in enumerate(self.beats(a, b, S.BEAT)):
            if i % kick_every == 0:
                self.kick_at(t, kick_g)
            if clap_on and i % 2 == 1:
                self.clap_at(t, 0.55 * g)
            if open_hats:
                self.b["drums"].add(t + S.BEAT / 2, hat(0.35 * g, True), pan=0.2)
        if hats == "16":
            for j, t in enumerate(self.beats(a, b, S.BEAT / 4)):
                if open_hats and j % 4 == 2:
                    continue
                self.b["drums"].add(t, hat(g * (0.5 if j % 2 else 0.3)), pan=-0.25)
        elif hats == "8":
            for j, t in enumerate(self.beats(a, b, S.BEAT / 2)):
                if j % 2 == 1:
                    self.b["drums"].add(t, hat(g * 0.45), pan=-0.25)

    def roll(self, a, b, g=0.5):
        """Accelerating snare roll."""
        t = a
        while t < b - 0.01:
            u = (t - a) / (b - a)
            step = S.BEAT / (2 if u < 0.5 else 4 if u < 0.8 else 8)
            self.b["drums"].add(t, snare(g * (0.3 + 0.7 * u)), pan=0.1)
            t += step

    def melody(self, a, octave_shift=0, g=1.0):
        t = a
        for beats, nn in S.MELODY:
            dur = beats * S.BEAT
            if nn is not None:
                f = S.freq(nn) * 2 ** octave_shift
                self.b["lead"].add(t, lead_note(f, dur * 0.92), g)
            t += dur

    def build(self):
        self.pads()
        self.intro()
        self.impacts()
        self.title()
        self.fourier_tone()
        # chapter 1 light beat
        for t in self.beats(32.0, 42.5, S.BEAT):
            self.kick_at(t, 0.55)
        for t in self.beats(36.0, 42.5, S.BEAT, S.BEAT / 2):
            self.b["drums"].add(t, hat(0.3))
        self.bassline(32.0, 42.5, "hold", 0.6)
        # chapter 2: drawing with circles
        for t in self.beats(48.0, 56.0, 2 * S.BEAT):
            self.kick_at(t, 0.7)
        self.arps(48.0, 64.0, 0.28, 4, bright=0.7)
        self.bassline(48.0, 56.0, "hold", 0.7)
        self.bassline(56.0, 62.0, "pump", 0.8)
        self.groove(56.0, 62.0, 1, False, "8", False, 0.8, 0.85)
        self.roll(60.0, 64.0, 0.45)
        self.b["fx"].add(60.0, riser(4.0), 0.9)
        self.groove(64.0, 78.0, 1, True, "16", True, 1.0)
        self.bassline(64.0, 78.0, "pump", 1.0)
        self.arps(64.0, 80.0, 0.4, 4)
        self.b["fx"].add(77.0, downlifter(3.0), 0.5)
        # chapter 3: sound
        self.chord_section()
        self.melody_intro()
        self.groove(96.0, 112.0, 1, True, "16", True, 1.0)
        self.bassline(96.0, 112.0, "drive", 1.0)
        self.arps(96.0, 112.0, 0.35, 4)
        self.melody(96.0)
        # chapter 4: images
        self.groove(112.0, 128.0, 1, False, "8", True, 0.85)
        self.groove(128.0, 142.0, 1, True, "16", True, 0.9)
        self.bassline(112.0, 142.0, "pump", 0.9)
        self.arps(112.0, 144.0, 0.3, 4, pattern=(0, 2, 4, 5, 3, 1, 4, 2))
        self.roll(140.0, 144.0, 0.45)
        self.b["fx"].add(140.0, riser(4.0), 0.8)
        # beyond
        self.groove(144.0, 166.0, 1, True, "16", True, 1.0)
        self.bassline(144.0, 166.0, "drive", 1.0)
        self.arps(144.0, 166.0, 0.35, 5, pattern=(0, 1, 2, 3, 4, 5, 4, 2))
        self.melody(144.0, 0, 0.9)
        self.b["fx"].add(165.5, gw_chirp(3.4, 3.0), 0.9)  # merger at ~168.5
        self.b["fx"].add(168.5, impact(3.0), 0.45)
        self.bassline(166.0, 172.0, "hold", 0.6)
        self.groove(172.0, 190.0, 1, True, "16", True, 1.0)
        self.bassline(172.0, 190.0, "drive", 1.0)
        self.arps(172.0, 196.0, 0.35, 5, pattern=(0, 1, 2, 3, 4, 5, 4, 2))
        self.melody(176.0, 1, 0.75)
        self.groove(190.0, 194.0, 1, True, "8", False, 0.9)
        self.bassline(190.0, 194.0, "pump", 0.9)
        self.b["fx"].add(193.0, downlifter(3.0), 0.5)
        for t in [148.0, 154.0, 160.0, 166.0, 172.0, 178.0, 184.0, 190.0]:
            self.b["fx"].add(t - 0.6, riser(0.6, 800, 6000, 60, int(t), 1.5), 0.35)
        # heat breakdown
        seq = [(196.5, "E5"), (197.5, "A4"), (198.5, "C5"), (199.5, "B4"), (200.5, "A4"), (201.5, "C5"),
               (202.5, "F5"), (203.5, "E5")]
        for t, nn in seq:
            self.b["bell"].add(t, bell(S.freq(nn), 3.5), 0.45, pan=RNG.uniform(-0.4, 0.4))
        self.bassline(196.0, 204.0, "hold", 0.5)
        # montage build
        for t in self.beats(204.0, 212.0, S.BEAT):
            self.kick_at(t, 0.85)
        self.arps(204.0, 212.0, 0.35, 4)
        self.roll(208.0, 212.0, 0.5)
        self.b["fx"].add(206.0, riser(6.0, 150, 3000), 1.0)
        self.bassline(204.0, 212.0, "pump", 0.9)
        # finale
        self.groove(212.0, 226.0, 1, True, "16", True, 1.0)
        self.bassline(212.0, 228.0, "drive", 1.0)
        self.arps(212.0, 228.0, 0.35, 5)
        self.melody(212.0, 0, 1.0)
        self.melody(212.0, 1, 0.35)
        # bookend: a single pure circle-tone
        n = int(6.5 * SR)
        t = np.arange(n) / SR
        self.b["tone"].add(225.0, np.sin(2 * np.pi * 220 * t) * np.clip(t / 2.0, 0, 1) * np.clip((6.5 - t) / 3.0, 0, 1) * 0.12)
        self.b["bell"].add(228.0, bell(S.freq("A5"), 4.0), 0.5)
        self.b["bell"].add(228.0, bell(S.freq("E5"), 4.0), 0.4)

    def melody_intro(self):
        # 88-96: first half of the melody, softer, with plucks and pad (the 'ear' section)
        t = 88.0
        for beats, nn in S.MELODY[:16]:
            dur = beats * S.BEAT
            if nn is not None and t < 95.9:
                self.b["lead"].add(t, lead_note(S.freq(nn), dur * 0.9), 0.75)
            t += dur
        self.arps(88.0, 96.0, 0.22, 4, pattern=(0, 2, 4, 2), step=S.BEAT / 2, bright=0.6)
        self.bassline(88.0, 96.0, "hold", 0.6)

    # ---- mix
    def sidechain(self):
        env = np.zeros(N)
        tt = np.arange(int(0.45 * SR)) / SR
        shape = np.exp(-tt / 0.11) * np.clip(tt / 0.004, 0, 1)
        for t, g in self.kicks:
            i0 = int(t * SR)
            n = min(len(shape), N - i0)
            env[i0:i0 + n] = np.maximum(env[i0:i0 + n], shape[:n] * min(1, g))
        return 1 - 0.55 * env

    def mix(self):
        b = {k: v.x for k, v in self.b.items()}
        sc = self.sidechain()
        pad = b["pad"] * (0.55 + 0.45 * sc)
        bass = onepole_lp(b["bass"], 1800) * sc
        arp = b["arp"] * (0.4 + 0.6 * sc)
        lead = b["lead"]
        lead_fx = pingpong(lead, 0.375, 0.35, 0.28)
        arp_fx = pingpong(arp, 0.375, 0.3, 0.2)
        dry = (pad * 1.0 + bass * 1.0 + arp * 0.9 + lead * 1.0 + b["kick"] * 1.0 + b["drums"] * 0.8
               + b["fx"] * 0.9 + b["tone"] * 1.0 + b["chord"] * 1.0 + b["bell"] * 0.8 + lead_fx + arp_fx)
        send = (pad * 0.5 + arp * 0.6 + lead * 0.5 + b["drums"] * 0.15 + b["fx"] * 0.5 + b["bell"] * 1.0
                + b["tone"] * 0.25 + b["chord"] * 0.35 + lead_fx * 0.5)
        wet = reverb(send, 0.55, 3.5, 0.6)
        out = (dry + wet) * self.master_env()
        # glue: gentle compression + limiter
        out = self.limit(out)
        return out.astype(np.float32)

    def master_env(self):
        """A held breath before the title impact, and a clean fade at the very end."""
        t = np.arange(N) / SR
        g = np.ones(N)
        suck = np.clip((t - 15.0) / 0.15, 0, 1) * np.clip((16.0 - t) / 0.02, 0, 1)
        g *= 1 - 0.96 * suck
        g *= np.clip((S.DURATION - t) / 2.6, 0, 1) ** 1.5
        return g

    def limit(self, x, ceiling=0.93):
        from scipy.ndimage import maximum_filter1d
        x = x / (np.percentile(np.abs(x), 99.95) + 1e-9) * 0.9
        thr = 0.8
        peak = maximum_filter1d(np.max(np.abs(x), axis=0), int(0.004 * SR))
        target = np.minimum(1.0, thr / np.maximum(peak, 1e-6))
        a = math.exp(-1 / (0.15 * SR))
        g = lfilter([1 - a], [1, -a], target)
        g = np.minimum(g, target)  # never let a peak through
        y = x * g
        y = np.tanh(y * 1.2) / np.tanh(1.2)
        return y * (ceiling / (np.max(np.abs(y)) + 1e-9))


def write_wav(path, x):
    import wave
    y = np.clip(x.T, -1, 1)
    pcm = (y * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def render_soundtrack(path):
    sc = Score()
    sc.build()
    out = sc.mix()
    write_wav(path, out)
    np.save(os.path.splitext(path)[0] + "_kicks.npy", np.array(sc.kicks))
    return out, sc


def spectrum_frames(audio, fps, n_bins=256, fmin=40.0, fmax=12000.0, nfft=4096):
    """Per-video-frame log-frequency magnitude spectrum in [0,1]."""
    mono = audio.mean(0)
    n_frames = int(S.DURATION * fps) + 1
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    edges = np.geomspace(fmin, fmax, n_bins + 1)
    idx = np.searchsorted(freqs, edges)
    out = np.zeros((n_frames, n_bins), np.float32)
    pad = np.concatenate([np.zeros(nfft), mono, np.zeros(nfft)])
    for f in range(n_frames):
        c = int(f / fps * SR) + nfft
        seg = pad[c - nfft // 2: c + nfft // 2] * win
        mag = np.abs(np.fft.rfft(seg))
        row = np.array([mag[idx[i]:max(idx[i + 1], idx[i] + 1)].max() for i in range(n_bins)])
        out[f] = row
    return (20 * np.log10(out + 1e-6)).astype(np.float32)  # raw dB; mapped to [0,1] in data.spectrum()
