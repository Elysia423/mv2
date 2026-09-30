"""Shared timeline and schedules used by both the soundtrack and the picture."""
import numpy as np

BPM = 120
BEAT = 60.0 / BPM
BAR = 4 * BEAT

# ------------------------------------------------------------------ section boundaries (seconds, 120 BPM)
T_COLD = 0.0
T_TITLE = 12.0
T_EULER = 18.0
T_SQUARE = 22.0
T_TF3D = 30.0
T_PANELS = 42.0
T_BUTTERFLY = 48.0
T_QUICK = 58.0
T_PTOLEMY = 62.0
T_CHORD = 66.0
T_WIND = 72.0
T_EAR = 80.0
T_SHAZAM = 84.0
T_DROP = 88.0
T_NOISE = 94.5
T_WAVES2D = 100.0
T_IMGBUILD = 104.0
T_SPEC3D = 112.0
T_FILTER = 118.0
T_JPEG = 124.0
T_BEYOND = 130.0
VIGNETTE_NAMES = ["prism", "optics", "dna", "mri", "gw", "sph", "tides", "wifi", "quantum", "ai", "fft"]
VIGNETTES = [(n, 134.0 + 4.0 * i) for i, n in enumerate(VIGNETTE_NAMES)]
T_GW = dict(VIGNETTES)["gw"]
GW_MERGE = T_GW + 3.0
T_LAPLACE = 178.0
T_WAVELET = 184.0
T_HEAT = 190.0
T_MONTAGE = 198.0
T_FINALE = 206.0
T_END = 226.0
DURATION = T_END

IMPACTS = [T_TITLE, 50.0, T_DROP, T_BEYOND, T_FINALE]

# ------------------------------------------------------------------ chapter 1: circles -> waves
# number of circles over time (visual chain length == number of audible harmonics)
C1_COUNT_KEYS = [
    (18.0, 1), (23.5, 2), (24.0, 3), (24.5, 4), (25.0, 5), (25.5, 7), (26.0, 10), (26.5, 15),
    (27.0, 24), (27.5, 40), (28.0, 60),
]
C1_SAW = 36.0     # morph square -> saw
C1_ECG = 38.0     # morph saw -> heartbeat
C1_MORPH = 0.6


def c1_count(t):
    """Integer number of circles visible at time t."""
    n = 1
    for tk, c in C1_COUNT_KEYS:
        if t >= tk:
            n = c
    return n


def c1_shape_weights(t):
    """(w_square, w_saw, w_ecg)."""
    a = np.clip((t - C1_SAW) / C1_MORPH, 0, 1)
    b = np.clip((t - C1_ECG) / C1_MORPH, 0, 1)
    a = a * a * (3 - 2 * a)
    b = b * b * (3 - 2 * b)
    return (1 - a), a * (1 - b), b


# ------------------------------------------------------------------ harmony
NOTE_IDX = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
            "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def midi(name):
    n, o = name[:-1], int(name[-1])
    return 12 * (o + 1) + NOTE_IDX[n]


def freq(name):
    m = midi(name) if isinstance(name, str) else name
    return 440.0 * 2 ** ((m - 69) / 12.0)


CHORDS = {
    "Am": {"root": "A", "tones": ["A", "C", "E"], "pad": ["A2", "E3", "A3", "C4", "E4"], "bass": "A1"},
    "F": {"root": "F", "tones": ["F", "A", "C"], "pad": ["F2", "C3", "F3", "A3", "C4"], "bass": "F1"},
    "C": {"root": "C", "tones": ["C", "E", "G"], "pad": ["C3", "G3", "C4", "E4", "G4"], "bass": "C2"},
    "G": {"root": "G", "tones": ["G", "B", "D"], "pad": ["G2", "D3", "G3", "B3", "D4"], "bass": "G1"},
    "Am9": {"root": "A", "tones": ["A", "C", "E", "B"], "pad": ["A2", "E3", "A3", "B3", "C4", "E4"], "bass": "A1"},
}
PROG = ["Am", "F", "C", "G"]


def chord_events():
    """List of (start, end, chord_name)."""
    ev = []

    def cyc(a, b, step=4.0, start_idx=0):
        t = a
        i = start_idx
        while t < b - 1e-6:
            ev.append((t, min(b, t + step), PROG[i % 4]))
            t += step
            i += 1

    ev.append((0.0, 12.0, "Am"))
    ev.append((12.0, 15.0, "Am9"))
    ev.append((15.0, 18.0, "F"))
    cyc(18.0, 42.0)
    cyc(42.0, 66.0)
    ev.append((66.0, 80.0, "Am"))
    cyc(80.0, 88.0)
    cyc(88.0, 104.0)
    cyc(104.0, 130.0)
    cyc(130.0, 150.0)
    ev.append((150.0, 154.0, "Am"))
    ev.append((154.0, 158.0, "G"))
    cyc(158.0, 178.0)
    cyc(178.0, 190.0)
    ev.append((190.0, 194.0, "Am"))
    ev.append((194.0, 198.0, "F"))
    ev.append((198.0, 202.0, "C"))
    ev.append((202.0, 206.0, "G"))
    cyc(206.0, 222.0)
    ev.append((222.0, 230.0, "Am9"))
    return ev


def chord_at(t):
    for a, b, c in chord_events():
        if a <= t < b:
            return c
    return "Am"


# the pure chord shown in the sound chapter (exactly three sine waves)
SOUND_CHORD = [("A3", 220.0), ("C4", freq("C4")), ("E4", freq("E4"))]

# lead melody (beats, note or None) over an 8-bar Am F C G phrase (2 bars per chord)
MELODY = [
    # Am
    (1, "E5"), (1, "A5"), (0.5, "G5"), (0.5, "E5"), (1, "C5"),
    (1.5, "D5"), (0.5, "C5"), (2, "A4"),
    # F
    (1, "C5"), (1, "F5"), (0.5, "E5"), (0.5, "C5"), (1, "A4"),
    (1.5, "C5"), (0.5, "D5"), (2, "C5"),
    # C
    (1, "G4"), (1, "C5"), (1, "E5"), (1, "G5"),
    (1.5, "A5"), (0.5, "G5"), (1, "E5"), (1, "C5"),
    # G
    (1, "D5"), (1, "G5"), (0.5, "F5"), (0.5, "D5"), (1, "B4"),
    (3, "D5"), (1, None),
]


# ------------------------------------------------------------------ chapter 1 wave coefficients
KMAX = 120


def _ecg(u):
    """One heartbeat period, u in [0,1)."""
    def g(c, w, a):
        d = (u - c + 0.5) % 1.0 - 0.5
        return a * np.exp(-0.5 * (d / w) ** 2)
    return (g(0.18, 0.028, 0.13) + g(0.355, 0.009, -0.12) + g(0.39, 0.011, 1.0) + g(0.425, 0.010, -0.28)
            + g(0.64, 0.05, 0.30))


ECG_PEAK_U = 0.39


def _coeffs():
    k = np.arange(1, KMAX + 1)
    sq = np.where(k % 2 == 1, 4 / (np.pi * k), 0.0).astype(complex)
    saw = (2 * (-1.0) ** (k + 1) / (np.pi * k)).astype(complex)
    N = 2048
    u = np.arange(N) / N
    X = np.fft.fft(_ecg(u))
    ecg = (2.0 / N) * 1j * X[1:KMAX + 1]
    ecg_dc = X[0].real / N
    # scale so the R peak sits at +1 relative to the baseline
    ecg = ecg / 1.0
    return k, sq, saw, ecg, ecg_dc


C1_K, C1_SQ, C1_SAW_C, C1_ECG_C, C1_ECG_DC = _coeffs()


def c1_coeffs(t):
    """Complex coefficients c_k (k=1..KMAX): y = Im(sum c_k e^{i k theta})."""
    ws, wa, we = c1_shape_weights(t)
    return ws * C1_SQ + wa * C1_SAW_C + we * C1_ECG_C


def c1_kactive(t):
    """Highest harmonic index currently active."""
    n = c1_count(t)
    return 2 * n - 1 if t < C1_SAW else KMAX


def c1_omega(t):
    """Visual rotation speed (rev/s)."""
    return 0.5 + 0.5 * float(np.clip((t - 37.6) / 0.6, 0, 1))


_TH_T = np.arange(18.0, 46.0, 0.001)
_TH = np.concatenate([[0.0], np.cumsum([2 * np.pi * c1_omega(x) * 0.001 for x in _TH_T[:-1]])])


def c1_theta(t):
    return float(np.interp(t, _TH_T, _TH)) + (2 * np.pi * 0.5 * (t - 18.0) if t < 18.0 else 0.0)


def heart_beats():
    """Times at which the ECG R-peak passes (for the heartbeat sound)."""
    u = _TH / (2 * np.pi) - ECG_PEAK_U
    idx = np.where(np.floor(u[1:]) > np.floor(u[:-1]))[0]
    return [float(_TH_T[i]) for i in idx if _TH_T[i] > C1_ECG + 0.3 and _TH_T[i] < T_PANELS - 0.2]
