"""Whiten and band-pass the public LIGO GW150914 strain (LOSC tutorial files) -> assets/ligo/gw150914.npz.

The whitening is itself a Fourier trick: divide the spectrum by the noise amplitude spectrum, then invert.
"""
import os

import numpy as np
from scipy.signal import butter, filtfilt, welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "assets", "ligo")
T_EVENT = 1126259462.44
GPS0 = 1126259446


def load(name):
    import h5py
    with h5py.File(os.path.join(D, name), "r") as f:
        return np.array(f["strain/Strain"][...], dtype=np.float64)


def whiten(x, fs):
    f, psd = welch(x, fs, nperseg=4 * fs)
    X = np.fft.rfft(x * np.hanning(len(x)) ** 0.1)
    fr = np.fft.rfftfreq(len(x), 1 / fs)
    asd = np.sqrt(np.interp(fr, f, psd))
    return np.fft.irfft(X / asd, len(x))


def main():
    fs = 4096
    h = load("H-H1_LOSC_4_V2-1126259446-32.hdf5")
    l = load("L-L1_LOSC_4_V2-1126259446-32.hdf5")
    b, a = butter(4, [35 / (fs / 2), 350 / (fs / 2)], btype="band")
    hw = filtfilt(b, a, whiten(h, fs))
    lw = filtfilt(b, a, whiten(l, fs))
    t = np.arange(len(h)) / fs + GPS0 - T_EVENT
    sel = (t > -0.6) & (t < 0.2)
    # Livingston saw the wave ~7 ms earlier and with opposite sign (detector orientation)
    lw_shift = -np.roll(lw, int(0.007 * fs))
    # time-frequency map with Morlet wavelets (a wavelet transform!)
    ts = t[sel]
    x = hw[sel]
    freqs = np.geomspace(25, 400, 90)
    tf = np.zeros((len(freqs), len(ts)))
    for i, f0 in enumerate(freqs):
        sig = 6 / (2 * np.pi * f0)
        tt = np.arange(-4 * sig, 4 * sig, 1 / fs)
        w = np.exp(2j * np.pi * f0 * tt) * np.exp(-tt ** 2 / (2 * sig ** 2))
        tf[i] = np.abs(np.convolve(x, w / np.abs(w).sum(), mode="same")) ** 2
    np.savez(os.path.join(D, "gw150914.npz"), t=ts, h=x, l=lw_shift[sel], freqs=freqs, tf=tf)
    print("saved", ts.shape, tf.shape)


if __name__ == "__main__":
    main()
