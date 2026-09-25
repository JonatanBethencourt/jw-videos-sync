"""
Landmark audio fingerprinting (Shazam-style).

IMPORTANT: this file and docs/js/fingerprint.js implement the SAME algorithm
with the SAME parameters. If you change a parameter here, change it there too
(build_index.py writes PARAMS into catalog.json and the web app refuses to run
if they differ).
"""
import numpy as np
from scipy.ndimage import maximum_filter

PARAMS = {
    "sr": 8000,          # analysis sample rate (Hz)
    "nfft": 1024,        # FFT size (samples)
    "hop": 128,          # hop between frames -> 16 ms time resolution
    "fMin": 8,           # lowest bin used (~62 Hz)
    "fMax": 480,         # highest bin used (~3750 Hz)
    "peakNT": 12,        # peak neighbourhood, +/- frames
    "peakNF": 12,        # peak neighbourhood, +/- bins
    "block": 64,         # frames per density block (~1 s)
    "peaksPerBlock": 20, # strongest peaks kept per block
    "minMag": 0.01,      # ignore peaks quieter than this (linear magnitude)
    "fanOut": 5,         # pairs per anchor peak
    "dtMin": 2,          # min frame gap in a pair
    "dtMax": 63,         # max frame gap in a pair (6 bits)
    "dfMax": 100,        # max bin gap in a pair
    "tQ": 1,             # dt is quantised by >> tQ in the hash (reverb tolerance)
}

P = PARAMS


def spectrogram(x: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrogram, shape [frames, fMax - fMin + 1]."""
    x = np.asarray(x, dtype=np.float64)
    n, hop = P["nfft"], P["hop"]
    if len(x) < n:
        return np.zeros((0, P["fMax"] - P["fMin"] + 1))
    frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic Hann
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    return spec[:, P["fMin"]: P["fMax"] + 1]


def find_peaks(mag: np.ndarray):
    """Return list of (t, f) peaks, f is the absolute FFT bin index."""
    if mag.shape[0] == 0:
        return []
    mx = maximum_filter(
        mag,
        size=(2 * P["peakNT"] + 1, 2 * P["peakNF"] + 1),
        mode="constant",
        cval=-np.inf,
    )
    ts, fs = np.nonzero((mag == mx) & (mag > P["minMag"]))
    vals = mag[ts, fs]
    blocks = ts // P["block"]
    peaks = []
    for b in np.unique(blocks):
        sel = np.nonzero(blocks == b)[0]
        if len(sel) > P["peaksPerBlock"]:
            sel = sel[np.argsort(-vals[sel])[: P["peaksPerBlock"]]]
        for i in sel:
            peaks.append((int(ts[i]), int(fs[i]) + P["fMin"]))
    peaks.sort()
    return peaks


def hashes_from_peaks(peaks):
    """Pair peaks into 24-bit hashes. Returns (hashes uint32[], times uint32[])."""
    out_h, out_t = [], []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        c = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < P["dtMin"]:
                continue
            if dt > P["dtMax"]:
                break
            if abs(f2 - f1) > P["dfMax"]:
                continue
            out_h.append((f1 << 15) | (f2 << 6) | (dt >> P["tQ"]))
            out_t.append(t1)
            c += 1
            if c >= P["fanOut"]:
                break
    return np.array(out_h, dtype=np.uint32), np.array(out_t, dtype=np.uint32)


def fingerprint(x: np.ndarray):
    """Audio at PARAMS['sr'] mono float -> (hashes, frame_times)."""
    return hashes_from_peaks(find_peaks(spectrogram(x)))


def frames_to_seconds(frames):
    return np.asarray(frames, dtype=np.float64) * P["hop"] / P["sr"]
