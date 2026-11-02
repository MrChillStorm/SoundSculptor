#!/usr/bin/env python3
"""Long, seamless noise loops shaped like a short recording, for simulated aircraft.

Random noise is shaped to the recording, so the loop never repeats the recording's own audio:

  - spectrum: the recording's power spectral density at the finest resolution its length allows (Welch). The whole
    loop is synthesised as one spectrum (random phases, the target magnitude in every bin) and transformed at once,
    so it is exactly periodic: no seam, and a steady recording gives steady noise (no frame-by-frame phases);
  - steady tones: a peak standing out of the broadband floor no wider than the analysis window's main lobe is a
    steady tone and becomes a pure sinusoid at its frequency and power (as noise it would beat at a few hertz).
    Wider peaks are narrowband noise and stay in the noise;
  - fluctuation: the recording's band levels (octave bands, 20 ms frames) are compared with those of steady noise
    of the same spectrum, which fluctuate by themselves. Only the excess, its modulation spectrum and how the bands
    move together (their cross-spectral matrix, smoothed over modulation frequency), is added to the noise as slow
    band gains, and the estimate is shrunk towards adding none, so a steady recording stays steady (--steady: none);
  - level: the recording's RMS; one gain for the whole loop if its true peak would pass -0.5 dBFS (a limiter would
    pump and break the loop).

Any WAV the recording comes in (8-bit unsigned, 16- or 32-bit integer, float; mono or stereo, channels separately)
is read at its own scale, and the loop is written as 16-bit at the recording's sample rate. Short recordings work
as they are: the analysis resolution follows the length.
"""

import argparse
import os
import sys

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# Bose Aviation Headset X EQ
freqs = np.array([0, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 24000])
gains = np.array([0, -2.2, -7.0, 0.0, -5.2, -7.8, -4.0, -9.0, -8.4, -7.8, -10.4, -9.8, -10.6, -10.2, -7.4, -8.8, -13.4, -13.2, -9.8, -11, -16.4, -23.0, -19.2, -18.6, -21.4, -19.0, -19.0])
gains_linear = 10 ** (gains / 20)

NFFT = {'low': 4096, 'medium': 16384, 'high': 65536}   # longest analysis segment (finest resolution)
TONE_DB = 10.0       # a steady tone stands this far out of the broadband floor ...
TONE_WIDTH = 1.5     # ... and is no wider than this many times the Hann main lobe's half-power width (1.44 bins)
HOP = 0.010          # s, band level frames (20 ms Hann windows)
EDGES = [88.0 * 2 ** k for k in range(10)]   # octave bands of the fluctuation (88 Hz ... 45 kHz)


def read_wav(path):
    """samples in -1..1 (8-bit unsigned, 16-/32-bit integer or float), channels as columns, DC removed"""
    fs, x = wav.read(path)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / -float(np.iinfo(x.dtype).min)
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return fs, x - x.mean(axis=0)


def apply_headset_filter(fs, data):
    """Apply headset EQ filter in frequency domain (circular: the loop stays seamless)"""
    X = np.fft.rfft(data)
    return np.fft.irfft(X * np.interp(np.fft.rfftfreq(len(data), 1 / fs), freqs, gains_linear), len(data))


# ---------------------------------------------------------------- spectrum and tones

def analyse(x, fs, nmax):
    """Welch PSD at the finest resolution with at least three segments, its broadband floor (running median over a
    sixth of an octave), the steady tones (frequency, power) and the PSD without them"""
    n = int(min(nmax, 2 ** np.floor(np.log2(len(x) / 2))))
    f, P = signal.welch(x, fs, window='hann', nperseg=n, detrend=False)
    df = f[1]
    lP = np.log(P + 1e-30)
    fl = np.empty_like(lP)
    for i in range(len(lP)):
        w = max(4, int(i * (2 ** (1 / 12) - 1)))
        fl[i] = np.median(lP[max(0, i - w):i + w + 1])
    floor = np.exp(fl)
    ex = 10 * (lP - fl) / np.log(10)
    pk, _ = signal.find_peaks(ex, height=TONE_DB)
    width = signal.peak_widths(P, pk, rel_height=0.5)[0] if len(pk) else []
    tones, Pn = [], P.copy()
    for k, w in zip(pk, width):
        if w > TONE_WIDTH * 1.44 or k < 2 or k > len(P) - 4:
            continue
        lo, hi = k - 3, k + 4
        p = np.sum(np.maximum(P[lo:hi] - floor[lo:hi], 0)) * df
        a, b, c = lP[k - 1], lP[k], lP[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
        tones.append(((k + d) * df, p))
        Pn[lo:hi] = floor[lo:hi]
    return f, P, Pn, tones


def spectrum(f, Pn, tones, fs, N, rng):
    """rfft spectra (noise, tones) of N-sample periodic signals: noise with the PSD Pn (random phases), tones as
    sinusoids on the nearest bin (within half a bin, 1/(2 N/fs) Hz)"""
    fk = np.fft.rfftfreq(N, 1 / fs)
    S = np.exp(np.interp(fk, f, np.log(Pn + 1e-30)))
    S[0] = 0.0
    S[fk > f[-1]] = 0.0
    Xn = N * np.sqrt(S * fs / (2 * N)) * np.exp(2j * np.pi * rng.random(len(fk)))
    Xt = np.zeros_like(Xn)
    for ft, p in tones:
        k = int(round(ft * N / fs))
        if 0 < k < len(fk) - 1:
            Xt[k] += 0.5 * N * np.sqrt(2 * p) * np.exp(2j * np.pi * rng.random())
    return Xn, Xt


# ---------------------------------------------------------------- fluctuation

def band_masks(n, fs):
    """complementary masks (they add up to 1) splitting an n-point rfft into octave bands, half-octave crossovers"""
    lf = np.log2(np.maximum(np.fft.rfftfreq(n, 1 / fs), 1e-3))
    edges = [e for e in EDGES if e * 2 ** 0.25 < fs / 2]
    low = [np.sin(0.5 * np.pi * np.clip(0.5 - (lf - np.log2(e)) * 2, 0, 1)) ** 2 for e in edges]
    return [low[0]] + [low[i + 1] - low[i] for i in range(len(low) - 1)] + [1 - low[-1]]


def band_levels(X, n, fs, hop):
    """octave band levels (dB, bands x frames) of the signal with rfft X, in 2-hop Hann windows every hop samples"""
    w = np.hanning(2 * hop + 1)[1:]
    out = []
    for m in band_masks(n, fs):
        y = np.fft.irfft(X * m, n)
        fr = np.lib.stride_tricks.sliding_window_view(y ** 2, 2 * hop)[::hop]
        out.append(10 * np.log10(fr @ w / w.sum() + 1e-30))
    return np.array(out)


def csd(D, T):
    """one-sided cross-spectral density matrices (modulation frequency x bands x bands, per cycle per frame) of the
    band level deviations D (bands x frames), Welch over T-frame Hann segments, half overlapping"""
    w = np.hanning(T + 1)[1:]
    segs = np.lib.stride_tricks.sliding_window_view(D, T, axis=1)[:, ::max(1, T // 2), :]
    F = np.fft.rfft((segs - segs.mean(axis=2, keepdims=True)) * w, axis=2)       # bands x segments x bins
    C = 2 * np.einsum('bsk,csk->kbc', F, F.conj()) / segs.shape[1] / np.sum(w ** 2)
    return C, segs.shape[1]


def smooth(C, q):
    """average over modulation frequency, a third of an octave wide (at least q bins)"""
    out = np.empty_like(C)
    for k in range(len(C)):
        h = max(q // 2, int(k * (2 ** (1 / 6) - 1)))
        out[k] = C[max(1, k - h):k + h + 1].mean(axis=0) if k else 0
    return out


def fluctuation(x, fs, Xn, Xt, N, hop, rng):
    """the band gains (bands x N samples, power-normalised) that add the recording's excess fluctuation, and how
    much (dB rms per band). A band gets the excess of its level variance over steady noise's only where that
    excess is beyond the scatter steady noise itself shows over stretches as long as the recording (2 sigma):
    so a steady recording gets none. Its modulation spectrum and how the bands move together come from the
    difference of the cross-spectral matrices, scaled to those variances."""
    Ts = (len(x) - 2 * hop) // hop
    Ls = band_levels(np.fft.rfft(x), len(x), fs, hop)[:, 3:Ts - 3]               # the recording's ends left out
    Lr = band_levels(Xn + Xt, N, fs, hop)                                        # steady noise, same spectrum
    T = Ls.shape[1]
    pieces = np.array([Lr[:, i:i + T].var(axis=1) for i in range(0, Lr.shape[1] - T + 1, max(1, T // 2))])
    excess = Ls.var(axis=1) - pieces.mean(axis=0)
    excess = np.where(excess > 2 * pieces.std(axis=0), excess, 0.0)
    if not np.any(excess > 0):
        return None, np.zeros(len(excess))
    Cs, _ = csd(Ls - Ls.mean(axis=1, keepdims=True), T)
    Cr, _ = csd(Lr - Lr.mean(axis=1, keepdims=True), T)
    Ce = smooth(Cs, 3) - smooth(Cr, 3)
    Tn = N // hop
    nu_s = np.arange(len(Ce)) / T
    nu_o = np.arange(Tn // 2 + 1) / Tn
    B = Ce.shape[1]
    Z = np.zeros((B, Tn // 2 + 1), complex)
    for k in range(1, Tn // 2):
        if nu_o[k] > nu_s[-1]:
            break
        i = np.searchsorted(nu_s, nu_o[k])
        t = (nu_o[k] - nu_s[i - 1]) / (nu_s[i] - nu_s[i - 1])
        M = (1 - t) * Ce[i - 1] + t * Ce[i]
        lam, V = np.linalg.eigh(0.5 * (M + M.conj().T))
        root = (V * np.sqrt(np.maximum(lam, 0))) @ V.conj().T
        z = (rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2)
        Z[:, k] = np.sqrt(Tn / 2) * root @ z
    Zs = np.zeros((B, N // 2 + 1), complex)
    Zs[:, :Tn // 2 + 1] = Z
    m = np.fft.irfft(Zs, N, axis=1) * (N / Tn)              # dB, at the sample rate (band limited, periodic)
    m *= np.sqrt(excess / np.maximum(m.var(axis=1), 1e-12))[:, None]
    g = 10 ** (m / 20)
    g /= np.sqrt(np.mean(g ** 2, axis=1, keepdims=True))
    return g, m.std(axis=1)


# ---------------------------------------------------------------- the loop

def make_loop(x, fs, seconds, nmax, steady, rng, label=''):
    hop = int(round(HOP * fs))
    N = int(round(seconds * fs / hop)) * hop
    f, P, Pn, tones = analyse(x, fs, nmax)
    Xn, Xt = spectrum(f, Pn, tones, fs, N, rng)
    print(f"  {label}analysis resolution {f[1]:.2f} Hz; {len(tones)} steady tone(s)"
          + (': ' + ', '.join(f"{ft:.1f} Hz ({10 * np.log10(p / np.sum(P * f[1])):+.0f} dB)" for ft, p in
                              sorted(tones, key=lambda t: -t[1])[:6]) if tones else ''))
    y = np.fft.irfft(Xn, N)
    g, added = (None, None) if steady else fluctuation(x, fs, Xn, Xt, N, hop, rng)
    if g is None and not steady:
        print(f"  {label}no fluctuation beyond steady noise's: steady")
    if g is not None:
        masks = band_masks(N, fs)
        y = sum(np.fft.irfft(Xn * mk, N) * gb for mk, gb in zip(masks, g))
        # modulating smears the spectrum a little: the bands' levels back to the recording's (one static gain each)
        Y = np.fft.rfft(y)
        Y *= sum(mk * np.sqrt(np.sum(np.abs(Xn) ** 2 * mk) / max(np.sum(np.abs(Y) ** 2 * mk), 1e-30)) for mk in masks)
        y = np.fft.irfft(Y, N)
        print(f"  {label}fluctuation added beyond steady noise, dB rms by octave band: "
              + ' '.join(f"{a:.2f}" for a in added))
    y = y + np.fft.irfft(Xt, N)
    return y * np.sqrt(np.mean(x ** 2) / np.mean(y ** 2))


def report(x, y, fs):
    """the loop against the recording: level, third-octave spectrum, level wander, the seam"""
    def wander(z, T):
        n = int(T * fs)
        fr = np.lib.stride_tricks.sliding_window_view(z, n)[::max(1, n // 2)]
        return (10 * np.log10(np.mean(fr ** 2, 1) + 1e-30)).std()

    def third_octaves(z):
        f, P = signal.welch(z, fs, nperseg=min(4096, 2 ** int(np.log2(len(z) // 2))))
        c = 1000 * 2.0 ** (np.arange(-15, 14) / 3)
        c = c[(c > 40) & (c * 2 ** (1 / 6) < fs / 2)]
        return np.array([10 * np.log10(np.sum(P[(f >= k * 2 ** (-1 / 6)) & (f < k * 2 ** (1 / 6))]) + 1e-30) for k in c])

    rms = lambda z: 20 * np.log10(np.sqrt(np.mean(z ** 2)) + 1e-30)
    d = third_octaves(y) - third_octaves(x)
    seam = abs(y[0] - y[-1]) / np.std(np.diff(y))
    print(f"  level {rms(y):.1f} dBFS (recording {rms(x):.1f}); third-octave spectrum within {np.abs(d).max():.1f} dB "
          f"of the recording's (rms {np.sqrt(np.mean(d ** 2)):.1f} dB)")
    print(f"  level wander over 20/100/400 ms: {wander(y, 0.02):.2f}/{wander(y, 0.1):.2f}/{wander(y, 0.4):.2f} dB "
          f"(recording {wander(x, 0.02):.2f}/{wander(x, 0.1):.2f}/{wander(x, 0.4):.2f}); "
          f"loop seam step {seam:.1f} x a typical sample step")


def process_audio(input_file, output_file, seconds, apply_filter, quality='high', steady=False, seed=None):
    fs, x = read_wav(input_file)
    rng = np.random.default_rng(seed)
    print(f"{input_file}: {len(x) / fs:.2f} s, {fs} Hz, {x.shape[1]} channel(s)")
    y = np.stack([make_loop(x[:, c], fs, seconds, NFFT[quality], steady, rng,
                            f"channel {c + 1}: " if x.shape[1] > 1 else '') for c in range(x.shape[1])], axis=1)
    for c in range(x.shape[1]):
        report(x[:, c], y[:, c], fs)
    if apply_filter:
        print("Applying headset filter...")
        y = np.stack([apply_headset_filter(fs, y[:, c]) for c in range(y.shape[1])], axis=1)
    # true peak (4x oversampled, circularly: the loop wraps) at most -0.5 dBFS, by one gain for the whole loop
    peak = np.abs(signal.resample(y, 4 * len(y), axis=0)).max()
    if peak > 10 ** (-0.5 / 20):
        y *= 10 ** (-0.5 / 20) / peak
        print(f"  scaled by {20 * np.log10(10 ** (-0.5 / 20) / peak):.1f} dB to keep the true peak at -0.5 dBFS")
    d = (rng.random(y.shape) - rng.random(y.shape)) / 32768.0          # triangular dither
    out = np.clip(np.round((y + d) * 32767.0), -32768, 32767).astype(np.int16)
    if out.shape[1] == 1:
        out = out[:, 0]
    wav.write(output_file, fs, out)
    print(f"Output written to {output_file} ({len(y) / fs:.2f} s)")
    if out.ndim > 1 and out.shape[1] == 2:
        split_stereo_to_mono_files(output_file, out, fs)


def split_stereo_to_mono_files(filename, data, sample_rate):
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    base_filename, ext = os.path.splitext(filename)
    left_filename = f"{base_filename}-left{ext}"
    right_filename = f"{base_filename}-right{ext}"
    wav.write(left_filename, sample_rate, left_channel)
    wav.write(right_filename, sample_rate, right_channel)


def main():
    parser = argparse.ArgumentParser(
        description='Long, seamless noise loops shaped like a short recording: its spectrum at full resolution, its '
                    'steady tones as sinusoids, and only the fluctuation it has beyond steady noise.')
    parser.add_argument('input_file', help='Input WAV file path.')
    parser.add_argument('-o', '--output_file', default='output.wav',
                        help='Output WAV file path. Default is output.wav.')
    parser.add_argument('-l', '--length', type=float, default=15,
                        help='Length of the loop in seconds. Default is 15.')
    parser.add_argument('-f', '--filter', action='store_true',
                        help='Apply headset filter to the output.')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high'], default='high',
                        help='Finest analysis resolution (segments of 4096/16384/65536 samples, as the recording\'s '
                             'length allows). Default is high.')
    parser.add_argument('--steady', action='store_true',
                        help='Steady noise only: add none of the recording\'s fluctuation.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducible noise generation.')
    args = parser.parse_args()
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
    process_audio(args.input_file, args.output_file, args.length, args.filter, args.quality, args.steady, args.seed)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nAircraft noise loop generator")
        print("=" * 50)
        print("\nShapes random noise like a short recording, as a long seamless loop:")
        print("  • its spectrum at the finest resolution the recording allows")
        print("  • its steady tones as pure sinusoids")
        print("  • only the fluctuation it has beyond steady noise (a steady recording stays steady)")
        print("  • its level, the true peak at most -0.5 dBFS; no seam")
        print("\nUsage: python3 generate_noise.py <input_file> [-o output_file] [-l length] [-f] [-q quality] [--steady] [-s seed]")
        print("\nArguments:")
        print("  <input_file>  Path to the input WAV file.")
        print("  -o            Path to the output WAV file. Default is 'output.wav'.")
        print("  -l            Length in seconds. Default is 15.")
        print("  -f            Apply headset filter to the output.")
        print("  -q            Quality: low/medium/high (finest analysis resolution). Default is high.")
        print("  --steady      Steady noise only, none of the recording's fluctuation.")
        print("  -s            Random seed for reproducible results.")
        print("\nExample: python3 generate_noise.py engine_noise.wav -o engine_loop.wav -l 30 -f -s 42\n")
    else:
        main()
