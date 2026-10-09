"""Synthesise the showcase soundtrack from the film's event list (no audio files needed).

Each splash is filtered noise ("tishhh") with a bubbly crackle and, for hard impacts, a low "plop".
Loudness and length follow the splash size, so a belly flop roars and a clean rip barely hisses.
"""
import math
import wave

import numpy as np

SR = 44100


def biquad(x, kind, f, q=0.707):
    """RBJ cookbook filter. ``f`` may be a number or an array (a sweep)."""
    n = len(x)
    f = np.broadcast_to(np.asarray(f, dtype=float), (n,))
    y = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        w = 2 * math.pi * f[i] / SR
        c, s = math.cos(w), math.sin(w)
        a = s / (2 * q)
        if kind == "hp":
            b0, b1, b2 = (1 + c) / 2, -(1 + c), (1 + c) / 2
        elif kind == "lp":
            b0, b1, b2 = (1 - c) / 2, 1 - c, (1 - c) / 2
        else:  # band pass, constant peak gain
            b0, b1, b2 = a, 0.0, -a
        a0, a1, a2 = 1 + a, -2 * c, 1 - a
        xi = x[i]
        yi = (b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def env(n, attack, tau):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / tau)


def splash(size, speed, rng):
    dur = 0.45 + min(1.7, size * 0.022)
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    hiss = biquad(biquad(noise, "hp", 1900), "bp", 5200, 0.8) * 1.6 + biquad(noise, "hp", 3500) * 0.7
    body = biquad(rng.standard_normal(n), "bp", 650, 0.9) * min(1.0, size / 35)
    # bubbles: random short blips that make the tail fizz instead of sounding like static
    crackle = np.convolve((rng.random(n) < 0.004) * rng.standard_normal(n), np.hanning(90), "same")
    crackle = biquad(crackle, "bp", 2400, 2.5) * 2.5
    sig = (hiss + body * 1.4) * env(n, 0.004, dur / 3.2) + crackle * env(n, 0.05, dur / 2.2)
    # the impact itself: a short falling thump, stronger for harder entries
    thump = np.sin(2 * math.pi * np.cumsum(np.linspace(110, 45, n)) / SR) * env(n, 0.002, 0.07) * min(1.0, size / 30)
    sig = sig / (np.abs(sig).max() + 1e-9) + thump * 0.9
    loud = min(1.0, 0.18 + size / 55) * min(1.0, 0.5 + speed / 25)
    return sig * loud


def board(rng):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    tw = np.sin(2 * math.pi * (150 + 12 * np.sin(2 * math.pi * 9 * t)) * t) * env(n, 0.003, 0.12)
    click = biquad(rng.standard_normal(n), "bp", 1200, 1.2) * env(n, 0.001, 0.012) * 0.6
    return (tw * 0.55 + click) * 0.55


def whoosh(rng):
    n = int(0.8 * SR)
    sweep = 500 * (5 ** np.sin(np.linspace(0, math.pi, n)))
    sig = biquad(rng.standard_normal(n), "bp", sweep, 1.4)
    return sig / (np.abs(sig).max() + 1e-9) * np.sin(np.linspace(0, math.pi, n)) ** 2 * 0.16


def rise(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 196 * 2 ** (t / dur)  # one octave up over the segment
    ph = 2 * math.pi * np.cumsum(f) / SR
    pad = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    return pad * np.sin(np.linspace(0, math.pi, n)) ** 1.5 * 0.035


def chime():
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    sig = sum(a * np.sin(2 * math.pi * f * t) * np.exp(-t / d) for f, a, d in ((880, 1, 0.9), (1320, 0.6, 0.7), (1760, 0.35, 0.5)))
    return sig * np.minimum(1, t / 0.005) * 0.18


def soundtrack(events, duration, path, seed=5):
    rng = np.random.default_rng(seed)
    out = np.zeros(int((duration + 2.5) * SR))
    for e in events:
        k = e["kind"]
        if k == "splash":
            s = splash(e["splash"], e.get("speed", 8), rng) * e.get("vol", 1)
        elif k == "board":
            s = board(rng)
        elif k == "whoosh":
            s = whoosh(rng)
        elif k == "rise":
            s = rise(e["dur"])
        elif k == "chime":
            s = chime()
        else:
            continue
        i = int(e["t"] * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[: j - i]
    out = np.tanh(out * 1.1) * 0.9  # gentle limiter so overlapping splashes never clip
    out = out[: int(duration * SR)]
    pcm = (np.clip(out, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm, 2).tobytes())
