#!/usr/bin/env python3
"""Deterministic test-signal generation and mono PCM WAV I/O.

Python never runs the audio path; it only produces known inputs for the
C++/Zig/Rust implementations and writes/reads the WAV subset from the spec.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_DURATION_S = 1.0
DEFAULT_AMPLITUDE = 0.5
PCM_SCALE = 32768.0

SignalFn = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# WAV I/O (RIFF/WAVE, signed 16-bit PCM LE, mono)
# ---------------------------------------------------------------------------


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples in [-1, 1] as signed 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    pcm = np.clip(np.rint(x * PCM_SCALE), -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV; return (f32 samples in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"stereo rejected in v1: {path} has {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got sampwidth={wf.getsampwidth()}")
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return pcm / np.float32(PCM_SCALE), int(sample_rate)


# ---------------------------------------------------------------------------
# Signal generators (all deterministic given seed / parameters)
# ---------------------------------------------------------------------------


def _time(n: int, sample_rate: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / float(sample_rate)


def exact_bin_tone(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    fft_size: int = 1024,
    bin_index: int = 10,
    amplitude: float = DEFAULT_AMPLITUDE,
    phase: float = 0.0,
) -> np.ndarray:
    """Sinusoid whose frequency lands exactly on an FFT bin (no spectral leakage)."""
    n = int(round(duration_s * sample_rate))
    freq = bin_index * sample_rate / fft_size
    t = _time(n, sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)


def two_tone(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    f1: float = 440.0,
    f2: float = 3000.0,
    a1: float = DEFAULT_AMPLITUDE,
    a2: float = DEFAULT_AMPLITUDE,
) -> np.ndarray:
    """Sum of two sinusoids — default pair for low-pass / high-pass filter tests."""
    n = int(round(duration_s * sample_rate))
    t = _time(n, sample_rate)
    x = a1 * np.sin(2.0 * np.pi * f1 * t) + a2 * np.sin(2.0 * np.pi * f2 * t)
    return x.astype(np.float32)


def hum(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    tone_hz: float = 440.0,
    hum_hz: float = 60.0,
    tone_amp: float = DEFAULT_AMPLITUDE,
    hum_amp: float = 0.35,
    harmonics: int = 3,
) -> np.ndarray:
    """Clean tone plus mains hum and a few odd harmonics."""
    n = int(round(duration_s * sample_rate))
    t = _time(n, sample_rate)
    x = tone_amp * np.sin(2.0 * np.pi * tone_hz * t)
    for h in range(1, harmonics + 1):
        amp = hum_amp / h
        x = x + amp * np.sin(2.0 * np.pi * hum_hz * h * t)
    peak = np.max(np.abs(x))
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


def chirp(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    f0: float = 100.0,
    f1: float = 8000.0,
    amplitude: float = DEFAULT_AMPLITUDE,
) -> np.ndarray:
    """Linear frequency sweep — the rising diagonal that justifies the STFT."""
    n = int(round(duration_s * sample_rate))
    t = _time(n, sample_rate)
    # Instantaneous phase of a linear chirp: 2π (f0 t + (f1-f0) t² / (2 T))
    T = duration_s if duration_s > 0 else 1.0
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
    return (amplitude * np.sin(phase)).astype(np.float32)


def pulsed_tone(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    frequency: float = 1000.0,
    pulse_hz: float = 5.0,
    duty: float = 0.3,
    amplitude: float = DEFAULT_AMPLITUDE,
) -> np.ndarray:
    """Tone gated by a periodic rectangular envelope — for time localization checks."""
    n = int(round(duration_s * sample_rate))
    t = _time(n, sample_rate)
    tone = amplitude * np.sin(2.0 * np.pi * frequency * t)
    period = 1.0 / pulse_hz
    gate = ((t % period) < (duty * period)).astype(np.float64)
    # Soft edges to reduce click energy
    fade = max(1, int(0.002 * sample_rate))
    if fade > 1:
        kernel = np.hanning(2 * fade)
        gate = np.convolve(gate, kernel / kernel.sum(), mode="same")
        gate = np.clip(gate, 0.0, 1.0)
    return (tone * gate).astype(np.float32)


def white_noise(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
    amplitude: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic white noise for reading filter frequency responses."""
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def silence(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    return np.zeros(n, dtype=np.float32)


GENERATORS: dict[str, SignalFn] = {
    "exact_bin": exact_bin_tone,
    "two_tone": two_tone,
    "hum": hum,
    "chirp": chirp,
    "pulsed": pulsed_tone,
    "noise": white_noise,
    "silence": silence,
}


def generate_all(
    out_dir: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = DEFAULT_DURATION_S,
) -> dict[str, Path]:
    """Write the standard suite of test WAVs into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite: dict[str, np.ndarray] = {
        "exact_bin": exact_bin_tone(sample_rate, duration_s),
        "two_tone": two_tone(sample_rate, duration_s),
        "hum": hum(sample_rate, duration_s),
        "chirp": chirp(sample_rate, duration_s),
        "pulsed": pulsed_tone(sample_rate, duration_s),
        "noise": white_noise(sample_rate, duration_s),
    }

    paths: dict[str, Path] = {}
    for name, samples in suite.items():
        path = out_dir / f"{name}.wav"
        write_wav(path, samples, sample_rate)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate deterministic mono PCM WAV test signals.",
    )
    p.add_argument(
        "signal",
        nargs="?",
        choices=sorted(GENERATORS) + ["all"],
        default="all",
        help="Signal to generate (default: all)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/generated"),
        help="Output file or directory (default: data/generated)",
    )
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Duration in seconds")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for noise")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.signal == "all":
        paths = generate_all(args.output, args.sample_rate, args.duration)
        for name, path in paths.items():
            print(f"wrote {name}: {path}")
        return 0

    kwargs: dict = {
        "sample_rate": args.sample_rate,
        "duration_s": args.duration,
    }
    if args.signal == "noise":
        kwargs["seed"] = args.seed

    samples = GENERATORS[args.signal](**kwargs)
    out = args.output
    if out.is_dir() or not out.suffix:
        out = Path(out) / f"{args.signal}.wav"
    write_wav(out, samples, args.sample_rate)
    print(f"wrote {args.signal}: {out} ({len(samples)} samples @ {args.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
