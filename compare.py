#!/usr/bin/env python3
"""Cross-language numerical comparison and benchmark aggregation.

Compares WAV samples and .stft power matrices pairwise across implementations,
reporting max / mean / RMS error and signal-to-error ratio. Also aggregates
machine-readable benchmark JSON into runtime, speedup, and stage plots.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from signals import read_wav
from spectra import read_stft, render_pair


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


@dataclass
class ErrorReport:
    name: str
    n: int
    max_abs: float
    mean_abs: float
    rms: float
    ser_db: float  # signal-to-error ratio

    def pretty(self) -> str:
        return (
            f"{self.name}: n={self.n}  max={self.max_abs:.6e}  "
            f"mean={self.mean_abs:.6e}  rms={self.rms:.6e}  SER={self.ser_db:.2f} dB"
        )


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if a.size != b.size:
        # Document truncation; callers still get a useful metric on the overlap.
        pass
    return a[:n], b[:n]


def compare_arrays(a: np.ndarray, b: np.ndarray, name: str = "arrays") -> ErrorReport:
    x, y = _align(a, b)
    if x.size == 0:
        return ErrorReport(name, 0, math.nan, math.nan, math.nan, math.nan)
    err = x - y
    max_abs = float(np.max(np.abs(err)))
    mean_abs = float(np.mean(np.abs(err)))
    rms = float(np.sqrt(np.mean(err * err)))
    sig = float(np.sqrt(np.mean(x * x)))
    if rms <= 0.0:
        ser_db = math.inf
    elif sig <= 0.0:
        ser_db = -math.inf
    else:
        ser_db = 20.0 * math.log10(sig / rms)
    return ErrorReport(name, int(x.size), max_abs, mean_abs, rms, ser_db)


def compare_wavs(path_a: str | Path, path_b: str | Path, name: str | None = None) -> ErrorReport:
    a, sr_a = read_wav(path_a)
    b, sr_b = read_wav(path_b)
    if sr_a != sr_b:
        raise ValueError(f"sample rate mismatch: {path_a}={sr_a} vs {path_b}={sr_b}")
    label = name or f"{Path(path_a).name} vs {Path(path_b).name}"
    return compare_arrays(a, b, label)


def compare_stft(path_a: str | Path, path_b: str | Path, name: str | None = None) -> ErrorReport:
    a = read_stft(path_a)
    b = read_stft(path_b)
    if (a.frames, a.bins) != (b.frames, b.bins):
        raise ValueError(
            f"stft shape mismatch: {path_a} {(a.frames, a.bins)} vs {path_b} {(b.frames, b.bins)}"
        )
    label = name or f"{Path(path_a).name} vs {Path(path_b).name}"
    return compare_arrays(a.power, b.power, label)


def pairwise_reports(
    paths: dict[str, Path],
    *,
    kind: str = "wav",
) -> list[ErrorReport]:
    """Compare every pair in a {label: path} map."""
    labels = sorted(paths)
    reports: list[ErrorReport] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            name = f"{la} vs {lb}"
            if kind == "wav":
                reports.append(compare_wavs(paths[la], paths[lb], name))
            elif kind == "stft":
                reports.append(compare_stft(paths[la], paths[lb], name))
            else:
                raise ValueError(f"unknown kind {kind!r}")
    return reports


def check_tolerance(report: ErrorReport, max_abs: float, rms: float) -> bool:
    ok_max = report.max_abs <= max_abs
    ok_rms = report.rms <= rms
    return ok_max and ok_rms


# ---------------------------------------------------------------------------
# Filter behavior helpers (two-tone power retention)
# ---------------------------------------------------------------------------


def band_power(samples: np.ndarray, sample_rate: int, f_lo: float, f_hi: float) -> float:
    """Welch-ish periodogram power in [f_lo, f_hi] via rFFT of the whole buffer."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    power = (spec.real * spec.real + spec.imag * spec.imag) / (window @ window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(power[mask]))


def two_tone_filter_score(
    original: np.ndarray,
    filtered: np.ndarray,
    sample_rate: int,
    keep_hz: float = 440.0,
    reject_hz: float = 3000.0,
    half_width: float = 50.0,
) -> dict[str, float]:
    """Return retained/rejected power ratios for the two-tone filter test."""
    o_keep = band_power(original, sample_rate, keep_hz - half_width, keep_hz + half_width)
    f_keep = band_power(filtered, sample_rate, keep_hz - half_width, keep_hz + half_width)
    o_rej = band_power(original, sample_rate, reject_hz - half_width, reject_hz + half_width)
    f_rej = band_power(filtered, sample_rate, reject_hz - half_width, reject_hz + half_width)

    keep_ratio = f_keep / o_keep if o_keep > 0 else math.nan
    reject_ratio = f_rej / o_rej if o_rej > 0 else math.nan
    return {
        "keep_hz": keep_hz,
        "reject_hz": reject_hz,
        "keep_ratio": keep_ratio,
        "reject_ratio": reject_ratio,
        "keep_ok": float(keep_ratio >= 0.90) if keep_ratio == keep_ratio else 0.0,
        "reject_ok": float(reject_ratio <= 0.01) if reject_ratio == reject_ratio else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark JSON aggregation
# ---------------------------------------------------------------------------


def load_benchmarks(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        text = path.read_text()
        text = text.strip()
        if not text:
            continue
        # Accept a JSON array, a single object, or NDJSON
        if text[0] == "[":
            data = json.loads(text)
            rows.extend(data)
        elif text[0] == "{":
            # Could be one object or NDJSON of objects
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def plot_runtime_vs_size(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    benchmark: str = "fft",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Group by (language, backend)
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("benchmark") != benchmark:
            continue
        key = (str(r.get("language", "?")), str(r.get("backend", "?")))
        size = int(r["fft_size"])
        med = float(r["median_ns"])
        groups.setdefault(key, []).append((size, med))

    for (lang, backend), pts in sorted(groups.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] / 1e3 for p in pts]  # µs
        ax.plot(xs, ys, marker="o", label=f"{lang}/{backend}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("FFT size")
    ax.set_ylabel("Median time (µs)")
    ax.set_title(f"Runtime vs size — {benchmark}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_speedup(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    benchmark: str = "fft",
) -> Path:
    """SIMD speedup over scalar, per language."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str, int], float] = {}
    for r in rows:
        if r.get("benchmark") != benchmark:
            continue
        key = (str(r["language"]), str(r["backend"]), int(r["fft_size"]))
        by_key[key] = float(r["median_ns"])

    fig, ax = plt.subplots(figsize=(8, 5))
    languages = sorted({k[0] for k in by_key})
    for lang in languages:
        sizes = sorted({k[2] for k in by_key if k[0] == lang})
        xs, ys = [], []
        for n in sizes:
            scalar = by_key.get((lang, "scalar", n))
            simd = by_key.get((lang, "simd", n))
            if scalar and simd and simd > 0:
                xs.append(n)
                ys.append(scalar / simd)
        if xs:
            ax.plot(xs, ys, marker="o", label=lang)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("FFT size")
    ax.set_ylabel("Speedup (scalar / simd)")
    ax.set_title(f"SIMD speedup — {benchmark}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_stage_breakdown(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    fft_size: int | None = None,
) -> Path:
    """Grouped bar chart of median times by benchmark stage."""
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate median_ns by (language, backend, benchmark)
    agg: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        if fft_size is not None and int(r.get("fft_size", -1)) != fft_size:
            continue
        key = (str(r.get("language", "?")), str(r.get("backend", "?")), str(r.get("benchmark", "?")))
        agg.setdefault(key, []).append(float(r["median_ns"]))

    stages = sorted({k[2] for k in agg})
    series = sorted({(k[0], k[1]) for k in agg})

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stages), dtype=np.float64)
    width = 0.8 / max(len(series), 1)
    for i, (lang, backend) in enumerate(series):
        heights = []
        for stage in stages:
            vals = agg.get((lang, backend, stage), [])
            heights.append((float(np.median(vals)) / 1e6) if vals else 0.0)  # ms
        ax.bar(x + i * width, heights, width=width, label=f"{lang}/{backend}")

    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("Median time (ms)")
    title = "Stage breakdown"
    if fft_size is not None:
        title += f" (N={fft_size})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_report_json(reports: list[ErrorReport], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in reports]
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare outputs and plot benchmarks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    wav = sub.add_parser("wav", help="Compare two or more WAV files pairwise")
    wav.add_argument("files", nargs="+", type=Path, help="label=path or bare paths")
    wav.add_argument("--max-abs", type=float, default=None, help="Fail if max abs error exceeds this")
    wav.add_argument("--max-rms", type=float, default=None, help="Fail if RMS error exceeds this")
    wav.add_argument("-o", "--output", type=Path, default=None, help="Write JSON report")

    stft = sub.add_parser("stft", help="Compare two or more .stft files pairwise")
    stft.add_argument("files", nargs="+", type=Path, help="label=path or bare paths")
    stft.add_argument("--max-abs", type=float, default=None)
    stft.add_argument("--max-rms", type=float, default=None)
    stft.add_argument("-o", "--output", type=Path, default=None)
    stft.add_argument(
        "--diff-dir",
        type=Path,
        default=None,
        help="If exactly two files, also render difference spectrograms here",
    )

    filt = sub.add_parser("filter-score", help="Two-tone keep/reject power ratios")
    filt.add_argument("original", type=Path)
    filt.add_argument("filtered", type=Path)
    filt.add_argument("--keep", type=float, default=440.0)
    filt.add_argument("--reject", type=float, default=3000.0)

    bench = sub.add_parser("bench", help="Aggregate benchmark JSON into plots")
    bench.add_argument("files", nargs="+", type=Path, help="Benchmark JSON / NDJSON files")
    bench.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/bench"),
        help="Output directory for plots",
    )
    bench.add_argument("--benchmark", default="fft", help="Benchmark name for runtime/speedup plots")
    bench.add_argument("--fft-size", type=int, default=None, help="Filter stage breakdown to this size")

    return p


def _parse_labeled(files: list[Path]) -> dict[str, Path]:
    """Accept 'label=path' or bare paths (label = stem)."""
    out: dict[str, Path] = {}
    for item in files:
        text = str(item)
        if "=" in text:
            label, path = text.split("=", 1)
            out[label] = Path(path)
        else:
            path = Path(text)
            label = path.stem
            # Disambiguate duplicate stems
            if label in out:
                label = f"{label}_{len(out)}"
            out[label] = path
    return out


def _run_compare(args: argparse.Namespace, kind: str) -> int:
    paths = _parse_labeled(args.files)
    if len(paths) < 2:
        raise SystemExit("need at least two files to compare")

    reports = pairwise_reports(paths, kind=kind)
    failed = False
    for r in reports:
        print(r.pretty())
        if args.max_abs is not None and r.max_abs > args.max_abs:
            print(f"  FAIL max_abs {r.max_abs:.6e} > {args.max_abs:.6e}")
            failed = True
        if args.max_rms is not None and r.rms > args.max_rms:
            print(f"  FAIL rms {r.rms:.6e} > {args.max_rms:.6e}")
            failed = True

    if args.output:
        write_report_json(reports, args.output)
        print(f"wrote {args.output}")

    if kind == "stft" and getattr(args, "diff_dir", None) and len(paths) == 2:
        labels = sorted(paths)
        a = read_stft(paths[labels[0]])
        b = read_stft(paths[labels[1]])
        rendered = render_pair(a, b, args.diff_dir, stem=f"{labels[0]}_vs_{labels[1]}")
        for k, p in rendered.items():
            print(f"wrote {k}: {p}")

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "wav":
        return _run_compare(args, "wav")

    if args.cmd == "stft":
        return _run_compare(args, "stft")

    if args.cmd == "filter-score":
        orig, sr = read_wav(args.original)
        filt, sr2 = read_wav(args.filtered)
        if sr != sr2:
            raise SystemExit(f"sample rate mismatch: {sr} vs {sr2}")
        score = two_tone_filter_score(orig, filt, sr, keep_hz=args.keep, reject_hz=args.reject)
        print(json.dumps(score, indent=2))
        ok = score["keep_ratio"] >= 0.90 and score["reject_ratio"] <= 0.01
        return 0 if ok else 1

    if args.cmd == "bench":
        rows = load_benchmarks(args.files)
        if not rows:
            raise SystemExit("no benchmark rows loaded")
        out = Path(args.output)
        p1 = plot_runtime_vs_size(rows, out / f"runtime_{args.benchmark}.png", benchmark=args.benchmark)
        p2 = plot_speedup(rows, out / f"speedup_{args.benchmark}.png", benchmark=args.benchmark)
        p3 = plot_stage_breakdown(rows, out / "stages.png", fft_size=args.fft_size)
        print(f"wrote {p1}")
        print(f"wrote {p2}")
        print(f"wrote {p3}")
        return 0

    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
