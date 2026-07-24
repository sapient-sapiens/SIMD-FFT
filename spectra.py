#!/usr/bin/env python3
"""Read/write the shared .stft diagnostic format and render spectrograms.

Format (little-endian), from the project specification:

    Offset  Size  Field
    0       4     magic "STFT"
    4       4     version (u32)
    8       4     frames
    12      4     bins
    16      4     sample rate
    20      4     window size
    24      4     hop size
    28      4     flags
    32      …     f32 matrix, time-major, row-major, shape (frames, bins)

Power: P[t,k] = Re(X)^2 + Im(X)^2 for nonnegative bins (bins = N/2 + 1).
Display: D[t,k] = 10 log10(P[t,k] + ε).
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MAGIC = b"STFT"
HEADER_STRUCT = struct.Struct("<4sIIIIIII")
HEADER_SIZE = HEADER_STRUCT.size  # 32
VERSION = 1
EPS = 1e-12

# Flag bits (reserved; keep minimal)
FLAG_POWER = 0  # matrix stores power (default)
FLAG_MAGNITUDE = 1 << 0
FLAG_POST_FILTER = 1 << 1


@dataclass
class StftFile:
    version: int
    frames: int
    bins: int
    sample_rate: int
    window_size: int
    hop_size: int
    flags: int
    power: np.ndarray  # shape (frames, bins), float32/float64

    @property
    def frequencies(self) -> np.ndarray:
        """Physical frequencies for each bin: f_k = k * fs / N."""
        k = np.arange(self.bins, dtype=np.float64)
        return k * self.sample_rate / self.window_size

    @property
    def times(self) -> np.ndarray:
        """Frame center times in seconds (analysis frame starts at t*H)."""
        starts = np.arange(self.frames, dtype=np.float64) * self.hop_size / self.sample_rate
        return starts + (self.window_size / (2.0 * self.sample_rate))

    def to_db(self, eps: float = EPS) -> np.ndarray:
        return 10.0 * np.log10(np.asarray(self.power, dtype=np.float64) + eps)


def read_stft(path: str | Path) -> StftFile:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{path}: file too short for .stft header")

    magic, version, frames, bins, sample_rate, window_size, hop_size, flags = HEADER_STRUCT.unpack_from(
        data, 0
    )
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if frames <= 0 or bins <= 0:
        raise ValueError(f"{path}: invalid shape frames={frames} bins={bins}")

    expected = HEADER_SIZE + frames * bins * 4
    if len(data) < expected:
        raise ValueError(f"{path}: truncated payload (have {len(data)}, need {expected})")

    matrix = np.frombuffer(data, dtype="<f4", count=frames * bins, offset=HEADER_SIZE)
    power = matrix.reshape(frames, bins).astype(np.float64, copy=True)
    return StftFile(
        version=version,
        frames=frames,
        bins=bins,
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        flags=flags,
        power=power,
    )


def write_stft(
    path: str | Path,
    power: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    flags: int = 0,
    version: int = VERSION,
) -> None:
    """Write a power matrix as a .stft diagnostic file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    p = np.asarray(power, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"power must be 2-D (frames, bins), got shape {p.shape}")
    frames, bins = p.shape

    header = HEADER_STRUCT.pack(
        MAGIC,
        int(version),
        int(frames),
        int(bins),
        int(sample_rate),
        int(window_size),
        int(hop_size),
        int(flags),
    )
    path.write_bytes(header + p.astype("<f4", copy=False).tobytes(order="C"))


def db_limits(
    *arrays: np.ndarray,
    floor_db: float = -80.0,
    ceiling_db: float | None = None,
) -> tuple[float, float]:
    """Shared color scale for before/after spectrograms."""
    stacked = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return floor_db, 0.0
    hi = float(np.max(finite)) if ceiling_db is None else float(ceiling_db)
    lo = max(floor_db, hi + floor_db) if ceiling_db is None else hi + floor_db
    # Prefer a fixed dynamic range pinned to the global peak
    lo = hi + floor_db
    return lo, hi


def render_spectrogram(
    stft: StftFile,
    out_path: str | Path,
    *,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
    dpi: int = 120,
) -> Path:
    """Render one spectrogram; returns the written path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = stft.to_db()
    if vmin is None or vmax is None:
        lo, hi = db_limits(db)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = (
        float(stft.times[0] - (stft.hop_size / (2.0 * stft.sample_rate))) if stft.frames else 0.0,
        float(stft.times[-1] + (stft.hop_size / (2.0 * stft.sample_rate))) if stft.frames else 1.0,
        float(stft.frequencies[0]),
        float(stft.frequencies[-1]),
    )
    im = ax.imshow(
        db.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or out_path.stem)
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def render_pair(
    before: StftFile,
    after: StftFile,
    out_dir: str | Path,
    *,
    stem: str = "spectrogram",
    floor_db: float = -80.0,
    cmap: str = "magma",
) -> dict[str, Path]:
    """Render before, after, and difference with an identical color scale."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_b = before.to_db()
    db_a = after.to_db()
    if db_b.shape != db_a.shape:
        raise ValueError(f"shape mismatch: before {db_b.shape} vs after {db_a.shape}")

    vmin, vmax = db_limits(db_b, db_a, floor_db=floor_db)
    paths = {
        "before": render_spectrogram(
            before, out_dir / f"{stem}_before.png", title=f"{stem} before", vmin=vmin, vmax=vmax, cmap=cmap
        ),
        "after": render_spectrogram(
            after, out_dir / f"{stem}_after.png", title=f"{stem} after", vmin=vmin, vmax=vmax, cmap=cmap
        ),
    }

    # Difference in dB space (after − before); separate diverging scale
    diff = db_a - db_b
    peak = float(np.max(np.abs(diff))) if diff.size else 1.0
    peak = max(peak, 1e-3)
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = (
        float(before.times[0]) if before.frames else 0.0,
        float(before.times[-1]) if before.frames else 1.0,
        float(before.frequencies[0]),
        float(before.frequencies[-1]),
    )
    im = ax.imshow(
        diff.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-peak,
        vmax=peak,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{stem} difference (after − before)")
    fig.colorbar(im, ax=ax, label="Δ dB")
    fig.tight_layout()
    diff_path = out_dir / f"{stem}_diff.png"
    fig.savefig(diff_path, dpi=120)
    plt.close(fig)
    paths["diff"] = diff_path
    return paths


def power_from_complex_frames(frames: np.ndarray) -> np.ndarray:
    """Compute nonnegative-bin power from complex STFT frames (T, N) or (T, N/2+1)."""
    x = np.asarray(frames)
    if np.iscomplexobj(x):
        n = x.shape[-1]
        if n > 1 and (n & (n - 1)) == 0:
            # Full spectrum — keep DC..Nyquist
            x = x[..., : n // 2 + 1]
        return (x.real * x.real + x.imag * x.imag).astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render .stft diagnostic files as spectrograms.")
    p.add_argument("stft", type=Path, nargs="+", help="One .stft file, or before after for a pair")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/spectra"),
        help="Output image path or directory",
    )
    p.add_argument("--floor-db", type=float, default=-80.0, help="Dynamic range floor relative to peak")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--info", action="store_true", help="Print header info and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    files = [read_stft(p) for p in args.stft]

    if args.info:
        for path, stft in zip(args.stft, files):
            print(
                f"{path}: version={stft.version} frames={stft.frames} bins={stft.bins} "
                f"fs={stft.sample_rate} N={stft.window_size} H={stft.hop_size} flags={stft.flags}"
            )
        return 0

    if len(files) == 1:
        out = args.output
        if out.suffix.lower() not in {".png", ".pdf", ".svg"}:
            out = Path(out) / f"{args.stft[0].stem}.png"
        db = files[0].to_db()
        vmin, vmax = db_limits(db, floor_db=args.floor_db)
        path = render_spectrogram(
            files[0], out, title=args.title, vmin=vmin, vmax=vmax
        )
        print(f"wrote {path}")
        return 0

    if len(files) == 2:
        out_dir = args.output if args.output.suffix == "" or args.output.is_dir() else args.output.parent
        stem = args.title or f"{args.stft[0].stem}_vs_{args.stft[1].stem}"
        paths = render_pair(files[0], files[1], out_dir, stem=stem, floor_db=args.floor_db)
        for kind, path in paths.items():
            print(f"wrote {kind}: {path}")
        return 0

    raise SystemExit("pass one .stft file, or exactly two for a before/after pair")


if __name__ == "__main__":
    raise SystemExit(main())
