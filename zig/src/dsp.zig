const std = @import("std");

pub const Backend = enum { scalar, simd };

const lanes = std.simd.suggestVectorLength(f32) orelse 4;
const Vec = @Vector(lanes, f32);

/// Periodic Hann: w[n] = sin²(π n / N). ~9% faster than the cos form (aarch64 ReleaseFast).
fn hannWindow(out: []f32) void {
    const n: f32 = @floatFromInt(out.len);
    for (out, 0..) |*w, i| {
        const s = @sin(std.math.pi * @as(f32, @floatFromInt(i)) / n);
        w.* = s * s;
    }
}

fn cmul(ar: f32, ai: f32, br: f32, bi: f32) struct { re: f32, im: f32 } {
    return .{ .re = ar * br - ai * bi, .im = ar * bi + ai * br };
}

fn cmulV(ar: Vec, ai: Vec, br: Vec, bi: Vec) struct { re: Vec, im: Vec } {
    return .{ .re = ar * br - ai * bi, .im = ar * bi + ai * br };
}

/// W[k] = exp(-2π i k / N) for k = 0 .. N-1. Call once; reuse across transforms.
fn fillTwiddles(wr: []f32, wi: []f32) void {
    const n = wr.len;
    const n_f: f32 = @floatFromInt(n);
    for (wr, wi, 0..) |*r, *i, k| {
        const angle = -2.0 * std.math.pi * @as(f32, @floatFromInt(k)) / n_f;
        r.* = @cos(angle);
        i.* = @sin(angle);
    }
}

fn dft(re: []f32, im: []f32, scratch_re: []f32, scratch_im: []f32) void {
    const n = re.len;
    @memcpy(scratch_re[0..n], re[0..n]);
    @memcpy(scratch_im[0..n], im[0..n]);

    const n_f: f32 = @floatFromInt(n);
    for (re, im, 0..) |*out_re, *out_im, k| {
        out_re.* = 0;
        out_im.* = 0;
        for (scratch_re[0..n], scratch_im[0..n], 0..) |x, y, j| {
            const angle = -2.0 * std.math.pi * @as(f32, @floatFromInt(j)) * @as(f32, @floatFromInt(k)) / n_f;
            const c = @cos(angle);
            const s = @sin(angle);
            out_re.* += x * c - y * s;
            out_im.* += x * s + y * c;
        }
    }
}

fn stageScalar(
    in_re: []const f32,
    in_im: []const f32,
    out_re: []f32,
    out_im: []f32,
    wr: []const f32,
    wi: []const f32,
    nq: usize,
    stride: usize,
) void {
    var p: usize = 0;
    while (p < nq) : (p += 1) {
        const w1r = wr[p * stride];
        const w1i = wi[p * stride];
        const w2r = wr[2 * p * stride];
        const w2i = wi[2 * p * stride];
        const w3r = wr[3 * p * stride];
        const w3i = wi[3 * p * stride];
        var q: usize = 0;
        while (q < stride) : (q += 1) {
            const ri0 = q + stride * (4 * p + 0);
            const ri1 = q + stride * (4 * p + 1);
            const ri2 = q + stride * (4 * p + 2);
            const ri3 = q + stride * (4 * p + 3);
            const wo0 = q + stride * (p + 0 * nq);
            const wo1 = q + stride * (p + 1 * nq);
            const wo2 = q + stride * (p + 2 * nq);
            const wo3 = q + stride * (p + 3 * nq);

            const ar = in_re[ri0];
            const ai = in_im[ri0];
            const t1 = cmul(in_re[ri1], in_im[ri1], w1r, w1i);
            const t2 = cmul(in_re[ri2], in_im[ri2], w2r, w2i);
            const t3 = cmul(in_re[ri3], in_im[ri3], w3r, w3i);

            out_re[wo0] = ar + t1.re + t2.re + t3.re;
            out_im[wo0] = ai + t1.im + t2.im + t3.im;
            out_re[wo1] = ar + t1.im - t2.re - t3.im;
            out_im[wo1] = ai - t1.re - t2.im + t3.re;
            out_re[wo2] = ar - t1.re + t2.re - t3.re;
            out_im[wo2] = ai - t1.im + t2.im - t3.im;
            out_re[wo3] = ar - t1.im - t2.re + t3.im;
            out_im[wo3] = ai + t1.re - t2.im - t3.re;
        }
    }
}

/// SIMD over q (same twiddle). Requires stride >= lanes.
fn stageSimdQ(
    in_re: []const f32,
    in_im: []const f32,
    out_re: []f32,
    out_im: []f32,
    wr: []const f32,
    wi: []const f32,
    nq: usize,
    stride: usize,
) void {
    var p: usize = 0;
    while (p < nq) : (p += 1) {
        const w1rv: Vec = @splat(wr[p * stride]);
        const w1iv: Vec = @splat(wi[p * stride]);
        const w2rv: Vec = @splat(wr[2 * p * stride]);
        const w2iv: Vec = @splat(wi[2 * p * stride]);
        const w3rv: Vec = @splat(wr[3 * p * stride]);
        const w3iv: Vec = @splat(wi[3 * p * stride]);
        var q: usize = 0;
        while (q + lanes <= stride) : (q += lanes) {
            const ri0 = q + stride * (4 * p + 0);
            const ri1 = q + stride * (4 * p + 1);
            const ri2 = q + stride * (4 * p + 2);
            const ri3 = q + stride * (4 * p + 3);
            const wo0 = q + stride * (p + 0 * nq);
            const wo1 = q + stride * (p + 1 * nq);
            const wo2 = q + stride * (p + 2 * nq);
            const wo3 = q + stride * (p + 3 * nq);

            const ar: Vec = in_re[ri0..][0..lanes].*;
            const ai: Vec = in_im[ri0..][0..lanes].*;
            const br: Vec = in_re[ri1..][0..lanes].*;
            const bi: Vec = in_im[ri1..][0..lanes].*;
            const cr: Vec = in_re[ri2..][0..lanes].*;
            const ci: Vec = in_im[ri2..][0..lanes].*;
            const dr: Vec = in_re[ri3..][0..lanes].*;
            const di: Vec = in_im[ri3..][0..lanes].*;

            const t1 = cmulV(br, bi, w1rv, w1iv);
            const t2 = cmulV(cr, ci, w2rv, w2iv);
            const t3 = cmulV(dr, di, w3rv, w3iv);

            out_re[wo0..][0..lanes].* = ar + t1.re + t2.re + t3.re;
            out_im[wo0..][0..lanes].* = ai + t1.im + t2.im + t3.im;
            out_re[wo1..][0..lanes].* = ar + t1.im - t2.re - t3.im;
            out_im[wo1..][0..lanes].* = ai - t1.re - t2.im + t3.re;
            out_re[wo2..][0..lanes].* = ar - t1.re + t2.re - t3.re;
            out_im[wo2..][0..lanes].* = ai - t1.im + t2.im - t3.im;
            out_re[wo3..][0..lanes].* = ar - t1.im - t2.re + t3.im;
            out_im[wo3..][0..lanes].* = ai + t1.re - t2.im - t3.re;
        }
        while (q < stride) : (q += 1) {
            const w1r = wr[p * stride];
            const w1i = wi[p * stride];
            const w2r = wr[2 * p * stride];
            const w2i = wi[2 * p * stride];
            const w3r = wr[3 * p * stride];
            const w3i = wi[3 * p * stride];
            const ri0 = q + stride * (4 * p + 0);
            const ri1 = q + stride * (4 * p + 1);
            const ri2 = q + stride * (4 * p + 2);
            const ri3 = q + stride * (4 * p + 3);
            const wo0 = q + stride * (p + 0 * nq);
            const wo1 = q + stride * (p + 1 * nq);
            const wo2 = q + stride * (p + 2 * nq);
            const wo3 = q + stride * (p + 3 * nq);
            const ar = in_re[ri0];
            const ai = in_im[ri0];
            const t1 = cmul(in_re[ri1], in_im[ri1], w1r, w1i);
            const t2 = cmul(in_re[ri2], in_im[ri2], w2r, w2i);
            const t3 = cmul(in_re[ri3], in_im[ri3], w3r, w3i);
            out_re[wo0] = ar + t1.re + t2.re + t3.re;
            out_im[wo0] = ai + t1.im + t2.im + t3.im;
            out_re[wo1] = ar + t1.im - t2.re - t3.im;
            out_im[wo1] = ai - t1.re - t2.im + t3.re;
            out_re[wo2] = ar - t1.re + t2.re - t3.re;
            out_im[wo2] = ai - t1.im + t2.im - t3.im;
            out_re[wo3] = ar - t1.im - t2.re + t3.im;
            out_im[wo3] = ai + t1.re - t2.im - t3.re;
        }
    }
}

/// SIMD over p (per-lane twiddles). Used when stride < lanes (esp. stride=1).
fn stageSimdP(
    in_re: []const f32,
    in_im: []const f32,
    out_re: []f32,
    out_im: []f32,
    wr: []const f32,
    wi: []const f32,
    nq: usize,
    stride: usize,
) void {
    if (stride == 1) {
        var p: usize = 0;
        while (p + lanes <= nq) : (p += lanes) {
            var ar: Vec = undefined;
            var ai: Vec = undefined;
            var br: Vec = undefined;
            var bi: Vec = undefined;
            var cr: Vec = undefined;
            var ci: Vec = undefined;
            var dr: Vec = undefined;
            var di: Vec = undefined;
            var w1r: Vec = undefined;
            var w1i: Vec = undefined;
            var w2r: Vec = undefined;
            var w2i: Vec = undefined;
            var w3r: Vec = undefined;
            var w3i: Vec = undefined;
            inline for (0..lanes) |lane| {
                const pp = p + lane;
                ar[lane] = in_re[4 * pp + 0];
                ai[lane] = in_im[4 * pp + 0];
                br[lane] = in_re[4 * pp + 1];
                bi[lane] = in_im[4 * pp + 1];
                cr[lane] = in_re[4 * pp + 2];
                ci[lane] = in_im[4 * pp + 2];
                dr[lane] = in_re[4 * pp + 3];
                di[lane] = in_im[4 * pp + 3];
                w1r[lane] = wr[pp];
                w1i[lane] = wi[pp];
                w2r[lane] = wr[2 * pp];
                w2i[lane] = wi[2 * pp];
                w3r[lane] = wr[3 * pp];
                w3i[lane] = wi[3 * pp];
            }
            const t1 = cmulV(br, bi, w1r, w1i);
            const t2 = cmulV(cr, ci, w2r, w2i);
            const t3 = cmulV(dr, di, w3r, w3i);
            out_re[p..][0..lanes].* = ar + t1.re + t2.re + t3.re;
            out_im[p..][0..lanes].* = ai + t1.im + t2.im + t3.im;
            out_re[p + nq ..][0..lanes].* = ar + t1.im - t2.re - t3.im;
            out_im[p + nq ..][0..lanes].* = ai - t1.re - t2.im + t3.re;
            out_re[p + 2 * nq ..][0..lanes].* = ar - t1.re + t2.re - t3.re;
            out_im[p + 2 * nq ..][0..lanes].* = ai - t1.im + t2.im - t3.im;
            out_re[p + 3 * nq ..][0..lanes].* = ar - t1.im - t2.re + t3.im;
            out_im[p + 3 * nq ..][0..lanes].* = ai + t1.re - t2.im - t3.re;
        }
        while (p < nq) : (p += 1) {
            const ar = in_re[4 * p + 0];
            const ai = in_im[4 * p + 0];
            const t1 = cmul(in_re[4 * p + 1], in_im[4 * p + 1], wr[p], wi[p]);
            const t2 = cmul(in_re[4 * p + 2], in_im[4 * p + 2], wr[2 * p], wi[2 * p]);
            const t3 = cmul(in_re[4 * p + 3], in_im[4 * p + 3], wr[3 * p], wi[3 * p]);
            out_re[p] = ar + t1.re + t2.re + t3.re;
            out_im[p] = ai + t1.im + t2.im + t3.im;
            out_re[p + nq] = ar + t1.im - t2.re - t3.im;
            out_im[p + nq] = ai - t1.re - t2.im + t3.re;
            out_re[p + 2 * nq] = ar - t1.re + t2.re - t3.re;
            out_im[p + 2 * nq] = ai - t1.im + t2.im - t3.im;
            out_re[p + 3 * nq] = ar - t1.im - t2.re + t3.im;
            out_im[p + 3 * nq] = ai + t1.re - t2.im - t3.re;
        }
        return;
    }

    var q: usize = 0;
    while (q < stride) : (q += 1) {
        var p: usize = 0;
        while (p + lanes <= nq) : (p += lanes) {
            var ar: Vec = undefined;
            var ai: Vec = undefined;
            var br: Vec = undefined;
            var bi: Vec = undefined;
            var cr: Vec = undefined;
            var ci: Vec = undefined;
            var dr: Vec = undefined;
            var di: Vec = undefined;
            var w1r: Vec = undefined;
            var w1i: Vec = undefined;
            var w2r: Vec = undefined;
            var w2i: Vec = undefined;
            var w3r: Vec = undefined;
            var w3i: Vec = undefined;
            var wo0: [lanes]usize = undefined;
            var wo1: [lanes]usize = undefined;
            var wo2: [lanes]usize = undefined;
            var wo3: [lanes]usize = undefined;
            inline for (0..lanes) |lane| {
                const pp = p + lane;
                ar[lane] = in_re[q + stride * (4 * pp + 0)];
                ai[lane] = in_im[q + stride * (4 * pp + 0)];
                br[lane] = in_re[q + stride * (4 * pp + 1)];
                bi[lane] = in_im[q + stride * (4 * pp + 1)];
                cr[lane] = in_re[q + stride * (4 * pp + 2)];
                ci[lane] = in_im[q + stride * (4 * pp + 2)];
                dr[lane] = in_re[q + stride * (4 * pp + 3)];
                di[lane] = in_im[q + stride * (4 * pp + 3)];
                w1r[lane] = wr[pp * stride];
                w1i[lane] = wi[pp * stride];
                w2r[lane] = wr[2 * pp * stride];
                w2i[lane] = wi[2 * pp * stride];
                w3r[lane] = wr[3 * pp * stride];
                w3i[lane] = wi[3 * pp * stride];
                wo0[lane] = q + stride * (pp + 0 * nq);
                wo1[lane] = q + stride * (pp + 1 * nq);
                wo2[lane] = q + stride * (pp + 2 * nq);
                wo3[lane] = q + stride * (pp + 3 * nq);
            }
            const t1 = cmulV(br, bi, w1r, w1i);
            const t2 = cmulV(cr, ci, w2r, w2i);
            const t3 = cmulV(dr, di, w3r, w3i);
            const r0 = ar + t1.re + t2.re + t3.re;
            const s0 = ai + t1.im + t2.im + t3.im;
            const r1 = ar + t1.im - t2.re - t3.im;
            const s1 = ai - t1.re - t2.im + t3.re;
            const r2 = ar - t1.re + t2.re - t3.re;
            const s2 = ai - t1.im + t2.im - t3.im;
            const r3 = ar - t1.im - t2.re + t3.im;
            const s3 = ai + t1.re - t2.im - t3.re;
            inline for (0..lanes) |lane| {
                out_re[wo0[lane]] = r0[lane];
                out_im[wo0[lane]] = s0[lane];
                out_re[wo1[lane]] = r1[lane];
                out_im[wo1[lane]] = s1[lane];
                out_re[wo2[lane]] = r2[lane];
                out_im[wo2[lane]] = s2[lane];
                out_re[wo3[lane]] = r3[lane];
                out_im[wo3[lane]] = s3[lane];
            }
        }
        while (p < nq) : (p += 1) {
            const w1r = wr[p * stride];
            const w1i = wi[p * stride];
            const w2r = wr[2 * p * stride];
            const w2i = wi[2 * p * stride];
            const w3r = wr[3 * p * stride];
            const w3i = wi[3 * p * stride];
            const ri0 = q + stride * (4 * p + 0);
            const ri1 = q + stride * (4 * p + 1);
            const ri2 = q + stride * (4 * p + 2);
            const ri3 = q + stride * (4 * p + 3);
            const wo0 = q + stride * (p + 0 * nq);
            const wo1 = q + stride * (p + 1 * nq);
            const wo2 = q + stride * (p + 2 * nq);
            const wo3 = q + stride * (p + 3 * nq);
            const ar = in_re[ri0];
            const ai = in_im[ri0];
            const t1 = cmul(in_re[ri1], in_im[ri1], w1r, w1i);
            const t2 = cmul(in_re[ri2], in_im[ri2], w2r, w2i);
            const t3 = cmul(in_re[ri3], in_im[ri3], w3r, w3i);
            out_re[wo0] = ar + t1.re + t2.re + t3.re;
            out_im[wo0] = ai + t1.im + t2.im + t3.im;
            out_re[wo1] = ar + t1.im - t2.re - t3.im;
            out_im[wo1] = ai - t1.re - t2.im + t3.re;
            out_re[wo2] = ar - t1.re + t2.re - t3.re;
            out_im[wo2] = ai - t1.im + t2.im - t3.im;
            out_re[wo3] = ar - t1.im - t2.re + t3.im;
            out_im[wo3] = ai + t1.re - t2.im - t3.re;
        }
    }
}

/// Stockham autosort radix-4 DIT. No digit-reverse; ping-pong via `scratch_*`.
/// SIMD: vectorize over q when stride >= lanes, else over p when nq >= lanes.
fn fft(
    re: []f32,
    im: []f32,
    scratch_re: []f32,
    scratch_im: []f32,
    wr: []const f32,
    wi: []const f32,
    backend: Backend,
) void {
    const n = re.len;
    if (n <= 1) return;

    var stages: usize = 0;
    var t = n;
    while (t > 1) : (t /= 4) stages += 1;

    var in_re = re;
    var in_im = im;
    var out_re = scratch_re;
    var out_im = scratch_im;

    var stage = stages;
    while (stage >= 1) : (stage -= 1) {
        var nq = n;
        var s: usize = 0;
        while (s < stage) : (s += 1) nq /= 4;
        var stride: usize = 1;
        var u: usize = 1;
        while (u < stage) : (u += 1) stride *= 4;

        if (backend == .simd and stride >= lanes) {
            stageSimdQ(in_re, in_im, out_re, out_im, wr, wi, nq, stride);
        } else if (backend == .simd and nq >= lanes) {
            stageSimdP(in_re, in_im, out_re, out_im, wr, wi, nq, stride);
        } else {
            stageScalar(in_re, in_im, out_re, out_im, wr, wi, nq, stride);
        }

        const tr = in_re;
        const ti = in_im;
        in_re = out_re;
        in_im = out_im;
        out_re = tr;
        out_im = ti;
    }

    if (in_re.ptr != re.ptr) {
        @memcpy(re, in_re);
        @memcpy(im, in_im);
    }
}

/// Inverse via conjugate → forward FFT → conjugate + 1/N (spec §4.3).
/// Reuses the same forward twiddles `wr`/`wi`.
fn ifft(
    re: []f32,
    im: []f32,
    scratch_re: []f32,
    scratch_im: []f32,
    wr: []const f32,
    wi: []const f32,
    backend: Backend,
) void {
    const n = re.len;
    if (n == 0) return;

    if (backend == .simd) {
        const neg: Vec = @splat(-1);
        var i: usize = 0;
        while (i + lanes <= n) : (i += lanes) {
            const v: Vec = im[i..][0..lanes].*;
            im[i..][0..lanes].* = v * neg;
        }
        while (i < n) : (i += 1) im[i] = -im[i];
    } else {
        for (im) |*y| y.* = -y.*;
    }

    fft(re, im, scratch_re, scratch_im, wr, wi, backend);

    const scale: f32 = 1.0 / @as(f32, @floatFromInt(n));
    if (backend == .simd) {
        const scale_v: Vec = @splat(scale);
        const neg_scale: Vec = @splat(-scale);
        var i: usize = 0;
        while (i + lanes <= n) : (i += lanes) {
            const rv: Vec = re[i..][0..lanes].*;
            const iv: Vec = im[i..][0..lanes].*;
            re[i..][0..lanes].* = rv * scale_v;
            im[i..][0..lanes].* = iv * neg_scale;
        }
        while (i < n) : (i += 1) {
            re[i] *= scale;
            im[i] = -im[i] * scale;
        }
    } else {
        for (re, im) |*r, *imag| {
            r.* *= scale;
            imag.* = -imag.* * scale;
        }
    }
}

/// Complete frames only (spec §4.6): start at 0, drop a trailing partial frame.
fn stftFrameCount(audio_len: usize, window_size: usize, hop_size: usize) usize {
    if (audio_len < window_size or hop_size == 0) return 0;
    return (audio_len - window_size) / hop_size + 1;
}

/// Time-major complex spectra: frame `t` lives in `re/im[t*n .. (t+1)*n]`.
pub const StftSpectra = struct {
    re: []f32,
    im: []f32,
    frames: usize,
    n: usize,

    pub fn deinit(self: *StftSpectra, allocator: std.mem.Allocator) void {
        allocator.free(self.re);
        allocator.free(self.im);
        self.* = undefined;
    }
};

/// STFT analysis. Audio is never written — overlapping hops would otherwise corrupt
/// each other under in-place FFT. Per frame we window-multiply into that frame's
/// output slot (O(N)), then FFT there with a reused scratch (O(N log N)). The
/// copy is not the bottleneck; skipping a second spectrum memcpy is the win.
pub fn stft(
    allocator: std.mem.Allocator,
    audio: []const f32,
    window_size: usize,
    hop_size: usize,
    backend: Backend,
) !StftSpectra {
    const n = window_size;
    if (n == 0 or (n & (n - 1)) != 0) return error.WindowSizeNotPowerOfTwo;
    // Radix-4 Stockham: N must be 4^k (4, 16, 64, 256, 1024, …).
    var t = n;
    while (t > 1) : (t /= 4) {
        if (t % 4 != 0) return error.WindowSizeNotPowerOfFour;
    }

    const frames = stftFrameCount(audio.len, n, hop_size);
    const total = frames * n;

    const re = try allocator.alloc(f32, total);
    errdefer allocator.free(re);
    const im = try allocator.alloc(f32, total);
    errdefer allocator.free(im);

    const window = try allocator.alloc(f32, n);
    defer allocator.free(window);
    const wr = try allocator.alloc(f32, n);
    defer allocator.free(wr);
    const wi = try allocator.alloc(f32, n);
    defer allocator.free(wi);
    const scratch_re = try allocator.alloc(f32, n);
    defer allocator.free(scratch_re);
    const scratch_im = try allocator.alloc(f32, n);
    defer allocator.free(scratch_im);

    hannWindow(window);
    fillTwiddles(wr, wi);

    var frame: usize = 0;
    while (frame < frames) : (frame += 1) {
        const start = frame * hop_size;
        const out_re = re[frame * n ..][0..n];
        const out_im = im[frame * n ..][0..n];

        // Fused gather + analysis window into the spectrum slot (audio untouched).
        if (backend == .simd) {
            var i: usize = 0;
            while (i + lanes <= n) : (i += lanes) {
                const x: Vec = audio[start + i ..][0..lanes].*;
                const w: Vec = window[i..][0..lanes].*;
                out_re[i..][0..lanes].* = x * w;
            }
            while (i < n) : (i += 1) {
                out_re[i] = audio[start + i] * window[i];
            }
        } else {
            for (out_re, audio[start .. start + n], window) |*y, x, w| {
                y.* = x * w;
            }
        }
        @memset(out_im, 0);

        fft(out_re, out_im, scratch_re, scratch_im, wr, wi, backend);
    }

    return .{ .re = re, .im = im, .frames = frames, .n = n };
}

// Testing stuff from here on out

fn fillInput(re: []f32, im: []f32) void {
    for (re, im, 0..) |*r, *i, idx| {
        r.* = @sin(0.7 * @as(f32, @floatFromInt(idx)) + 0.3);
        i.* = 0;
    }
}

fn maxAbsDiff(a_re: []const f32, a_im: []const f32, b_re: []const f32, b_im: []const f32) f32 {
    var m: f32 = 0;
    for (a_re, a_im, b_re, b_im) |ar, ai, br, bi| {
        m = @max(m, @abs(ar - br));
        m = @max(m, @abs(ai - bi));
    }
    return m;
}



test "hannWindow endpoints" {
    var w: [1024]f32 = undefined;
    hannWindow(&w);
    try std.testing.expectEqual(@as(f32, 0), w[0]);
    try std.testing.expect(@abs(w[512] - 1.0) < 1e-6);
}

test "fft scalar matches dft" {
    const sizes = [_]usize{ 4, 16, 64 };
    for (sizes) |n| {
        const allocator = std.testing.allocator;
        const re = try allocator.alloc(f32, n);
        defer allocator.free(re);
        const im = try allocator.alloc(f32, n);
        defer allocator.free(im);
        const d_re = try allocator.alloc(f32, n);
        defer allocator.free(d_re);
        const d_im = try allocator.alloc(f32, n);
        defer allocator.free(d_im);
        const scratch_re = try allocator.alloc(f32, n);
        defer allocator.free(scratch_re);
        const scratch_im = try allocator.alloc(f32, n);
        defer allocator.free(scratch_im);
        const wr = try allocator.alloc(f32, n);
        defer allocator.free(wr);
        const wi = try allocator.alloc(f32, n);
        defer allocator.free(wi);

        fillTwiddles(wr, wi);
        fillInput(re, im);
        @memcpy(d_re, re);
        @memcpy(d_im, im);
        fft(re, im, scratch_re, scratch_im, wr, wi, .scalar);
        dft(d_re, d_im, scratch_re, scratch_im);
        try std.testing.expect(maxAbsDiff(re, im, d_re, d_im) < 1e-3);
    }
}

test "fft simd matches scalar" {
    const sizes = [_]usize{ 4, 16, 64, 256, 1024 };
    for (sizes) |n| {
        const allocator = std.testing.allocator;
        const s_re = try allocator.alloc(f32, n);
        defer allocator.free(s_re);
        const s_im = try allocator.alloc(f32, n);
        defer allocator.free(s_im);
        const v_re = try allocator.alloc(f32, n);
        defer allocator.free(v_re);
        const v_im = try allocator.alloc(f32, n);
        defer allocator.free(v_im);
        const scratch_re = try allocator.alloc(f32, n);
        defer allocator.free(scratch_re);
        const scratch_im = try allocator.alloc(f32, n);
        defer allocator.free(scratch_im);
        const wr = try allocator.alloc(f32, n);
        defer allocator.free(wr);
        const wi = try allocator.alloc(f32, n);
        defer allocator.free(wi);

        fillTwiddles(wr, wi);
        fillInput(s_re, s_im);
        @memcpy(v_re, s_re);
        @memcpy(v_im, s_im);
        fft(s_re, s_im, scratch_re, scratch_im, wr, wi, .scalar);
        fft(v_re, v_im, scratch_re, scratch_im, wr, wi, .simd);
        try std.testing.expect(maxAbsDiff(s_re, s_im, v_re, v_im) < 1e-5);

    }
}

test "ifft round-trips fft" {
    const sizes = [_]usize{ 4, 16, 64, 256 };
    for (sizes) |n| {
        const allocator = std.testing.allocator;
        const re = try allocator.alloc(f32, n);
        defer allocator.free(re);
        const im = try allocator.alloc(f32, n);
        defer allocator.free(im);
        const orig_re = try allocator.alloc(f32, n);
        defer allocator.free(orig_re);
        const orig_im = try allocator.alloc(f32, n);
        defer allocator.free(orig_im);
        const scratch_re = try allocator.alloc(f32, n);
        defer allocator.free(scratch_re);
        const scratch_im = try allocator.alloc(f32, n);
        defer allocator.free(scratch_im);
        const wr = try allocator.alloc(f32, n);
        defer allocator.free(wr);
        const wi = try allocator.alloc(f32, n);
        defer allocator.free(wi);

        fillTwiddles(wr, wi);
        fillInput(re, im);
        // Complex input so conjugate path is exercised.
        for (im, 0..) |*y, idx| {
            y.* = @cos(0.4 * @as(f32, @floatFromInt(idx)));
        }
        @memcpy(orig_re, re);
        @memcpy(orig_im, im);

        fft(re, im, scratch_re, scratch_im, wr, wi, .scalar);
        ifft(re, im, scratch_re, scratch_im, wr, wi, .scalar);
        try std.testing.expect(maxAbsDiff(re, im, orig_re, orig_im) < 1e-4);
    }
}

test "ifft simd matches scalar" {
    const sizes = [_]usize{ 4, 16, 64, 256, 1024 };
    for (sizes) |n| {
        const allocator = std.testing.allocator;
        const s_re = try allocator.alloc(f32, n);
        defer allocator.free(s_re);
        const s_im = try allocator.alloc(f32, n);
        defer allocator.free(s_im);
        const v_re = try allocator.alloc(f32, n);
        defer allocator.free(v_re);
        const v_im = try allocator.alloc(f32, n);
        defer allocator.free(v_im);
        const scratch_re = try allocator.alloc(f32, n);
        defer allocator.free(scratch_re);
        const scratch_im = try allocator.alloc(f32, n);
        defer allocator.free(scratch_im);
        const wr = try allocator.alloc(f32, n);
        defer allocator.free(wr);
        const wi = try allocator.alloc(f32, n);
        defer allocator.free(wi);

        fillTwiddles(wr, wi);
        fillInput(s_re, s_im);
        for (s_im, 0..) |*y, idx| {
            y.* = @cos(0.4 * @as(f32, @floatFromInt(idx)));
        }
        @memcpy(v_re, s_re);
        @memcpy(v_im, s_im);

        // Same spectrum into both backends, then inverse.
        fft(s_re, s_im, scratch_re, scratch_im, wr, wi, .scalar);
        @memcpy(v_re, s_re);
        @memcpy(v_im, s_im);
        ifft(s_re, s_im, scratch_re, scratch_im, wr, wi, .scalar);
        ifft(v_re, v_im, scratch_re, scratch_im, wr, wi, .simd);
        try std.testing.expect(maxAbsDiff(s_re, s_im, v_re, v_im) < 1e-5);
    }
}

test "stft frame count drops partial" {
    // N=16, H=4, len=40 → starts 0,4,8,12,16,20,24; 28+16=44>40 → 7 frames.
    try std.testing.expectEqual(@as(usize, 7), stftFrameCount(40, 16, 4));
    try std.testing.expectEqual(@as(usize, 0), stftFrameCount(15, 16, 4));
    try std.testing.expectEqual(@as(usize, 1), stftFrameCount(16, 16, 4));
}

test "stft exact-bin cosine peaks at k" {
    const allocator = std.testing.allocator;
    const n: usize = 64;
    const hop: usize = 16;
    const k0: usize = 5;
    const audio = try allocator.alloc(f32, n + 3 * hop);
    defer allocator.free(audio);
    const n_f: f32 = @floatFromInt(n);
    for (audio, 0..) |*x, i| {
        x.* = @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(k0)) * @as(f32, @floatFromInt(i)) / n_f);
    }

    var spec = try stft(allocator, audio, n, hop, .scalar);
    defer spec.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 4), spec.frames);

    // First frame: peak power at ±k0 (Hann spreads a little; argmax still k0).
    var best_k: usize = 0;
    var best_p: f32 = -1;
    for (0..n) |k| {
        const r = spec.re[k];
        const im = spec.im[k];
        const p = r * r + im * im;
        if (p > best_p) {
            best_p = p;
            best_k = k;
        }
    }
    try std.testing.expect(best_k == k0 or best_k == n - k0);
}

test "stft simd matches scalar" {
    const allocator = std.testing.allocator;
    const n: usize = 256;
    const hop: usize = 64;
    const audio = try allocator.alloc(f32, 1024);
    defer allocator.free(audio);
    for (audio, 0..) |*x, i| {
        x.* = @sin(0.11 * @as(f32, @floatFromInt(i)));
    }

    var a = try stft(allocator, audio, n, hop, .scalar);
    defer a.deinit(allocator);
    var b = try stft(allocator, audio, n, hop, .simd);
    defer b.deinit(allocator);

    try std.testing.expectEqual(a.frames, b.frames);
    try std.testing.expect(maxAbsDiff(a.re, a.im, b.re, b.im) < 1e-4);
}
