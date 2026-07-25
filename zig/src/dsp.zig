const std = @import("std");

pub const Backend = enum { scalar, simd };

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

fn digitReverse4Index(i: usize, digits: usize) usize {
    var x = i;
    var y: usize = 0;
    var d: usize = 0;
    while (d < digits) : (d += 1) {
        y = (y << 2) | (x & 3);
        x >>= 2;
    }
    return y;
}

fn digitReverse4(re: []f32, im: []f32) void {
    const n = re.len;
    const digits = @ctz(n) / 2; // log4(n)
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const j = digitReverse4Index(i, digits);
        if (j > i) {
            const tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            const ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }
}

/// In-place DFT oracle (scratch holds a copy of the time-domain input).
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

/// In-place scalar radix-4 FFT. Requires N = 4^p.
/// Chosen over recursive r2/r4 and iterative r2 after local benches (see commit).
fn fft(re: []f32, im: []f32) void {
    const n = re.len;
    if (n <= 1) return;

    digitReverse4(re, im);

    var len: usize = 4;
    while (len <= n) : (len *= 4) {
        const q = len / 4;
        const len_f: f32 = @floatFromInt(len);
        var start: usize = 0;
        while (start < n) : (start += len) {
            var k: usize = 0;
            while (k < q) : (k += 1) {
                const kf: f32 = @floatFromInt(k);
                const idx0 = start + k;
                const idx1 = idx0 + q;
                const idx2 = idx0 + 2 * q;
                const idx3 = idx0 + 3 * q;

                const t0r = re[idx0];
                const t0i = im[idx0];
                const w1 = cmul(re[idx1], im[idx1], @cos(-2.0 * std.math.pi * kf / len_f), @sin(-2.0 * std.math.pi * kf / len_f));
                const w2 = cmul(re[idx2], im[idx2], @cos(-4.0 * std.math.pi * kf / len_f), @sin(-4.0 * std.math.pi * kf / len_f));
                const w3 = cmul(re[idx3], im[idx3], @cos(-6.0 * std.math.pi * kf / len_f), @sin(-6.0 * std.math.pi * kf / len_f));

                re[idx0] = t0r + w1.re + w2.re + w3.re;
                im[idx0] = t0i + w1.im + w2.im + w3.im;
                re[idx1] = t0r + w1.im - w2.re - w3.im;
                im[idx1] = t0i - w1.re - w2.im + w3.re;
                re[idx2] = t0r - w1.re + w2.re - w3.re;
                im[idx2] = t0i - w1.im + w2.im - w3.im;
                re[idx3] = t0r - w1.im - w2.re + w3.im;
                im[idx3] = t0i + w1.re - w2.im - w3.re;
            }
        }
    }
}

test "hannWindow endpoints" {
    var w: [1024]f32 = undefined;
    hannWindow(&w);
    try std.testing.expectEqual(@as(f32, 0), w[0]);
    try std.testing.expect(@abs(w[512] - 1.0) < 1e-6);
}

test "fft matches dft" {
    // Powers of 4 only (radix-4).
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

        for (re, d_re, im, d_im, 0..) |*r, *dr, *i, *di, idx| {
            const x = @sin(0.7 * @as(f32, @floatFromInt(idx)) + 0.3);
            r.* = x;
            dr.* = x;
            i.* = 0;
            di.* = 0;
        }

        fft(re, im);
        dft(d_re, d_im, scratch_re, scratch_im);

        var max_abs: f32 = 0;
        for (re, im, d_re, d_im) |fr, fi, dr, di| {
            max_abs = @max(max_abs, @abs(fr - dr));
            max_abs = @max(max_abs, @abs(fi - di));
        }
        try std.testing.expect(max_abs < 1e-3);
    }
}
