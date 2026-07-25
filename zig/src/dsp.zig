const std = @import("std");

pub const Backend = enum{scalar, simd}; 

/// Periodic Hann window: w[n] = sin²(π n / N).
/// Equivalent to 1/2 - 1/2 cos(2π n / N); the sin² form measured ~9% faster
/// than the cos form on aarch64-macos, Zig 0.16, -OReleaseFast (N=256..16384).
fn hannWindow(out: []f32) void {
    const n: f32 = @floatFromInt(out.len);
    for (out, 0..) |*w, i| {
        const s = @sin(std.math.pi * @as(f32, @floatFromInt(i)) / n);
        w.* = s * s;
    }
}

/// In-place DFT. 
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

/// In-place radix-2 Cooley–Tukey FFT (recursive, scalar)
fn fft(re: []f32, im: []f32, scratch_re: []f32, scratch_im: []f32) void {
    const n = re.len;
    if (n <= 1) return;

    const half = n / 2;


    for (0..half) |i| {
        scratch_re[i] = re[2 * i];
        scratch_im[i] = im[2 * i];
        scratch_re[half + i] = re[2 * i + 1];
        scratch_im[half + i] = im[2 * i + 1];
    }
    @memcpy(re[0..half], scratch_re[0..half]);
    @memcpy(im[0..half], scratch_im[0..half]);
    @memcpy(re[half..n], scratch_re[half..n]);
    @memcpy(im[half..n], scratch_im[half..n]);

    fft(re[0..half], im[0..half], scratch_re[0..half], scratch_im[0..half]);
    fft(re[half..n], im[half..n], scratch_re[half..n], scratch_im[half..n]);

    // X[k] = E[k] + W^k O[k],  X[k + N/2] = E[k] - W^k O[k]
    const n_f: f32 = @floatFromInt(n);
    for (0..half) |k| {
        const angle = -2.0 * std.math.pi * @as(f32, @floatFromInt(k)) / n_f;
        const wr = @cos(angle);
        const wi = @sin(angle);

        const er = re[k];
        const ei = im[k];
        const or_ = re[half + k];
        const oi = im[half + k];

        const tr = or_ * wr - oi * wi;
        const ti = or_ * wi + oi * wr;

        scratch_re[k] = er + tr;
        scratch_im[k] = ei + ti;
        scratch_re[half + k] = er - tr;
        scratch_im[half + k] = ei - ti;
    }
    @memcpy(re[0..n], scratch_re[0..n]);
    @memcpy(im[0..n], scratch_im[0..n]);
}

test "hannWindow endpoints" {
    var w: [1024]f32 = undefined;
    hannWindow(&w);
    try std.testing.expectEqual(@as(f32, 0), w[0]);
    try std.testing.expect(@abs(w[512] - 1.0) < 1e-6);
}

test "fft matches dft" {
    const sizes = [_]usize{ 4, 8, 16, 32 };
    for (sizes) |n| {
        const re_fft = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(re_fft);
        const im_fft = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(im_fft);
        const re_dft = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(re_dft);
        const im_dft = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(im_dft);
        const scratch_re = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(scratch_re);
        const scratch_im = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(scratch_im);

        // Deterministic real input (imag = 0).
        for (re_fft, re_dft, 0..) |*a, *b, i| {
            const x = @sin(0.7 * @as(f32, @floatFromInt(i)) + 0.3);
            a.* = x;
            b.* = x;
        }
        @memset(im_fft, 0);
        @memset(im_dft, 0);

        fft(re_fft, im_fft, scratch_re, scratch_im);
        dft(re_dft, im_dft, scratch_re, scratch_im);

        var max_abs: f32 = 0;
        for (re_fft, im_fft, re_dft, im_dft) |fr, fi, dr, di| {
            max_abs = @max(max_abs, @abs(fr - dr));
            max_abs = @max(max_abs, @abs(fi - di));
        }
        try std.testing.expect(max_abs < 1e-4);
    }
}
