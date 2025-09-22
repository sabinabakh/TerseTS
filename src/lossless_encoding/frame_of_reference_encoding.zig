// Copyright 2025 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Implementation of "Frame-of-Reference (FOR)" encoding for compressing and decompressing integer time stamps.
// Copyright 2025 TerseTS Contributors
// Licensed under the Apache License, Version 2.0

//! Frame-of-Reference (FOR) encoding for integer-valued timestamps (f64 API).
//! Flat buffer layout (native endian): [min: usize][offset_0: usize][offset_1: usize]...

const std = @import("std");
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const testing = std.testing;

const Method = tersets.Method;
const Error = tersets.Error;

/// Compress: writes [min][offset_0..] into `compressed_values`
/// Requires inputs to be finite, non-negative, integer-valued f64
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (uncompressed_values.len == 0) return error.UnsupportedInput;

    // Validate & find min in one pass
    var min_u: usize = undefined;
    var have_min = false;

    for (uncompressed_values) |f| {
        if (!std.math.isFinite(f) or f < 0 or f != @floor(f))
            return error.UnsupportedInput;

        const u: usize = @intFromFloat(f);
        if (!have_min or u < min_u) {
            min_u = u;
            have_min = true;
        }
    }
    // Write min
    try shared_functions.appendValue(usize, min_u, compressed_values);

    // Write offsets = value - min as usize
    for (uncompressed_values) |f| {
        const u: usize = @intFromFloat(f);
        const off: usize = u - min_u;
        try shared_functions.appendValue(usize, off, compressed_values);
    }
}

/// Decompress from back to original values
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    const W: usize = @sizeOf(usize);
    if (compressed_values.len < W) return error.UnsupportedInput;

    var cur: usize = 0;

    // read min
    const min_u: usize = std.mem.bytesToValue(usize, compressed_values[cur .. cur + W]);
    cur += W;

    // remaining must be whole number of usize offsets
    const rem = compressed_values.len - cur;
    if (rem % W != 0) return error.UnsupportedInput;

    const offsets = std.mem.bytesAsSlice(usize, compressed_values[cur..]);
    try decompressed_values.ensureTotalCapacity(decompressed_values.items.len + offsets.len);

    // reconstruct values
    for (offsets) |off| {
        const u: usize = min_u + off;
        try decompressed_values.append(@as(f64, @floatFromInt(u)));
    }
}

test "FOR roundtrip: mixed gaps and repeats" {
    const alloc = testing.allocator;
    const data = [_]f64{ 1000, 1000, 1003, 1003, 1003, 1010, 1011 };

    var buf = ArrayList(u8).init(alloc);
    defer buf.deinit();

    try compress(&data, &buf);

    var out = ArrayList(f64).init(alloc);
    defer out.deinit();

    try decompress(buf.items, &out);
    try testing.expectEqualSlices(f64, &data, out.items);
}

test "FOR: stress multiple back-to-back encode/decode" {
    const allocator = testing.allocator;
    const data = [_]f64{ 42, 43, 43, 44, 1000, 1000, 1001 };

    for (0..20) |_| {
        var buf = ArrayList(u8).init(allocator);
        defer buf.deinit();
        var out = ArrayList(f64).init(allocator);
        defer out.deinit();

        try compress(&data, &buf);
        try decompress(buf.items, &out);
        try testing.expectEqualSlices(f64, &data, out.items);
    }

    try tester.testCompressAndDecompress(
        allocator,
        &data,
        Method.FrameOfReferenceEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}
