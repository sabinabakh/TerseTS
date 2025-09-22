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

//! Implementation of "Variable-Length Quantity (VLQ)" encoding for compressing and decompressing integer time stamps.

const std = @import("std");
const ArrayList = std.ArrayList;
const mem = std.mem;

const tester = @import("../tester.zig");
const tersets = @import("../tersets.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const testing = std.testing;

const Error = tersets.Error;
const Method = tersets.Method;

/// Compress f64 values (assumed non-negative integers) into VLQ bytes
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Always write a header with the count (even for empty input)
    try shared_functions.appendValue(usize, uncompressed_values.len, compressed_values);

    if (uncompressed_values.len == 0) return;

    for (uncompressed_values) |f_value| {
        if (f_value < 0) return Error.UnsupportedInput;

        var value: u64 = @intFromFloat(f_value);

        // Emit 7-bit groups, MSB=1 indicates continuation.
        while (value >= 0x80) {
            const byte: u8 = @intCast((value & 0x7F) | 0x80);
            try compressed_values.append(byte);
            value >>= 7;
        }
        // Final group with MSB=0.
        try compressed_values.append(@intCast(value & 0x7F));
    }
}

/// Decompress VLQ bytes back to f64 values.
/// Expects: [count: usize][payload of exactly `count` VLQ numbers].
/// Count represents how many f64 values to decode.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    const w = @sizeOf(usize);
    if (compressed_values.len < w) return Error.UnsupportedInput;

    // Read the count header.
    const count = mem.bytesToValue(usize, compressed_values[0..w]);
    const vlq_data = compressed_values[w..];

    try decompressed_values.ensureTotalCapacity(decompressed_values.items.len + count);

    var decoded_count: usize = 0;
    var acc: u64 = 0;
    var shift_bits: usize = 0;

    // Decode values one by one.
    for (vlq_data) |byte| {
        const data_bits: u64 = @intCast(byte & 0x7F);

        if (shift_bits >= @bitSizeOf(u64)) return Error.UnsupportedInput;

        acc |= (data_bits << @intCast(shift_bits));

        if ((byte & 0x80) == 0) {
            // End of this value.
            decompressed_values.appendAssumeCapacity(@floatFromInt(acc));
            decoded_count += 1;

            // Reset for next value.
            acc = 0;
            shift_bits = 0;

            if (decoded_count == count) break;
        } else {
            // Continue with next 7-bit chunk.
            shift_bits += 7;
        }
    }

    // Must have decoded exactly `count` values
    if (decoded_count != count) return Error.UnsupportedInput;
}

test "vlq f64 roundtrip" {
    const allocator = testing.allocator;
    const data = [_]f64{ 0.0, 1.0, 127.0, 128.0, 255.0, 256.0, 16383.0, 16384.0 };

    try tester.testCompressAndDecompress(
        allocator,
        &data,
        Method.VariableLengthQuantityEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}

test "vlq large values" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 1_000_000.0, 4_294_967_295.0, 9_223_372_036_854_775_807.0 };

    try tester.testCompressAndDecompress(
        allocator,
        &data,
        Method.VariableLengthQuantityEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}

test "vlq empty input" {
    const allocator = testing.allocator;

    var buf = ArrayList(u8).init(allocator);
    defer buf.deinit();

    // Empty input writes just a header with count=0.
    try compress(&[_]f64{}, &buf);

    var out = ArrayList(f64).init(allocator);
    defer out.deinit();

    try decompress(buf.items, &out);
    try testing.expectEqual(@as(usize, 0), out.items.len);
}

test "vlq invalid negative value" {
    const allocator = testing.allocator;

    var buf = ArrayList(u8).init(allocator);
    defer buf.deinit();

    // Should reject negative values.
    try testing.expectError(Error.UnsupportedInput, compress(&[_]f64{ 1.0, -5.0, 10.0 }, &buf));
}
