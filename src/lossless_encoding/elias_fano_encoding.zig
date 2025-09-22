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

//! Elias-Fano encoding for compressing and decompressing monotonically increasing integer time stamps.

const std = @import("std");
const ArrayList = std.ArrayList;
const mem = std.mem;
const math = std.math;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const shared_functions = @import("../utilities/shared_functions.zig");

const Method = tersets.Method;
const Error = tersets.Error;

/// Compress monotonically increasing f64 values using Elias-Fano encoding
/// Input values should represent positive integers
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (uncompressed_values.len == 0) return Error.UnsupportedInput;

    const n = uncompressed_values.len;
    const allocator = compressed_values.allocator;
    var int_values = try ArrayList(u64).initCapacity(allocator, n);
    defer int_values.deinit();

    var prev: u64 = 0;
    for (uncompressed_values, 0..) |f, i| {
        const val: u64 = @intFromFloat(f);
        if (i > 0 and val <= prev) return Error.UnsupportedInput;
        int_values.appendAssumeCapacity(val);
        prev = val;
    }

    const universe_size: usize = @intCast(int_values.items[n - 1] + 1);
    const l: u6 = if (universe_size <= n) 0 else @intCast(math.log2_int(usize, universe_size / n));

    const lower_mask: u64 = if (l == 0) 0 else (@as(u64, 1) << l) - 1;
    const lower_bits = n * @as(usize, l);
    const lower_bytes = (lower_bits + 7) / 8;
    const upper_bits = n + (@as(usize, @intCast(int_values.items[n - 1])) >> l) + 1;
    const upper_bytes = (upper_bits + 7) / 8;

    try shared_functions.appendValue(usize, @as(usize, l), compressed_values);
    try shared_functions.appendValue(usize, n, compressed_values);
    try shared_functions.appendValue(usize, universe_size, compressed_values);

    const data_start = compressed_values.items.len;
    const total_bytes = lower_bytes + upper_bytes;
    try compressed_values.ensureTotalCapacity(data_start + total_bytes);
    try compressed_values.appendNTimes(0, total_bytes);

    // Pack lower bits
    if (l > 0) {
        var bit_pos: usize = 0;
        for (int_values.items) |val| {
            const low_bits = val & lower_mask;
            packBits(compressed_values.items[data_start..], bit_pos, low_bits, l);
            bit_pos += l;
        }
    }

    // Pack upper bits with unary encoding
    const upper_start = data_start + lower_bytes;
    for (int_values.items, 0..) |val, i| {
        const high_bits = val >> l;
        const bit_index = high_bits + i;
        const byte_idx = bit_index / 8;
        const bit_in_byte: u3 = @intCast(bit_index % 8);
        compressed_values.items[upper_start + byte_idx] |= (@as(u8, 1) << bit_in_byte);
    }
}

/// Pack bits into byte array at specified bit position
fn packBits(data: []u8, bit_offset: usize, value: u64, width: u6) void {
    if (width == 0) return;

    var remaining_bits = width;
    var remaining_value = value;
    var current_bit = bit_offset;

    while (remaining_bits > 0) {
        const byte_idx = current_bit / 8;
        const bit_in_byte: u3 = @intCast(current_bit % 8);
        const bits_available = 8 - @as(usize, bit_in_byte);
        const bits_to_write = @min(bits_available, remaining_bits);

        const mask: u8 = @intCast((@as(usize, 1) << @intCast(bits_to_write)) - 1);
        const chunk: u8 = @intCast(remaining_value & mask);

        data[byte_idx] |= chunk << bit_in_byte;

        remaining_value >>= @intCast(bits_to_write);
        remaining_bits -= @intCast(bits_to_write);
        current_bit += bits_to_write;
    }
}

/// Extract bits from byte array at specified bit position
fn extractBits(data: []const u8, bit_offset: usize, width: u6) u64 {
    if (width == 0) return 0;

    var result: u64 = 0;
    var remaining_bits = width;
    var current_bit = bit_offset;
    var result_shift: u6 = 0;

    while (remaining_bits > 0) {
        const byte_idx = current_bit / 8;
        if (byte_idx >= data.len) break;

        const bit_in_byte: u3 = @intCast(current_bit % 8);
        const bits_available = 8 - @as(usize, bit_in_byte);
        const bits_to_read = @min(bits_available, remaining_bits);

        const mask: u8 = @intCast((@as(usize, 1) << @intCast(bits_to_read)) - 1);
        const chunk = (data[byte_idx] >> bit_in_byte) & mask;

        result |= (@as(u64, chunk) << result_shift);

        remaining_bits -= @intCast(bits_to_read);
        current_bit += bits_to_read;
        result_shift += @intCast(bits_to_read);
    }

    return result;
}

/// Decompress Elias-Fano encoded data back to f64 values.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len < 3 * @sizeOf(usize)) {
        return Error.UnsupportedInput;
    }

    var cursor: usize = 0;

    // Read header
    const l_usize = mem.bytesToValue(usize, compressed_values[cursor .. cursor + @sizeOf(usize)]);
    const l: u6 = @intCast(l_usize);
    cursor += @sizeOf(usize);

    const n = mem.bytesToValue(usize, compressed_values[cursor .. cursor + @sizeOf(usize)]);
    cursor += @sizeOf(usize);

    const universe_size = mem.bytesToValue(usize, compressed_values[cursor .. cursor + @sizeOf(usize)]);
    cursor += @sizeOf(usize);

    if (n == 0) return Error.UnsupportedInput;

    // Calculate section sizes
    const lower_bits = n * @as(usize, l);
    const lower_bytes = (lower_bits + 7) / 8;
    const upper_bits = n + ((universe_size - 1) >> l) + 1;
    const upper_bytes = (upper_bits + 7) / 8;

    if (cursor + lower_bytes + upper_bytes > compressed_values.len) {
        return Error.UnsupportedInput;
    }

    const lower_data = compressed_values[cursor .. cursor + lower_bytes];
    const upper_data = compressed_values[cursor + lower_bytes .. cursor + lower_bytes + upper_bytes];

    // Decode values
    try decompressed_values.ensureTotalCapacity(n);

    var decoded_count: usize = 0;
    var bit_idx: usize = 0;

    while (decoded_count < n and bit_idx < upper_bits) {
        const byte_idx = bit_idx / 8;
        if (byte_idx >= upper_data.len) break;

        const bit_in_byte: u3 = @intCast(bit_idx % 8);
        const is_set = (upper_data[byte_idx] >> bit_in_byte) & 1;

        if (is_set == 1) {
            const high_bits = bit_idx - decoded_count;
            const low_bits = if (l > 0) extractBits(lower_data, decoded_count * l, l) else 0;
            const value = (high_bits << l) | low_bits;

            decompressed_values.appendAssumeCapacity(@as(f64, @floatFromInt(value)));
            decoded_count += 1;
        }

        bit_idx += 1;
    }

    if (decoded_count != n) return Error.UnsupportedInput;
}

test "elias-fano f64 roundtrip" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 2.0, 5.0, 7.0, 8.0, 13.0 };

    try tester.testCompressAndDecompress(
        allocator,
        &data,
        Method.EliasFanoEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}

test "elias-fano large sequence" {
    const allocator = std.testing.allocator;

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();

    // Create integer-like values
    var val: f64 = 1.0;
    for (0..100) |_| {
        try data.append(val);
        val = @floor(val + 1.0 + (val / 10.0));
    }

    try tester.testCompressAndDecompress(
        allocator,
        data.items,
        Method.EliasFanoEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}
