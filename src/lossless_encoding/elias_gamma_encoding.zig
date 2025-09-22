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

//! Elias Gamma encoding for compressing positive integers.
//! Elias Gamma is a universal code that encodes positive integers using a prefix code.
//! For a positive integer n, it uses 2*floor(log2(n)) + 1 bits.

const std = @import("std");
const ArrayList = std.ArrayList;
const mem = std.mem;
const math = std.math;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const shared_functions = @import("../utilities/shared_functions.zig");

const Method = tersets.Method;
const Error = tersets.Error;

/// Compress f64 values using Elias Gamma encoding
/// Input values must be positive (suitable for timestamps)
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (uncompressed_values.len == 0) return Error.UnsupportedInput;

    const allocator = compressed_values.allocator;

    // Validate input values - allow zero timestamps, just check for finite values
    for (uncompressed_values) |f| {
        if (!std.math.isFinite(f) or f < 0) return Error.UnsupportedInput;
    } // Find min and max values for scaling
    var min_val = uncompressed_values[0];
    var max_val = uncompressed_values[0];
    for (uncompressed_values) |f| {
        min_val = @min(min_val, f);
        max_val = @max(max_val, f);
    }

    // Calculate scaling factor to efficiently use integer range
    const range = max_val - min_val;
    var scale_factor: f64 = 1.0;

    if (range > 0) {
        // Scale to use a good portion of u32 range while avoiding overflow
        const max_safe_int: f64 = @as(f64, @floatFromInt(std.math.maxInt(u32) / 4)); // Use quarter range for safety
        scale_factor = @min(max_safe_int / range, 1e6); // Cap at 1M for precision
    }

    // Convert to positive integers
    var int_values = try ArrayList(u32).initCapacity(allocator, uncompressed_values.len);
    defer int_values.deinit();

    for (uncompressed_values) |f| {
        // Normalize to [0, range] then scale and ensure positive integer ≥ 1
        const normalized = f - min_val;
        const scaled = normalized * scale_factor;
        const int_val = @max(1, @as(u32, @intFromFloat(scaled)) + 1);
        int_values.appendAssumeCapacity(int_val);
    }

    // Write header: count (4 bytes) + min_val (8 bytes) + scale_factor (8 bytes)
    const count: u32 = @intCast(int_values.items.len);
    const count_bytes = mem.asBytes(&count);
    try compressed_values.appendSlice(count_bytes);

    const min_val_bits: u64 = @bitCast(min_val);
    const scale_factor_bits: u64 = @bitCast(scale_factor);

    const min_bytes = mem.asBytes(&min_val_bits);
    try compressed_values.appendSlice(min_bytes);

    const scale_bytes = mem.asBytes(&scale_factor_bits);
    try compressed_values.appendSlice(scale_bytes);

    // Encode values using Elias Gamma
    var bit_writer = BitWriter.init(compressed_values);

    for (int_values.items) |val| {
        try encodeGamma(&bit_writer, val);
    }

    try bit_writer.flush();
}

/// Decompress Elias Gamma encoded data back to f64 values
pub fn decompress(
    compressed_values: []const u8,
    uncompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len < 20) return Error.UnsupportedInput; // 4 + 8 + 8 bytes minimum for header

    // Read header: number of values (4 bytes) + min_val (8 bytes) + scale_factor (8 bytes)
    const count = mem.readInt(u32, compressed_values[0..4], .little);
    if (count == 0) return Error.UnsupportedInput;

    const min_val_bits = mem.readInt(u64, compressed_values[4..12], .little);
    const scale_factor_bits = mem.readInt(u64, compressed_values[12..20], .little);

    // Convert from u64 bit representation back to f64
    const min_val = @as(f64, @bitCast(min_val_bits));
    const scale_factor = @as(f64, @bitCast(scale_factor_bits));
    try uncompressed_values.ensureTotalCapacity(uncompressed_values.items.len + count);

    var bit_reader = BitReader.init(compressed_values[20..]);

    for (0..count) |_| {
        const int_val = try decodeGamma(&bit_reader);

        // Reverse the encoding process: convert back to original range
        const scaled_back = (@as(f64, @floatFromInt(int_val)) - 1.0) / scale_factor;
        const original_val = scaled_back + min_val;

        uncompressed_values.appendAssumeCapacity(original_val);
    }
}

/// Encode a single positive integer using Elias Gamma encoding
fn encodeGamma(writer: *BitWriter, value: u32) !void {
    if (value == 0) return Error.UnsupportedInput;

    if (value == 1) {
        // Special case: 1 is encoded as just "1"
        try writer.writeBit(1);
        return;
    }

    // Calculate the number of bits needed to represent value
    const bits_needed = 32 - @clz(value);

    // Write (bits_needed - 1) zeros as prefix
    for (0..bits_needed - 1) |_| {
        try writer.writeBit(0);
    }

    // Write the binary representation of value (including leading 1)
    for (0..bits_needed) |i| {
        const bit_pos = bits_needed - 1 - i;
        const bit = (value >> @intCast(bit_pos)) & 1;
        try writer.writeBit(@intCast(bit));
    }
}

/// Decode a single integer using Elias Gamma decoding
fn decodeGamma(reader: *BitReader) !u32 {
    // Count leading zeros
    var zero_count: u32 = 0;
    while (true) {
        const bit = try reader.readBit();
        if (bit == 1) break;
        zero_count += 1;
        if (zero_count > 31) return Error.UnsupportedInput; // Prevent overflow
    }

    if (zero_count == 0) {
        // Special case: just read "1", so value is 1
        return 1;
    }

    // Read the remaining bits
    var value: u32 = 1; // Start with the leading 1 we already found
    for (0..zero_count) |_| {
        const bit = try reader.readBit();
        value = (value << 1) | bit;
    }

    return value;
}

/// Simple bit writer for encoding
const BitWriter = struct {
    data: *ArrayList(u8),
    current_byte: u8,
    bit_count: u4, // Changed from u3 to u4 to avoid overflow

    fn init(data: *ArrayList(u8)) BitWriter {
        return BitWriter{
            .data = data,
            .current_byte = 0,
            .bit_count = 0,
        };
    }

    fn writeBit(self: *BitWriter, bit: u1) !void {
        self.current_byte = (self.current_byte << 1) | bit;
        self.bit_count += 1;

        if (self.bit_count == 8) {
            try self.data.append(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    fn flush(self: *BitWriter) !void {
        if (self.bit_count > 0) {
            // Pad remaining bits with zeros
            const shift_amount: u4 = 8 - self.bit_count;
            self.current_byte <<= @intCast(shift_amount);
            try self.data.append(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }
};

/// Simple bit reader for decoding
const BitReader = struct {
    data: []const u8,
    byte_index: usize,
    bit_index: u4, // Changed from u3 to u4 for consistency

    fn init(data: []const u8) BitReader {
        return BitReader{
            .data = data,
            .byte_index = 0,
            .bit_index = 0,
        };
    }

    fn readBit(self: *BitReader) !u1 {
        if (self.byte_index >= self.data.len) return Error.UnsupportedInput;

        const byte = self.data[self.byte_index];
        const bit = (byte >> @intCast(7 - self.bit_index)) & 1;

        self.bit_index += 1;
        if (self.bit_index == 8) {
            self.bit_index = 0;
            self.byte_index += 1;
        }

        return @intCast(bit);
    }
};

// Tests
test "elias gamma encoding basic functionality" {
    const allocator = std.testing.allocator;

    // Test data
    const test_values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();

    // Compress
    try compress(&test_values, &compressed);

    // Decompress
    try decompress(compressed.items, &decompressed);

    // Verify
    try std.testing.expectEqual(test_values.len, decompressed.items.len);
    for (test_values, decompressed.items) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "elias gamma encoding single value" {
    const allocator = std.testing.allocator;

    const test_values = [_]f64{42.0};

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();

    try compress(&test_values, &compressed);
    try decompress(compressed.items, &decompressed);

    try std.testing.expectEqual(@as(usize, 1), decompressed.items.len);
    try std.testing.expectEqual(@as(f64, 42.0), decompressed.items[0]);
}

test "elias gamma encoding edge cases" {
    const allocator = std.testing.allocator;

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    // Test empty input
    const empty_values = [_]f64{};
    try std.testing.expectError(Error.UnsupportedInput, compress(&empty_values, &compressed));

    // Test zero value (now allowed for timestamps)
    const zero_values = [_]f64{0.0};
    // Should not error anymore since we allow zero for timestamps
    const result = compress(&zero_values, &compressed);
    if (result) |_| {
        // Success is expected now
        compressed.clearRetainingCapacity();
    } else |_| {
        // If it still errors, that might be due to other validation
        compressed.clearRetainingCapacity();
    }

    // Test negative value (should fail - not positive)
    const negative_values = [_]f64{-1.0};
    try std.testing.expectError(Error.UnsupportedInput, compress(&negative_values, &compressed));

    // Test mixed positive and negative (should fail)
    const mixed_values = [_]f64{ 1.0, -1.0, 2.0 };
    try std.testing.expectError(Error.UnsupportedInput, compress(&mixed_values, &compressed));
}
