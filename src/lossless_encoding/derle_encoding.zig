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

//! Implementation of "Delta + RLE (DERLE)" encoding for compressing and decompressing integer time stamps.

const std = @import("std");
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const testing = std.testing;
const Error = tersets.Error;
const Method = tersets.Method;

const rle = @import("./run_length_encoding.zig");
const delta = @import("./delta_encoding.zig");

pub fn compress(uncompressed: []const f64, compressed: *ArrayList(u8)) !void {
    var delta_buf = ArrayList(f64).init(std.heap.page_allocator);
    defer delta_buf.deinit();

    try delta.compress(uncompressed, &delta_buf);
    try rle.compress(delta_buf.items, compressed);
}

pub fn decompress(compressed: []const u8, decompressed: *ArrayList(f64)) !void {
    var rle_buf = ArrayList(f64).init(std.heap.page_allocator);
    defer rle_buf.deinit();

    try rle.decompress(compressed, &rle_buf);
    try delta.decompress(rle_buf.items, decompressed);
}

test "delta+RLE roundtrip on regular increments" {
    const allocator = testing.allocator;

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    for (0..100) |i| try data.append(@floatFromInt(i));

    try tester.testCompressAndDecompress(
        allocator,
        data.items,
        Method.DERLEEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}

test "delta+RLE works with repeated timestamps" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1, 2, 3, 5, 10, 12, 14, 15, 16, 25, 30 };

    try tester.testCompressAndDecompress(
        allocator,
        &data,
        Method.DERLEEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}
