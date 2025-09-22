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

//! Implementation of "Delta Encoding" for compressing and decompressing integer time stamps.

const std = @import("std");
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");

const Method = tersets.Method;
const Error = tersets.Error;

pub fn compress(uncompressed: []const f64, compressed: *ArrayList(f64)) !void {
    if (uncompressed.len == 0) return Error.UnsupportedInput;

    var prev: f64 = uncompressed[0];
    try compressed.append(prev); // store first value since there's no delta for it

    for (uncompressed[1..]) |val| {
        const delta = val - prev;
        try compressed.append(delta);
        prev = val;
    }
}

pub fn decompress(compressed: []const f64, decompressed: *ArrayList(f64)) !void {
    if (compressed.len == 0) return Error.UnsupportedInput;

    var prev = compressed[0];
    try decompressed.append(prev);

    for (compressed[1..]) |delta| {
        const value = prev + delta;
        try decompressed.append(value);
        prev = value;
    }
}

test "delta encoding roundtrip on regular increments" {
    const allocator = std.testing.allocator;

    // 0, 1, 2, 3, 4, ... should compress to 0, 1, 1, 1, 1, ...
    var src = ArrayList(f64).init(allocator);
    defer src.deinit();
    for (0..100) |i| try src.append(@floatFromInt(i));

    var buf = ArrayList(f64).init(allocator);
    defer buf.deinit();

    try compress(src.items, &buf);

    var out = ArrayList(f64).init(allocator);
    defer out.deinit();

    try decompress(buf.items, &out);
    try std.testing.expectEqualSlices(f64, src.items, out.items);
}
