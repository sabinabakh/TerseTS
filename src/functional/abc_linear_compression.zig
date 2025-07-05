// Copyright 2024 TerseTS Contributors
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

//! Implementation of "ABCLinearApproximation" algorithm from the paper:
//! "Approximations of One-Dimensional Digital Signals Under the L^\infinity Norm.
//! Dalai, Marco, and Riccardo Leonardi.
//! IEEE Transactions on Signal Processing, 54, 8, 3111-3124.
//! https://doi.org/10.1109/TSP.2006.875394.
//! The name "ABCLinearApproximation" reflects the core geometric concept of the algorithm:
//! the optimal segment is determined by evaluating triplets of points A, B, and C.

const std = @import("std");
const mem = std.mem;
const time = std.time;
const rand = std.Random;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const testing = std.testing;
const Error = tersets.Error;
const Method = tersets.Method;

const shared_structs = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;
const Segment = shared_structs.Segment;

const ConvexHull = @import("../utilities/convex_hull.zig").ConvexHull;

/// Compresses `uncompressed_values` using the "ABCLinearApproximation" algorithm under the
/// L-infinity norm. The function writes the result to `compressed_values`. The `allocator`
/// is used to allocate memory for the convex hull. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (uncompressed_values.len < 2) return Error.UnsupportedInput;
    if (error_bound < 0.0) return Error.UnsupportedErrorBound;

    // The algorithm uses a convex hull to store a reduce set of significant points.
    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    var current_segment_start: usize = 0;
    while (current_segment_start < uncompressed_values.len - 1) {
        var last_valid_line: ?LinearFunction = null;

        // Insert the first point in the convex hull.
        try convex_hull.add(.{ .time = current_segment_start, .value = uncompressed_values[current_segment_start] });

        // Create a index to grow iterate over the time series from the current segment start.
        var index_over_segment = current_segment_start + 1;
        var last_valid_end = index_over_segment;

        // The 'index_over_segment' iterates through the points increasing the size of the current segment.
        while (index_over_segment < uncompressed_values.len) : (index_over_segment += 1) {
            // Section III-A, Step 1: Computing the Convex Hull.
            // Add next point to convex hull for current segment.
            try convex_hull.add(.{ .time = index_over_segment, .value = uncompressed_values[index_over_segment] });

            // Section III-A, Step 2-3: Find A, B, C and compute the solution line.
            // Try to compute the best fitting line using current convex hull points.
            const line = try findABCOptimalSegment(&convex_hull, allocator);

            // Compute maximum error over current segment.
            const max_error = try convex_hull.computeMaxError(line);

            if (max_error <= error_bound) {
                // If all points are within error_bound, the segment is still valid.
                last_valid_end = index_over_segment;
                last_valid_line = line;
            } else {
                // The error bound exceeded. Stop extending segment.
                break;
            }
        }

        // Store segment information (end_index, slope, intercept).
        if (last_valid_line) |valid_line| {
            try appendValue(usize, last_valid_end, compressed_values);
            // If there are only two points in the segment, store then directly to avoid numerical issues.
            if (current_segment_start + 1 == last_valid_end) {
                try appendValue(f64, uncompressed_values[current_segment_start], compressed_values);
                try appendValue(f64, uncompressed_values[last_valid_end], compressed_values);
            } else {
                try appendValue(f64, @floatCast(valid_line.slope), compressed_values);
                try appendValue(f64, @floatCast(valid_line.intercept), compressed_values);
            }
        } else {
            // If the the last valid line is not valid, then store the uncompressed values directly.
            try appendValue(usize, last_valid_end, compressed_values);
            try appendValue(f64, uncompressed_values[current_segment_start], compressed_values);
            try appendValue(f64, uncompressed_values[last_valid_end], compressed_values);
        }

        // Start next segment after last_valid_end.
        current_segment_start = last_valid_end + 1;

        // Clean convex hull for the new segment.
        convex_hull.clean();
    }
    // Store the last point if left.
    if (current_segment_start == uncompressed_values.len - 1) {
        const value = uncompressed_values[current_segment_start];
        const slope: f64 = 0.0;
        const intercept = value;

        try appendValue(usize, current_segment_start, compressed_values);
        try appendValue(f64, slope, compressed_values);
        try appendValue(f64, intercept, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "ABCLinearApproximation". The algorithm writes the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (end_index, slope, intercept).
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const fields = mem.bytesAsSlice(f64, compressed_values);
    var field_index: usize = 0;
    var segment_start: usize = 0;

    while (field_index + 2 < fields.len) : (field_index += 3) {
        const segment_end = @as(usize, @bitCast(fields[field_index]));
        const slope = fields[field_index + 1];
        const intercept = fields[field_index + 2];
        if (segment_start + 1 != segment_end) {
            for (segment_start..segment_end + 1) |t| {
                const x = @as(f64, @floatFromInt(t));
                const y = slope * x + intercept;
                try decompressed_values.append(y);
            }
        } else {
            try decompressed_values.append(slope);
            try decompressed_values.append(intercept);
        }
        segment_start = segment_end + 1;
    }
}

// Find the optimal segment using ABC structure from the `convex_hull`. Specifically, the
/// A and B points form a segment (AB) on the lower or upper hull. The C point is found on
/// the opposite hull with maximum deviation, projected vertically into the segment AB.
/// The `allocator` is used to create a AutoHashMap to control which point has been visited.
pub fn findABCOptimalSegment(convex_hull: *ConvexHull, allocator: mem.Allocator) Error!LinearFunction {
    const len = convex_hull.len();

    // Initialize first side l1 = (p0, p1).
    var point_a_index: usize = 0;
    var point_c_index: usize = 0;

    var finished = false;
    var visited = std.AutoHashMap(usize, void).init(allocator);
    defer visited.deinit();

    while (!finished) {
        if (point_a_index + 1 >= len) break; // No more sides.

        const point_a = convex_hull.at(point_a_index);
        const point_b = convex_hull.at(point_a_index + 1);

        // Find pivot point C.
        const pivot_point_c_idx = findPivotC(convex_hull, point_a_index);

        point_c_index = pivot_point_c_idx orelse {
            // No valid C found, use AB as the line (in case of just two points in convex hull).
            // Need at least three points to define a valid segment using the ABC method.
            const delta_time = @as(f64, @floatFromInt(point_b.time - point_a.time));

            const slope = (point_b.value - point_a.value) / delta_time;
            const intercept = point_a.value - slope * @as(f64, @floatFromInt(point_a.time));

            return LinearFunction{ .slope = slope, .intercept = intercept };
        };

        const point_c = convex_hull.at(point_c_index);

        if (visited.contains(point_a_index)) {
            break;
        }
        try visited.put(point_a_index, {}); // Mark as visited.

        // Determine relative x-position of pivot to l_i.
        if (point_c.time > point_b.time) {
            // x-external to right -> advance to next side.
            point_a_index += 1;
        } else if (point_c.time < point_a.time) {
            // x-external to left -> adjust upper hull between v(l_{i-1}) and v(l_i).
            if (point_a_index == 0) {
                // If we are already at the first side, we cannot move left.
                // In this case, just accept the current segment as optimal.
                finished = true;
            } else {
                // x-external to left -> adjust search: go back one side.
                point_a_index -= 1;
            }
        } else {
            // x-internal -> optimal segment found.
            finished = true;
        }
    }

    // Once finished, line from A to B is the optimal approximation.
    const start = convex_hull.at(point_a_index);
    const end = convex_hull.at(point_a_index + 1);
    const point_c = convex_hull.at(point_c_index);

    const delta_time = @as(f64, @floatFromInt(end.time - start.time));

    const slope = (end.value - start.value) / delta_time;

    const pred = slope * (@as(f64, @floatFromInt(point_c.time - start.time))) + start.value;
    const deviation = @abs(pred - point_c.value);

    const intercept = start.value - slope * @as(f64, @floatFromInt(start.time)) + deviation / 2;

    return LinearFunction{ .slope = slope, .intercept = intercept };
}

/// Find and return the pivot point C in the `convex_hull` based on the `point_a_index`.
fn findPivotC(convex_hull: *ConvexHull, point_a_index: usize) ?usize {
    const point_a = convex_hull.at(point_a_index);
    const point_b = convex_hull.at(point_a_index + 1);
    var max_dev: f64 = -1.0;
    var pivot_idx: ?usize = null;

    // Check all hull vertices for x-internal points.
    // Find v(l_i): the vertex that maximizes deviation from the side l_i.
    for (0..convex_hull.len()) |point_c_index| {
        const point_c = convex_hull.at(point_c_index);
        // Explicitly exclude point_a and B from being C.
        if (point_c_index != point_a_index and point_c_index != point_a_index + 1) {
            const dev = computeDeviation(point_a, point_b, point_c);
            if (dev > max_dev) {
                max_dev = dev;
                pivot_idx = point_c_index;
            }
        }
    }

    return pivot_idx;
}

/// Computes the vertical deviation of a given point `point_c` from the line segment defined by two
/// other points `point_a` and `point_b`. The deviation is calculated as the absolute difference
/// between the actual value of point_c and its projected value on the line segment.
fn computeDeviation(point_a: DiscretePoint, point_b: DiscretePoint, point_c: DiscretePoint) f64 {
    // Compute slope of side formed by `point_a` and `point_b`.
    const delta_time = @as(f64, @floatFromInt(point_b.time - point_a.time));
    const slope = (point_b.value - point_a.value) / delta_time;

    // Project point_c vertically onto the line defined by point_a and point_b. The expression
    // (time_point_c - time_point_a) may be negative if point_c is x-external
    // (i.e., point_c.time < point_a.time), which is valid and expected. We use floating-point
    // subtraction to avoid usize underflow.
    const time_point_a = @as(f64, @floatFromInt(point_a.time));
    const time_point_c = @as(f64, @floatFromInt(point_c.time));

    const pred = slope * (time_point_c - time_point_a) + point_a.value;

    // Deviation of point C from line defined by `slope` and `intercept`.
    return @abs(pred - point_c.value);
}

/// Helper to serialize the `value` into bytes and store it in `compressed`.
fn appendValue(comptime T: type, value: T, compressed: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try compressed.appendSlice(&bytes);
}

test "abc compressor can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateFiniteRandomValues,
        allocator,
        Method.ABCLinearApproximation,
        0,
        tersets.isWithinErrorBound,
    );
}

test "abc compressor identifies correct ABC points in the convex hull of a bigger size" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 5;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    try uncompressed_values.append(3);
    try uncompressed_values.append(2);
    try uncompressed_values.append(3.5);
    try uncompressed_values.append(5);
    try uncompressed_values.append(3);
    try uncompressed_values.append(4);
    try uncompressed_values.append(4);
    try uncompressed_values.append(3);
    try uncompressed_values.append(4.5);
    try uncompressed_values.append(3.5);
    try uncompressed_values.append(2.5);
    try uncompressed_values.append(2.5);
    try uncompressed_values.append(3.5);
    try uncompressed_values.append(2.5);
    try uncompressed_values.append(2.5);
    try uncompressed_values.append(2.5);
    try uncompressed_values.append(3);
    try uncompressed_values.append(3);
    try uncompressed_values.append(3);
    try uncompressed_values.append(3);
    try uncompressed_values.append(2.8);

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(
        uncompressed_values.items,
        &compressed_values,
        allocator,
        error_bound,
    );

    // Interpret the compressed bytes as values
    const fields = std.mem.bytesAsSlice(f64, compressed_values.items);
    const slope = fields[1];
    const intercept = fields[2];

    try testing.expect(@abs(slope - 0.036) <= 0.1);
    try testing.expect(@abs(intercept - 3.43) <= 0.1);

    // Ensure values are compressed in one single segment
    try std.testing.expectEqual(compressed_values.items.len / 8, 3);
}

test "abc compressor can compress and decompress random lines and error bound" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));
    for (0..max_lines) |_| {
        // Generate a random linear function and add it to the uncompressed values.
        try tester.generateRandomLinearFunction(&uncompressed_values, undefined);
    }

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.ABCLinearApproximation,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "abc compressor can compress and decompress odd size" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    // Add another element to make the uncompressed values of odd size.
    try uncompressed_values.append(tester.generateBoundedRandomValue(f64, 0, 1, undefined));

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.ABCLinearApproximation,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "abc compressor compresses and decompresses constant signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    const constant_value: f64 = tester.generateBoundedRandomValue(f64, 0, 1, undefined);

    const number_elements: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 100, 150, undefined)));

    for (0..number_elements) |_| {
        try uncompressed_values.append(constant_value);
    }

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.ABCLinearApproximation,
        error_bound,
        tersets.isWithinErrorBound,
    );
}
//
// test "abc compressor can handle wafer dataset from TSV file" {
//     const allocator = std.testing.allocator;
//
//     // Try to read the wafer data file
//     var wafer_data = readWaferTSV(allocator, "Wafer_TRAIN.tsv") catch |err| {
//         std.debug.print("Warning: Could not read Wafer_TRAIN.tsv: {}\n", .{err});
//         std.debug.print("Skipping wafer test. Make sure Wafer_TRAIN.tsv is in the working directory.\n", .{});
//         return;
//     };
//     defer wafer_data.deinit();
//
//     std.debug.print("Loaded {} data points from Wafer_TRAIN.tsv\n", .{wafer_data.items.len});
//
//     // Print all parsed values
//     // std.debug.print("\nAll parsed values:\n", .{});
//     // for (wafer_data.items, 0..) |value, i| {
//     //     if (i > 0 and i % 10 == 0) {
//     //         std.debug.print("\n", .{}); // New line every 10 values for readability
//     //     }
//     //     std.debug.print("{d:.3} ", .{value});
//     // }
//     // std.debug.print("\n\n", .{});
//
//     // Test with different error bounds
//     const error_bounds = [_]f32{ 1.0, 0.5, 0.1, 0.05 };
//
//     for (error_bounds) |error_bound| {
//         std.debug.print("\nTesting wafer data with error bound: {d}\n", .{error_bound});
//
//         var compressed_values = ArrayList(u8).init(allocator);
//         defer compressed_values.deinit();
//
//         // Add timeout/progress monitoring
//         const start_time = std.time.nanoTimestamp();
//         const max_duration_ns = 10_000_000_000; // 10 seconds
//
//         compress(
//             wafer_data.items,
//             &compressed_values,
//             allocator,
//             error_bound,
//         ) catch |err| {
//             const duration_ns = std.time.nanoTimestamp() - start_time;
//             const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
//
//             if (duration_ns > max_duration_ns) {
//                 std.debug.print("  x Compression timed out after {d:.1}ms\n", .{duration_ms});
//             } else {
//                 std.debug.print("  ✗ Compression failed: {} (after {d:.1}ms)\n", .{ err, duration_ms });
//             }
//             continue;
//         };
//
//         const end_time = std.time.nanoTimestamp();
//         const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
//
//         std.debug.print("  + Compression successful in {d:.1}ms\n", .{duration_ms});
//
//         // Calculate compression metrics
//         const original_bytes = wafer_data.items.len * 8; // f64 = 8 bytes
//         const compression_ratio = @as(f64, @floatFromInt(original_bytes)) / @as(f64, @floatFromInt(compressed_values.items.len));
//         std.debug.print("  Original: {} bytes, Compressed: {} bytes\n", .{ original_bytes, compressed_values.items.len });
//         std.debug.print("  Compression ratio: {d:.2}x\n", .{compression_ratio});
//
//         // Test decompression
//         var decompressed_values = ArrayList(f64).init(allocator);
//         defer decompressed_values.deinit();
//
//         decompress(compressed_values.items, &decompressed_values) catch |err| {
//             std.debug.print("  ✗ Decompression failed: {}\n", .{err});
//             continue;
//         };
//
//         std.debug.print("  + Decompression successful\n", .{});
//         std.debug.print("  Original size: {}, Decompressed size: {}\n", .{ wafer_data.items.len, decompressed_values.items.len });
//
//         // Verify error bound is satisfied
//         if (wafer_data.items.len == decompressed_values.items.len) {
//             var max_error: f64 = 0.0;
//             var error_count: usize = 0;
//
//             for (wafer_data.items, decompressed_values.items) |original, decompressed| {
//                 const abs_error = @abs(original - decompressed);
//                 if (abs_error > error_bound) {
//                     error_count += 1;
//                 }
//                 max_error = @max(max_error, abs_error);
//             }
//
//             std.debug.print("  Max error: {d:.6} (bound: {d})\n", .{ max_error, error_bound });
//             if (error_count == 0) {
//                 std.debug.print("  + All points within error bound\n", .{});
//             } else {
//                 std.debug.print("  ✗ {} points exceed error bound\n", .{error_count});
//             }
//         } else {
//             std.debug.print("  ✗ Size mismatch after decompression\n", .{});
//         }
//
//         // Only test one error bound if we find one that works quickly
//         if (duration_ms < 1000.0) { // Less than 1 second
//             std.debug.print("  Found working error bound: {d}\n", .{error_bound});
//             break;
//         }
//     }
// }

test "abc compressor compressor compresses and decompresses wafer dataset from TSV file" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Try to read the wafer data file
    var wafer_data = readWaferTSV(allocator, "Wafer_TRAIN.tsv") catch |err| {
        std.debug.print("Warning: Could not read Wafer_TRAIN.tsv: {}\n", .{err});
        std.debug.print("Skipping wafer test. Wafer_TRAIN.tsv might be not in the working directory.\n", .{});
        return;
    };
    defer wafer_data.deinit();

    std.debug.print("Loaded {} data points from Wafer_TRAIN.tsv\n", .{wafer_data.items.len});

    // // Print all parsed values
    // std.debug.print("\nAll parsed values:\n", .{});
    // for (wafer_data.items, 0..) |value, i| {
    //     if (i > 0 and i % 10 == 0) {
    //         std.debug.print("\n", .{}); // New line every 10 values
    //     }
    //     std.debug.print("{d:.3} ", .{value});
    // }
    // std.debug.print("\n\n", .{});

    try tester.testCompressAndDecompress(
        wafer_data.items,
        allocator,
        Method.ABCLinearApproximation,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

/// Read wafer data from TSV file:
/// train_data = pd.read_csv("Wafer_TRAIN.tsv", sep='\t', header=None)
/// if train_data.shape[1] > 1:
///     train_series = train_data.iloc[:, 1:].values.flatten()
///     train_series = train_series[~np.isnan(train_series)]
fn readWaferTSV(allocator: std.mem.Allocator, file_path: []const u8) !ArrayList(f64) {
    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        return err;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);

    _ = try file.readAll(contents);

    var train_series = ArrayList(f64).init(allocator);
    var lines = std.mem.splitScalar(u8, contents, '\n');
    var line_count: usize = 0;

    while (lines.next()) |line| {
        line_count += 1;
        if (line.len == 0) continue;

        var columns = std.mem.splitScalar(u8, line, '\t');
        var col_count: usize = 0;

        // Process all columns
        while (columns.next()) |col| {
            col_count += 1;
            if (col.len == 0) continue;

            // Skip first column
            if (col_count == 1) continue;

            const value = std.fmt.parseFloat(f64, col) catch |err| {
                // If we can't parse, skip this value
                std.debug.print("Warning: line {}, col {}: couldn't parse '{s}': {}\n", .{ line_count, col_count, col, err });
                continue;
            };

            // Skip NaN values
            if (std.math.isNan(value)) {
                continue;
            }

            try train_series.append(value);
        }
    }

    return train_series;
}
