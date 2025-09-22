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

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const shared_functions = @import("../utilities/shared_functions.zig");

pub fn extractABC(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // ABC format: (slope, intercept, end_index) triplets
    if (compressed_values.len % 24 != 0) return tersets.Error.UnsupportedInput;

    const components = mem.bytesAsSlice(f64, compressed_values);

    var i: usize = 0;
    while (i + 2 < components.len) : (i += 3) {
        const slope = components[i];
        const intercept = components[i + 1];
        const end_idx: usize = @bitCast(components[i + 2]);

        try coefficients.append(slope);
        try coefficients.append(intercept);
        try timestamps.append(end_idx);
    }
}

pub fn extractPMC(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // PMC layout: pairs of (f64 value, f64 bit-cast of usize end_index)
    if (compressed_values.len % 16 != 0) return Error.UnsupportedInput;

    const components = mem.bytesAsSlice(f64, compressed_values); // 2 f64 per pair

    for (0..components.len) |i| {
        if (i % 2 == 0) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

pub fn extractSwing(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    if ((compressed_values.len - 8) % 16 != 0) return Error.UnsupportedInput;

    const components = mem.bytesAsSlice(f64, compressed_values); // 2 f64 per pair
    for (0..components.len) |i| {
        if ((i == 0) or (i % 2 == 1)) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

pub fn extractSlideSwingDisconnected(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const components = mem.bytesAsSlice(f64, compressed_values); // 2 f64 per pair
    for (0..components.len) |i| {
        if ((i + 1) % 3 != 0) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

// TODO: Correct the extractor for sim
pub fn extractSimPiece(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // Variable-length format; must be at least 1 f64 (final last_timestamp).
    const items = mem.bytesAsSlice(f64, compressed_values);
    if (items.len == 0) return Error.UnsupportedInput;

    var i: usize = 0;

    // Parse all groups, keeping the very last item for final last_timestamp.
    while (i < items.len - 1) {
        // intercept (f64)
        const intercept = items[i];
        try coefficients.append(intercept);
        i += 1;

        // slopes_count (usize as f64 bits)
        if (i >= items.len - 1) return Error.UnsupportedInput;
        const slopes_count: usize = @bitCast(items[i]);
        try timestamps.append(slopes_count);
        i += 1;

        // slopes blocks
        var s: usize = 0;
        while (s < slopes_count) : (s += 1) {
            // slope (f64)
            if (i >= items.len - 1) return Error.UnsupportedInput;
            const slope = items[i];
            try coefficients.append(slope);
            i += 1;

            // timestamps_count (usize)
            if (i >= items.len - 1) return Error.UnsupportedInput;
            const tcount: usize = @bitCast(items[i]);
            try timestamps.append(tcount);
            i += 1;

            // deltas (usize each)
            var t: usize = 0;
            while (t < tcount) : (t += 1) {
                if (i >= items.len - 1) return Error.UnsupportedInput;
                const delta: usize = @bitCast(items[i]);
                try timestamps.append(delta);
                i += 1;
            }
        }
    }

    // Final last_timestamp (usize) is the last f64 in the payload
    if (i != items.len - 1) return Error.UnsupportedInput;
    const last_ts: usize = @bitCast(items[i]);
    try timestamps.append(last_ts);
}

// TODO: Correct the extractor for mix
pub fn extractMixPiece(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    const header = mem.bytesAsSlice(usize, compressed_values[0 .. 3 * @sizeOf(usize)]);

    const part1_count = header[0]; // Number of intercept groups in Part 1.
    const part2_count = header[1]; // Number of slope groups in Part 2.
    const part3_count = header[2]; // Number of ungrouped segments in Part 3.

    try timestamps.append(header[0]);
    try timestamps.append(header[1]);
    try timestamps.append(header[2]);

    var offset: usize = 3 * @sizeOf(usize);
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            const intercept = try shared_functions.readValue(f64, compressed_values, &offset);
            try coefficients.append(intercept);

            const slopes_count = try shared_functions.readValue(usize, compressed_values, &offset);
            try timestamps.append(slopes_count);

            for (0..slopes_count) |_| {
                const slope = try shared_functions.readValue(f64, compressed_values, &offset);

                try coefficients.append(slope);

                const timestamps_count = try shared_functions.readValue(
                    usize,
                    compressed_values,
                    &offset,
                );

                try timestamps.append(timestamps_count);

                for (0..timestamps_count) |_| {
                    const delta = try shared_functions.readValue(usize, compressed_values, &offset);

                    try timestamps.append(delta);
                }
            }
        }
    }
    if (part2_count > 0) {
        for (0..part2_count) |_| {
            const slope = try shared_functions.readValue(f64, compressed_values, &offset);
            try coefficients.append(slope);

            const pair_count = try shared_functions.readValue(usize, compressed_values, &offset);
            try timestamps.append(pair_count);

            for (0..pair_count) |_| {
                const intercept = try shared_functions.readValue(f64, compressed_values, &offset);
                try coefficients.append(intercept);

                const delta = try shared_functions.readValue(usize, compressed_values, &offset);
                try timestamps.append(delta);
            }
        }
    }
    if (part3_count > 0) {
        for (0..part3_count) |_| {
            const slope = try shared_functions.readValue(f64, compressed_values, &offset);
            try coefficients.append(slope);

            const intercept = try shared_functions.readValue(f64, compressed_values, &offset);
            try coefficients.append(intercept);

            const delta = try shared_functions.readValue(usize, compressed_values, &offset);
            try timestamps.append(delta);
        }
    }
    const final_timestamp = try shared_functions.readValue(usize, compressed_values, &offset);
    try timestamps.append(final_timestamp);
}

pub fn rebuildABC(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // ABC format stores triplets: (slope, intercept, end_index)
    // coefficients contains pairs: [slope, intercept, slope, intercept, ...]
    // timestamps contains: [end_index, end_index, ...]
    if (coefficients.len % 2 != 0) return Error.UnsupportedInput;
    if (timestamps.len * 2 != coefficients.len) return Error.UnsupportedInput;

    // Each triplet is 24 bytes (three f64). Reserve once.
    try compressed_values.ensureTotalCapacity(timestamps.len * 24);

    var coeff_idx: usize = 0;
    for (timestamps) |end_idx| {
        // Append slope (f64)
        const slope = coefficients[coeff_idx];
        try shared_functions.appendValue(f64, slope, compressed_values);
        coeff_idx += 1;

        // Append intercept (f64)
        const intercept = coefficients[coeff_idx];
        try shared_functions.appendValue(f64, intercept, compressed_values);
        coeff_idx += 1;

        // Append end_index as f64 bit-cast
        const end_idx_as_f64: f64 = @bitCast(end_idx);
        try shared_functions.appendValue(f64, end_idx_as_f64, compressed_values);
    }
}

pub fn rebuildPMC(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (timestamps.len != coefficients.len) return Error.UnsupportedInput;

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if (i % 2 == 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

pub fn rebuildSwing(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if ((i == 0) or (i % 2 == 1)) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

pub fn rebuildSlideSwingDisconnected(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 24);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if ((i + 1) % 3 != 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

pub fn rebuildSimPiece(
    timestamps: []const usize, // IN: [slopes_count, (time_count, deltas...), ..., last_time]
    coefficients: []const f64, // IN: [intercept, slope, slope, ...]
    compressed_values: *ArrayList(u8), // OUT: payload only (no method byte)
) Error!void {
    // We need at least the final last_timestamp.
    if (timestamps.len == 0) return Error.UnsupportedInput;

    var ci: usize = 0; // index into coefficients
    var ti: usize = 0; // index into timestamps

    // For each intercept:
    while (ci < coefficients.len) {
        // intercept
        try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
        ci += 1;

        if (ti >= timestamps.len) return Error.UnsupportedInput;
        const slopes_count = timestamps[ti];
        try shared_functions.appendValue(usize, slopes_count, compressed_values);
        ti += 1;

        // slopes for this intercept
        var s: usize = 0;
        while (s < slopes_count) : (s += 1) {
            if (ci >= coefficients.len) return Error.UnsupportedInput;
            // slope
            try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
            ci += 1;

            if (ti >= timestamps.len) return Error.UnsupportedInput;
            const tcount = timestamps[ti];
            try shared_functions.appendValue(usize, tcount, compressed_values);
            ti += 1;

            // deltas
            var t: usize = 0;
            while (t < tcount) : (t += 1) {
                if (ti >= timestamps.len) return Error.UnsupportedInput;
                const delta = timestamps[ti];
                try shared_functions.appendValue(usize, delta, compressed_values);
                ti += 1;
            }
        }
    }

    // Must have exactly one trailing last_timestamp remaining
    if (ti >= timestamps.len) return Error.UnsupportedInput;
    const last_ts = timestamps[ti];
    try shared_functions.appendValue(usize, last_ts, compressed_values);
    ti += 1;

    // No extra data should remain
    if (ti != timestamps.len) return Error.UnsupportedInput;
}

pub fn rebuildMixPiece(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    const part1_count = timestamps[0]; // Number of intercept groups in Part 1.
    const part2_count = timestamps[1]; // Number of slope groups in Part 2.
    const part3_count = timestamps[2]; // Number of ungrouped segments in Part 3.

    try shared_functions.appendValue(usize, part1_count, compressed_values);
    try shared_functions.appendValue(usize, part2_count, compressed_values);
    try shared_functions.appendValue(usize, part3_count, compressed_values);

    var timestamps_idx: usize = 3;
    var coefficients_idx: usize = 0;
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            const intercept = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, intercept, compressed_values);
            coefficients_idx += 1;

            const slopes_count = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, slopes_count, compressed_values);
            timestamps_idx += 1;

            for (0..slopes_count) |_| {
                const slope = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, slope, compressed_values);
                coefficients_idx += 1;

                const timestamps_count = timestamps[timestamps_idx];
                try shared_functions.appendValue(usize, timestamps_count, compressed_values);
                timestamps_idx += 1;

                for (0..timestamps_count) |_| {
                    const delta = timestamps[timestamps_idx];
                    try shared_functions.appendValue(usize, delta, compressed_values);
                    timestamps_idx += 1;
                }
            }
        }
    }
    if (part2_count > 0) {
        for (0..part2_count) |_| {
            const slope = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, slope, compressed_values);
            coefficients_idx += 1;

            const pair_count = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, pair_count, compressed_values);
            timestamps_idx += 1;

            for (0..pair_count) |_| {
                const intercept = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, intercept, compressed_values);
                coefficients_idx += 1;

                const delta = timestamps[timestamps_idx];
                try shared_functions.appendValue(usize, delta, compressed_values);
                timestamps_idx += 1;
            }
        }
    }
    if (part3_count > 0) {
        for (0..part3_count) |_| {
            const slope = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, slope, compressed_values);
            coefficients_idx += 1;

            const intercept = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, intercept, compressed_values);
            coefficients_idx += 1;

            const delta = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, delta, compressed_values);
            timestamps_idx += 1;
        }
    }
    const final_timestamp = timestamps[timestamps_idx];
    try shared_functions.appendValue(usize, final_timestamp, compressed_values);
}
