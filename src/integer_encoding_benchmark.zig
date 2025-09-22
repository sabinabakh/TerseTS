const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const Timer = std.time.Timer;

const tersets = @import("tersets.zig");
const extractors = @import("utilities/extractors.zig");

// Import functional approximation methods
const abc_linear = @import("functional_approximation/abc_linear_approximation.zig");
const swing_slide = @import("functional_approximation/swing_slide_filter.zig");
const sim_piece = @import("functional_approximation/sim_piece.zig");
const poor_mans_compression = @import("functional_approximation/poor_mans_compression.zig");
const mix_piece = @import("functional_approximation/mix_piece.zig");

// Import integer encoding methods
const rle_encoding = @import("lossless_encoding/run_length_encoding.zig");
const vlq_encoding = @import("lossless_encoding/variable_length_quantity_encoding.zig");
const for_encoding = @import("lossless_encoding/frame_of_reference_encoding.zig");
const elias_fano_encoding = @import("lossless_encoding/elias_fano_encoding.zig");
const elias_gamma_encoding = @import("lossless_encoding/elias_gamma_encoding.zig");
const derle_encoding = @import("lossless_encoding/derle_encoding.zig");

const BenchmarkResult = struct {
    dataset_name: []const u8,
    compression_method: tersets.Method,
    timestamps_encoding_method: tersets.Method,
    original_size_bytes: usize,
    data_point_count: usize,
    compressed_size_bytes: usize,
    timestamps_encoded_size_bytes: usize,
    total_compressed_size_bytes: usize,
    compression_time_ns: u64,
    timestamps_encoding_time_ns: u64,
    total_compression_time_ns: u64,
    compression_ratio: f64,
    timestamps_encoding_ratio: f64,
    timestamp_count: usize,
    error_bound: f32,
    success: bool,
    error_msg: ?[]const u8,

    pub fn deinit(self: *BenchmarkResult, allocator: Allocator) void {
        if (self.error_msg) |msg| {
            allocator.free(msg);
        }
    }
};

const TimeSeriesDataset = struct {
    name: []const u8,
    data: []f64,

    pub fn deinit(self: *TimeSeriesDataset, allocator: Allocator) void {
        allocator.free(self.data);
    }
};

/// Load time series data from CSV file
fn loadTimeSeriesCSV(allocator: Allocator, file_path: []const u8) ![]f64 {
    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        print("Failed to open file {s}: {any}\n", .{ file_path, err });
        return err;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);
    _ = try file.readAll(contents);

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();

    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        const value = std.fmt.parseFloat(f64, trimmed) catch |err| {
            print("Failed to parse value '{s}': {any}\n", .{ trimmed, err });
            continue;
        };
        try data.append(value);
    }

    return data.toOwnedSlice();
}

/// Load all time series datasets from loaded_time_series folder
fn loadTimeSeries(allocator: Allocator) ![]TimeSeriesDataset {
    //For debugging: create a simple test time series
    // var loaded_datasets = ArrayList(TimeSeriesDataset).init(allocator);
    // errdefer {
    //     for (loaded_datasets.items) |*dataset| {
    //         dataset.deinit(allocator);
    //     }
    //     loaded_datasets.deinit();
    // }

    // // Create a simple test dataset: sine wave with some noise
    // const test_size = 10;
    // const test_data = try allocator.alloc(f64, test_size);

    // for (test_data, 0..) |*value, i| {
    //     const x = @as(f64, @floatFromInt(i)) * 0.1;
    //     value.* = @sin(x) + @sin(x * 2.5) + @as(f64, @floatFromInt(i % 10)) * 0.01;
    // }

    // try loaded_datasets.append(TimeSeriesDataset{
    //     .name = "test_sine_wave",
    //     .data = test_data,
    // });

    // print("Loaded test_sine_wave: {d} data points\n", .{test_data.len});

    // return loaded_datasets.toOwnedSlice();

    // Original CSV loading code (commented out for debugging):
    const datasets = [_]struct { name: []const u8, file: []const u8 }{
        .{ .name = "australian_electricity", .file = "src/benchmarking/loaded_time_series/australian_electricity_demand_series_0.csv" },
        .{ .name = "solar_4_seconds", .file = "src/benchmarking/loaded_time_series/solar_4_seconds_series_0.csv" },
        .{ .name = "wind_4_seconds", .file = "src/benchmarking/loaded_time_series/wind_4_seconds_series_0.csv" },
        .{ .name = "oikolab_weather", .file = "src/benchmarking/loaded_time_series/oikolab_weather_series_0.csv" },
    };

    var loaded_datasets = ArrayList(TimeSeriesDataset).init(allocator);
    errdefer {
        for (loaded_datasets.items) |*dataset| {
            dataset.deinit(allocator);
        }
        loaded_datasets.deinit();
    }

    for (datasets) |dataset_info| {
        const data = loadTimeSeriesCSV(allocator, dataset_info.file) catch |err| {
            print("Failed to load {s}: {any}\n", .{ dataset_info.name, err });
            continue;
        };

        try loaded_datasets.append(TimeSeriesDataset{
            .name = dataset_info.name,
            .data = data,
        });

        print("Loaded {s}: {d} data points\n", .{ dataset_info.name, data.len });
    }

    return loaded_datasets.toOwnedSlice();
}

/// Apply functional approximation method to time series data
fn applyFunctionalApproximation(
    allocator: Allocator,
    data: []const f64,
    method: tersets.Method,
    error_bound: f32,
) !struct { compressed: ArrayList(u8), time_ns: u64 } {
    var compressed = ArrayList(u8).init(allocator);
    errdefer compressed.deinit();

    var timer = try Timer.start();

    try switch (method) {
        .ABCLinearApproximation => abc_linear.compress(allocator, data, &compressed, error_bound),
        .SwingFilter => swing_slide.compressSwingFilter(data, &compressed, error_bound),
        .SwingFilterDisconnected => swing_slide.compressSwingFilterDisconnected(data, &compressed, error_bound),
        .SlideFilter => swing_slide.compressSlideFilter(allocator, data, &compressed, error_bound),
        .SimPiece => sim_piece.compress(allocator, data, &compressed, error_bound),
        .MixPiece => mix_piece.compress(allocator, data, &compressed, error_bound),
        .PoorMansCompressionMidrange => poor_mans_compression.compressMidrange(data, &compressed, error_bound),
        .PoorMansCompressionMean => poor_mans_compression.compressMean(data, &compressed, error_bound),

        else => return tersets.Error.UnknownMethod,
    };

    const elapsed_ns = timer.read();
    return .{ .compressed = compressed, .time_ns = elapsed_ns };
}

/// Apply integer encoding method to timestamp data
fn applyIntegerEncoding(
    allocator: Allocator,
    timestamps: []const usize,
    method: tersets.Method,
) !struct { compressed: ArrayList(u8), time_ns: u64 } {
    var compressed = ArrayList(u8).init(allocator);
    errdefer compressed.deinit();

    // Convert usize timestamps to f64 for encoding methods that expect f64
    var f64_timestamps = ArrayList(f64).init(allocator);
    defer f64_timestamps.deinit();

    for (timestamps) |ts| {
        try f64_timestamps.append(@floatFromInt(ts));
    }

    var timer = try Timer.start();

    try switch (method) {
        .DERLEEncoding => derle_encoding.compress(f64_timestamps.items, &compressed),
        .RunLengthEncoding => rle_encoding.compress(f64_timestamps.items, &compressed),
        .VariableLengthQuantityEncoding => vlq_encoding.compress(f64_timestamps.items, &compressed),
        .FrameOfReferenceEncoding => for_encoding.compress(f64_timestamps.items, &compressed),
        .EliasFanoEncoding => elias_fano_encoding.compress(f64_timestamps.items, &compressed),
        .EliasGammaEncoding => elias_gamma_encoding.compress(f64_timestamps.items, &compressed),
        else => return tersets.Error.UnknownMethod,
    };

    const elapsed_ns = timer.read();
    return .{ .compressed = compressed, .time_ns = elapsed_ns };
}

/// Extract timestamps and coefficients from compressed data based on functional method
fn extractTimestampsAndCoefficients(
    compressed_data: []const u8,
    method: tersets.Method,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) !void {
    switch (method) {
        .ABCLinearApproximation => {
            try extractors.extractABC(compressed_data, timestamps, coefficients);
        },
        .SimPiece => {
            try extractors.extractSimPiece(compressed_data, timestamps, coefficients);
        },
        .SwingFilter => {
            try extractors.extractSwing(compressed_data, timestamps, coefficients);
        },
        .SwingFilterDisconnected, .SlideFilter => {
            try extractors.extractSlideSwingDisconnected(compressed_data, timestamps, coefficients);
        },
        .PoorMansCompressionMidrange, .PoorMansCompressionMean => {
            try extractors.extractPMC(compressed_data, timestamps, coefficients);
        },
        .MixPiece => {
            try extractors.extractMixPiece(compressed_data, timestamps, coefficients);
        },
        else => {
            return tersets.Error.UnknownMethod;
        },
    }
}

/// Rebuild compressed data from timestamps and coefficients based on functional method
fn rebuildCompressedData(
    allocator: Allocator,
    timestamps: []const usize,
    coefficients: []const f64,
    method: tersets.Method,
    compressed_data: *ArrayList(u8),
) !void {
    _ = allocator;
    switch (method) {
        .ABCLinearApproximation => {
            try extractors.rebuildABC(timestamps, coefficients, compressed_data);
        },
        .SimPiece => {
            try extractors.rebuildSimPiece(timestamps, coefficients, compressed_data);
        },
        .SwingFilter => {
            try extractors.rebuildSwing(timestamps, coefficients, compressed_data);
        },
        .SwingFilterDisconnected, .SlideFilter => {
            try extractors.rebuildSlideSwingDisconnected(timestamps, coefficients, compressed_data);
        },
        .PoorMansCompressionMidrange, .PoorMansCompressionMean => {
            try extractors.rebuildPMC(timestamps, coefficients, compressed_data);
        },
        .MixPiece => {
            try extractors.rebuildMixPiece(timestamps, coefficients, compressed_data);
        },
        else => {
            return tersets.Error.UnknownMethod;
        },
    }
}

/// Run benchmark for a single dataset and method combination
fn runSingleBenchmark(
    allocator: Allocator,
    dataset_name: []const u8,
    data: []const f64,
    functional_method: tersets.Method,
    integer_method: tersets.Method,
    error_bound: f32,
) !BenchmarkResult {
    // Original size of the time series in bytes
    const original_size = data.len * @sizeOf(f64);

    var result = BenchmarkResult{
        .dataset_name = dataset_name,
        .compression_method = functional_method,
        .timestamps_encoding_method = integer_method,
        .original_size_bytes = original_size,
        .data_point_count = data.len,
        .compressed_size_bytes = 0,
        .timestamps_encoded_size_bytes = 0,
        .total_compressed_size_bytes = 0,
        .compression_time_ns = 0,
        .timestamps_encoding_time_ns = 0,
        .total_compression_time_ns = 0,
        .compression_ratio = 0.0,
        .timestamps_encoding_ratio = 0.0,
        .timestamp_count = 0,
        .error_bound = error_bound,
        .success = false,
        .error_msg = null,
    };

    // Step 1: Apply functional approximation
    const compression_result = applyFunctionalApproximation(
        allocator,
        data,
        functional_method,
        error_bound,
    ) catch |err| {
        result.error_msg = try allocator.dupe(u8, @errorName(err));
        return result;
    };
    defer compression_result.compressed.deinit();

    print("Compressed data: {any} for functional method {s} with error bound {d}\n", .{ compression_result.compressed.items, @tagName(functional_method), error_bound });

    // var decompressed = ArrayList(f64).init(allocator);
    // defer decompressed.deinit();
    // abc_linear.decompress(functional_result.compressed.items, &decompressed) catch |err| {
    //     print("Decompression failed: {any}\n", .{err});
    // };
    // print("Decompressed data: {any}\n", .{decompressed.items});

    result.compression_time_ns = compression_result.time_ns;
    result.compressed_size_bytes = compression_result.compressed.items.len;

    // Step 2: Extract timestamps and coefficients from compressed data
    var timestamps = ArrayList(usize).init(allocator);
    defer timestamps.deinit();
    var coefficients = ArrayList(f64).init(allocator);
    defer coefficients.deinit();

    extractTimestampsAndCoefficients(
        compression_result.compressed.items,
        functional_method,
        &timestamps,
        &coefficients,
    ) catch |err| {
        result.error_msg = try allocator.dupe(u8, @errorName(err));
        return result;
    };

    print("Extracted {d} timestamps and {d} coefficients for functional method {s}. Timestamps: {any}\n", .{ timestamps.items.len, coefficients.items.len, @tagName(functional_method), timestamps.items });

    result.timestamp_count = timestamps.items.len;

    // Step 3: Apply integer encoding to timestamps
    const timestamps_encoding_result = applyIntegerEncoding(
        allocator,
        timestamps.items,
        integer_method,
    ) catch |err| {
        result.error_msg = try allocator.dupe(u8, @errorName(err));
        return result;
    };
    defer timestamps_encoding_result.compressed.deinit();

    result.timestamps_encoded_size_bytes = timestamps_encoding_result.compressed.items.len;
    result.timestamps_encoding_time_ns = timestamps_encoding_result.time_ns;

    // Step 4: Calculate totals
    result.total_compressed_size_bytes = @sizeOf(u32) + result.compressed_size_bytes + result.timestamps_encoded_size_bytes;
    result.total_compression_time_ns = result.compression_time_ns + result.timestamps_encoding_time_ns;
    result.compression_ratio = @as(f64, @floatFromInt(result.original_size_bytes)) / @as(f64, @floatFromInt(result.total_compressed_size_bytes));

    const timestamps_size_bytes = timestamps.items.len * @sizeOf(usize);
    result.timestamps_encoding_ratio = @as(f64, @floatFromInt(timestamps_size_bytes)) / @as(f64, @floatFromInt(result.timestamps_encoded_size_bytes));

    result.success = true;
    return result;
}

/// Run benchmark with pre-computed functional compression results to avoid redundant computation
fn runBenchmarkWithPrecompressed(
    allocator: Allocator,
    dataset_name: []const u8,
    data: []const f64,
    functional_method: tersets.Method,
    integer_method: tersets.Method,
    error_bound: f32,
    compressed_data: []const u8,
    compression_time_ns: u64,
    timestamps: []const usize,
    coefficients: []const f64,
) !BenchmarkResult {
    _ = coefficients; // Not used in this function but kept for consistency

    // Original size of the time series in bytes
    const original_size = data.len * @sizeOf(f64);

    var result = BenchmarkResult{
        .dataset_name = dataset_name,
        .compression_method = functional_method,
        .timestamps_encoding_method = integer_method,
        .original_size_bytes = original_size,
        .data_point_count = data.len,
        .compressed_size_bytes = compressed_data.len,
        .timestamps_encoded_size_bytes = 0,
        .total_compressed_size_bytes = 0,
        .compression_time_ns = compression_time_ns,
        .timestamps_encoding_time_ns = 0,
        .total_compression_time_ns = 0,
        .compression_ratio = 0.0,
        .timestamps_encoding_ratio = 0.0,
        .timestamp_count = timestamps.len,
        .error_bound = error_bound,
        .success = false,
        .error_msg = null,
    };

    // Apply integer encoding to timestamps
    const timestamps_encoding_result = applyIntegerEncoding(
        allocator,
        timestamps,
        integer_method,
    ) catch |err| {
        result.error_msg = try allocator.dupe(u8, @errorName(err));
        return result;
    };
    defer timestamps_encoding_result.compressed.deinit();

    result.timestamps_encoded_size_bytes = timestamps_encoding_result.compressed.items.len;
    result.timestamps_encoding_time_ns = timestamps_encoding_result.time_ns;

    // Calculate totals
    result.total_compressed_size_bytes = @sizeOf(u32) + result.compressed_size_bytes + result.timestamps_encoded_size_bytes;
    result.total_compression_time_ns = result.compression_time_ns + result.timestamps_encoding_time_ns;
    result.compression_ratio = @as(f64, @floatFromInt(result.original_size_bytes)) / @as(f64, @floatFromInt(result.total_compressed_size_bytes));

    const timestamps_size_bytes = timestamps.len * @sizeOf(usize);
    result.timestamps_encoding_ratio = @as(f64, @floatFromInt(timestamps_size_bytes)) / @as(f64, @floatFromInt(result.timestamps_encoded_size_bytes));

    result.success = true;
    return result;
}

/// Print benchmark results
fn printResults(results: []const BenchmarkResult) void {
    print("\n=== INTEGER ENCODING BENCHMARK RESULTS ===\n\n", .{});

    print("{s:<20} {s:<15} {s:<12} {s:<6} {s:<8} {s:<10} {s:<6} {s:<8} {s:<4} {s}\n", .{ "Dataset", "Functional", "Integer", "Bound", "Points", "Orig(KB)", "CR", "Time(ms)", "TS", "Status" });
    print("{s}\n", .{"---------------------------------------------------------------------------------"});

    var successful: usize = 0;
    for (results) |result| {
        const status = if (result.success) "✓" else "✗";
        if (result.success) successful += 1;

        const time_ms = @as(f64, @floatFromInt(result.total_compression_time_ns)) / 1_000_000.0;
        const orig_kb = @as(f64, @floatFromInt(result.original_size_bytes)) / 1024.0;

        if (result.success) {
            print("{s:<20} {s:<15} {s:<12} {:.2} {d:<8} {:.0} {:.1} {:.2} {d:<4} {s}\n", .{
                result.dataset_name[0..@min(19, result.dataset_name.len)],
                @tagName(result.compression_method)[0..@min(14, @tagName(result.compression_method).len)],
                @tagName(result.timestamps_encoding_method)[0..@min(11, @tagName(result.timestamps_encoding_method).len)],
                result.error_bound,
                result.data_point_count,
                orig_kb,
                result.compression_ratio,
                time_ms,
                result.timestamp_count,
                status,
            });
        } else {
            print("{s:<20} {s:<15} {s:<12} {:.2} {d:<8} {:.0} - - - {s}\n", .{
                result.dataset_name[0..@min(19, result.dataset_name.len)],
                @tagName(result.compression_method)[0..@min(14, @tagName(result.compression_method).len)],
                @tagName(result.timestamps_encoding_method)[0..@min(11, @tagName(result.timestamps_encoding_method).len)],
                result.error_bound,
                result.data_point_count,
                orig_kb,
                status,
            });
        }
    }

    print("\nSummary: {d}/{d} combinations successful\n", .{ successful, results.len });

    if (successful > 0) {
        var total_cr: f64 = 0;
        var best_cr: f64 = 0;
        for (results) |result| {
            if (result.success) {
                total_cr += result.compression_ratio;
                if (result.compression_ratio > best_cr) {
                    best_cr = result.compression_ratio;
                }
            }
        }
        const avg_cr = total_cr / @as(f64, @floatFromInt(successful));
        print("Average compression ratio: {:.2}\n", .{avg_cr});
        print("Best compression ratio: {:.2}\n", .{best_cr});
    }
}

/// Write results to CSV file
fn writeCSV(allocator: Allocator, results: []const BenchmarkResult, file_path: []const u8) !void {
    _ = allocator;
    const file = std.fs.cwd().createFile(file_path, .{}) catch |err| {
        print("Failed to create CSV file {s}: {any}\n", .{ file_path, err });
        return err;
    };
    defer file.close();

    const writer = file.writer();

    // CSV Header
    try writer.print("dataset,compression_method,timestamps_encoding_method,error_bound,data_point_count,original_size_bytes,compressed_size_bytes,timestamps_encoded_bytes,total_compressed_bytes,compression_time_ns,timestamps_encoding_time_ns,total_time_ns,total_compression_ratio,timestamps_encoding_ratio,timestamp_count,success,error\n", .{});

    // CSV Data
    for (results) |result| {
        try writer.print("{s},{s},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d:.6},{d:.6},{d},{any},{s}\n", .{
            result.dataset_name,
            @tagName(result.compression_method),
            @tagName(result.timestamps_encoding_method),
            result.error_bound,
            result.data_point_count,
            result.original_size_bytes,
            result.compressed_size_bytes,
            result.timestamps_encoded_size_bytes,
            result.total_compressed_size_bytes,
            result.compression_time_ns,
            result.timestamps_encoding_time_ns,
            result.total_compression_time_ns,
            result.compression_ratio,
            result.timestamps_encoding_ratio,
            result.timestamp_count,
            result.success,
            result.error_msg orelse "",
        });
    }
}

/// Main benchmark function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== INTEGER ENCODING BENCHMARK (Time Series) ===\n\n", .{});

    // Load time series datasets
    const datasets = loadTimeSeries(allocator) catch |err| {
        print("Failed to load datasets: {any}\n", .{err});
        return err;
    };
    defer {
        for (datasets) |*dataset| {
            dataset.deinit(allocator);
        }
        allocator.free(datasets);
    }

    // Print loaded time series data for debugging
    for (datasets) |dataset| {
        print("Dataset '{s}': {d} data points\n", .{ dataset.name, dataset.data.len });
        for (dataset.data, 0..) |value, i| {
            if (i > 0) print(", ", .{});
            print("{:.3}", .{value});
        }
        print("\n\n", .{});
    }

    if (datasets.len == 0) {
        print("No datasets loaded. Make sure CSV files exist in src/integer_encoding_benchmarking/loaded_time_series/\n", .{});
        return;
    }

    // Define method combinations to test
    const functional_methods = [_]tersets.Method{
        .ABCLinearApproximation,
        .SwingFilterDisconnected,
        .SwingFilter,
        .SlideFilter,
        .PoorMansCompressionMidrange,
        .PoorMansCompressionMean,
        //.SimPiece,
        //.MixPiece,
    };

    const integer_methods = [_]tersets.Method{
        .DERLEEncoding,
        .RunLengthEncoding,
        .VariableLengthQuantityEncoding,
        .FrameOfReferenceEncoding,
        .EliasFanoEncoding,
        .EliasGammaEncoding,
    };

    const error_bounds = [_]f32{ 0.01, 0.1, 1.0 };

    var results = ArrayList(BenchmarkResult).init(allocator);
    defer {
        for (results.items) |*result| {
            result.deinit(allocator);
        }
        results.deinit();
    }

    // Total combinations to run
    const total_combinations = datasets.len * functional_methods.len * integer_methods.len * error_bounds.len;
    var current_combination: usize = 0;

    print("Running {d} benchmark combinations...\n", .{total_combinations});
    print("Optimization: Functional compression is computed once per (dataset, method, error_bound) triplet,\n", .{});
    print("then all integer encoding methods are applied to the same extracted timestamps.\n\n", .{});

    // Run benchmarks with optimized caching
    for (datasets) |dataset| {
        print("{s}: {d} points\n", .{ dataset.name, dataset.data.len });

        for (functional_methods) |func_method| {
            for (error_bounds) |error_bound| {
                // Step 1: Apply functional approximation once per (dataset, method, error_bound) combination
                const compression_result = applyFunctionalApproximation(
                    allocator,
                    dataset.data,
                    func_method,
                    error_bound,
                ) catch |err| {
                    // If functional compression fails, skip all integer method combinations for this triplet
                    current_combination += integer_methods.len;
                    print("  Functional compression failed for {s} (ε={:.2}): {any} - skipping {d} combinations\n", .{
                        @tagName(func_method),
                        error_bound,
                        err,
                        integer_methods.len,
                    });
                    continue;
                };
                defer compression_result.compressed.deinit();

                print("Compressed data: {any} for functional method {s} with error bound {d}\n", .{ compression_result.compressed.items, @tagName(func_method), error_bound });

                // Step 2: Extract timestamps and coefficients once
                var timestamps = ArrayList(usize).init(allocator);
                defer timestamps.deinit();
                var coefficients = ArrayList(f64).init(allocator);
                defer coefficients.deinit();

                extractTimestampsAndCoefficients(
                    compression_result.compressed.items,
                    func_method,
                    &timestamps,
                    &coefficients,
                ) catch |err| {
                    // If extraction fails, skip all integer method combinations for this triplet
                    current_combination += integer_methods.len;
                    print("  Timestamp extraction failed for {s} (ε={:.2}): {any} - skipping {d} combinations\n", .{
                        @tagName(func_method),
                        error_bound,
                        err,
                        integer_methods.len,
                    });
                    continue;
                };

                print("Extracted {d} timestamps and {d} coefficients for functional method {s}. Timestamps: {any}\n", .{ timestamps.items.len, coefficients.items.len, @tagName(func_method), timestamps.items });

                // Step 3: Now apply each integer encoding method to the same timestamps
                for (integer_methods) |int_method| {
                    current_combination += 1;

                    const result = runBenchmarkWithPrecompressed(
                        allocator,
                        dataset.name,
                        dataset.data,
                        func_method,
                        int_method,
                        error_bound,
                        compression_result.compressed.items,
                        compression_result.time_ns,
                        timestamps.items,
                        coefficients.items,
                    ) catch |err| {
                        print("  [{d}/{d}] {s} + {s} (ε={:.2}) ❌ {any}\n", .{
                            current_combination,
                            total_combinations,
                            @tagName(func_method)[0..@min(12, @tagName(func_method).len)],
                            @tagName(int_method)[0..@min(10, @tagName(int_method).len)],
                            error_bound,
                            err,
                        });
                        continue;
                    };

                    try results.append(result);

                    const status = if (result.success) "✓" else "❌";
                    if (result.success) {
                        print("  [{d}/{d}] {s} + {s} (ε={:.2}) {s} CR: {:.1}\n", .{
                            current_combination,
                            total_combinations,
                            @tagName(func_method)[0..@min(12, @tagName(func_method).len)],
                            @tagName(int_method)[0..@min(10, @tagName(int_method).len)],
                            error_bound,
                            status,
                            result.compression_ratio,
                        });
                    } else {
                        print("  [{d}/{d}] {s} + {s} (ε={:.2}) {s} {s}\n", .{
                            current_combination,
                            total_combinations,
                            @tagName(func_method)[0..@min(12, @tagName(func_method).len)],
                            @tagName(int_method)[0..@min(10, @tagName(int_method).len)],
                            error_bound,
                            status,
                            result.error_msg orelse "Unknown error",
                        });
                    }
                }
            }
        }
        print("\n", .{});
    }

    // Print and save results
    print("Benchmark completed!\n", .{});
    printResults(results.items);

    // Save to CSV
    const csv_path = "src/benchmarking/integer_encoding_benchmark_results/integer_encoding_benchmark_results.csv";
    writeCSV(allocator, results.items, csv_path) catch |err| {
        print("Failed to write CSV: {any}\n", .{err});
    };
    print("\nResults saved to: {s}\n", .{csv_path});
}
