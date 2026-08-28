const std = @import("std");

const N = 4; // number of tokens
const D = 4; // embedding dimension
const Dh = 2; // attention head dimension

fn matmul(
    comptime rows: usize,
    comptime inner: usize,
    comptime columns: usize,
    lhs: [rows][inner]f32,
    rhs: [inner][columns]f32,
) [rows][columns]f32 {
    var result: [rows][columns]f32 = undefined;

    for (0..rows) |row| {
        for (0..columns) |column| {
            result[row][column] = 0.0;
            for (0..inner) |i| {
                result[row][column] += lhs[row][i] * rhs[i][column];
            }
        }
    }

    return result;
}

fn printMatrix(
    comptime rows: usize,
    comptime columns: usize,
    writer: *std.Io.Writer,
    name: []const u8,
    matrix: [rows][columns]f32,
) !void {
    try writer.print("{s} result:\n", .{name});
    for (matrix) |row| {
        try writer.print(" {any}\n", .{row});
    }
    try writer.flush();
}

pub fn main(init: std.process.Init) !void {
    // X: N x D = 4 x 4
    const X: [N][D]f32 = .{
        .{ 0.0, 1.0, 0.0, 2.0 },
        .{ 1.0, 0.0, 0.5, 2.0 },
        .{ 1.0, 0.0, 0.0, 1.0 },
        .{ 2.0, 0.5, 1.0, 0.0 },
    };
    // Wq: D x Dh 4 x 2
    const Wq: [D][Dh]f32 = .{
        .{ 0.5, 1.0 },
        .{ 1.0, 0.0 },
        .{ 2.0, 0.5 },
        .{ 0.5, 2.0 },
    };
    // Wk: D x Dh 4 x 2
    const Wk: [D][Dh]f32 = .{
        .{ 1.5, 2.0 },
        .{ 1.0, 1.0 },
        .{ 0.5, 1.5 },
        .{ 1.0, 0.0 },
    };
    // Wv: D x Dh 4 x 2
    const Wv: [D][Dh]f32 = .{
        .{ 2.5, 0.0 },
        .{ 0.0, 1.5 },
        .{ 1.0, 2.0 },
        .{ 1.0, 0.5 },
    };

    // X * Wq = Q: N x D * D x Dh = N x Dh
    const Q = matmul(N, D, Dh, X, Wq);
    // X * Wk = K: N x D * D x Dh = N x Dh
    const K = matmul(N, D, Dh, X, Wk);
    // X * Wv = V: N x D * D x Dh = N x Dh
    const V = matmul(N, D, Dh, X, Wv);

    var buffer: [1024]u8 = undefined;
    var stdout_file: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;

    try printMatrix(N, Dh, stdout, "Q", Q);
    try printMatrix(N, Dh, stdout, "K", K);
    try printMatrix(N, Dh, stdout, "V", V);
}
