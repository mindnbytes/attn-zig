const std = @import("std");

const N = 4; // number of tokens
const D = 4; // embedding dimension
const Dh = 2; // attention head dimension
const sqrtDh = std.math.sqrt(@as(f32, Dh)); // scale

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
    // Print stuff
    var buffer: [1024]u8 = undefined;
    var stdout_file: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;

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

    try printMatrix(N, Dh, stdout, "Q", Q);
    try printMatrix(N, Dh, stdout, "K", K);
    try printMatrix(N, Dh, stdout, "V", V);

    // Attention scores Q * Kt: N x Dh * Dh x N = N x N
    // Each query measures its compatibility with every token's key
    // Q * K -> where/how strongly to attend
    // Query of token i (row) has a Key relevance score of all tokens (colums)
    // We aren't realizing transposed K
    var attn_score: [N][N]f32 = undefined;
    for (0..N) |row| {
        for (0..N) |col| {
            attn_score[row][col] = 0;
            for (0..Dh) |dim| {
                attn_score[row][col] += Q[row][dim] * K[col][dim];
            }
            attn_score[row][col] /= sqrtDh;
        }
    }
    try printMatrix(N, N, stdout, "attn_score", attn_score);
    // Causal attention mask - you can do it during the previous step
    // I just want to print everything out
    for (0..N) |row| {
        for (0..N) |col| {
            if (col > row) {
                attn_score[row][col] = -std.math.inf(f32);
            }
        }
    }
    try printMatrix(N, N, stdout, "attn_score with mask applied", attn_score);

    // copy scores to weights
    var attn_weight: [N][N]f32 = undefined;
    @memcpy(&attn_weight, &attn_score);
    // softmax - but stable (mathematically the same)
    for (0..N) |row| {
        // 1. Find max
        var row_max = -std.math.inf(f32);
        for (0..N) |col| {
            row_max = @max(row_max, attn_weight[row][col]);
        }
        // 2. Exp(score - max) and aggregate it into the row sum
        var row_exp_sum: f32 = 0;
        for (0..N) |col| {
            const exp_score = std.math.exp(attn_weight[row][col] - row_max);
            attn_weight[row][col] = exp_score;
            row_exp_sum += exp_score;
        }
        //  3. Scale the exp score by the row exp score sum
        for (0..N) |col| {
            attn_weight[row][col] /= row_exp_sum;
        }
    }

    try printMatrix(N, N, stdout, "attention weights", attn_weight);

    // compute final part of it - weighted by query-key correspondense pieces of information from V
    const attn = matmul(N, N, Dh, attn_weight, V);
    try printMatrix(N, Dh, stdout, "attention finally", attn);
}
