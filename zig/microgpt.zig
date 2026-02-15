// microgpt.zig
// The most atomic way to train and inference a GPT in pure Zig.
// Ported from Karpathy's microgpt.py — zero dependencies beyond std.

const std = @import("std");
const math = std.math;
const print = std.debug.print;

// =========================================================================
// Configuration
// =========================================================================

const N_EMBD = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 16;
const HEAD_DIM = N_EMBD / N_HEAD;
const NUM_STEPS = 1000;
const LR: f64 = 0.01;
const BETA1: f64 = 0.85;
const BETA2: f64 = 0.99;
const EPS_ADAM: f64 = 1e-8;
const TEMPERATURE: f64 = 0.5;
const ARENA_SIZE = 2_000_000;
const MAX_DOCS = 50_000;
const MAX_DOC_LEN = 64;
const MAX_CHARS = 128;

// =========================================================================
// Arena-based autograd
// =========================================================================

const Value = struct {
    data: f64,
    grad: f64,
    c0: i32,
    c1: i32,
    lg0: f64,
    lg1: f64,
    nc: u8,
};

var arena: [ARENA_SIZE]Value = undefined;
var arena_counter: usize = 0;
var num_params_base: usize = 0;

fn valNew(data: f64) usize {
    const idx = arena_counter;
    arena_counter += 1;
    arena[idx] = .{ .data = data, .grad = 0, .c0 = -1, .c1 = -1, .lg0 = 0, .lg1 = 0, .nc = 0 };
    return idx;
}

fn valAdd(a: usize, b: usize) usize {
    const idx = valNew(arena[a].data + arena[b].data);
    arena[idx].c0 = @intCast(a);
    arena[idx].c1 = @intCast(b);
    arena[idx].lg0 = 1;
    arena[idx].lg1 = 1;
    arena[idx].nc = 2;
    return idx;
}

fn valMul(a: usize, b: usize) usize {
    const idx = valNew(arena[a].data * arena[b].data);
    arena[idx].c0 = @intCast(a);
    arena[idx].c1 = @intCast(b);
    arena[idx].lg0 = arena[b].data;
    arena[idx].lg1 = arena[a].data;
    arena[idx].nc = 2;
    return idx;
}

fn valAddC(a: usize, c: f64) usize { return valAdd(a, valNew(c)); }
fn valMulC(a: usize, c: f64) usize { return valMul(a, valNew(c)); }
fn valNeg(a: usize) usize { return valMulC(a, -1.0); }
fn valDivC(a: usize, c: f64) usize { return valMulC(a, 1.0 / c); }

fn valPowC(a: usize, n: f64) usize {
    const ad = arena[a].data;
    const idx = valNew(math.pow(f64, ad, n));
    arena[idx].c0 = @intCast(a);
    arena[idx].lg0 = n * math.pow(f64, ad, n - 1.0);
    arena[idx].nc = 1;
    return idx;
}

fn valDiv(a: usize, b: usize) usize { return valMul(a, valPowC(b, -1.0)); }

fn valExpV(a: usize) usize {
    const e = @exp(arena[a].data);
    const idx = valNew(e);
    arena[idx].c0 = @intCast(a);
    arena[idx].lg0 = e;
    arena[idx].nc = 1;
    return idx;
}

fn valLogV(a: usize) usize {
    const ad = arena[a].data;
    const idx = valNew(@log(ad));
    arena[idx].c0 = @intCast(a);
    arena[idx].lg0 = 1.0 / ad;
    arena[idx].nc = 1;
    return idx;
}

fn valRelu(a: usize) usize {
    const ad = arena[a].data;
    const idx = valNew(if (ad > 0) ad else 0);
    arena[idx].c0 = @intCast(a);
    arena[idx].lg0 = if (ad > 0) 1.0 else 0.0;
    arena[idx].nc = 1;
    return idx;
}

fn sumValues(vals: []const usize) usize {
    var s = vals[0];
    for (vals[1..]) |v| { s = valAdd(s, v); }
    return s;
}

// =========================================================================
// Backward (iterative DFS topo sort)
// =========================================================================

var visited_buf: [ARENA_SIZE]bool = undefined;
var topo_buf: [ARENA_SIZE]usize = undefined;
var topo_count: usize = 0;

const DFSFrame = struct { node: usize, ci: u8 };
var dfs_stack: [ARENA_SIZE]DFSFrame = undefined;

fn backward(root: usize) void {
    @memset(visited_buf[0..arena_counter], false);
    topo_count = 0;

    var sp: usize = 0;
    dfs_stack[0] = .{ .node = root, .ci = 0 };
    sp = 1;

    while (sp > 0) {
        const frame = &dfs_stack[sp - 1];
        const node = frame.node;

        if (visited_buf[node]) { sp -= 1; continue; }

        if (frame.ci < arena[node].nc) {
            const child: usize = if (frame.ci == 0) @intCast(arena[node].c0) else @intCast(arena[node].c1);
            frame.ci += 1;
            if (!visited_buf[child]) {
                dfs_stack[sp] = .{ .node = child, .ci = 0 };
                sp += 1;
            }
        } else {
            visited_buf[node] = true;
            topo_buf[topo_count] = node;
            topo_count += 1;
            sp -= 1;
        }
    }

    for (topo_buf[0..topo_count]) |i| { arena[i].grad = 0; }
    arena[root].grad = 1.0;

    var i: usize = topo_count;
    while (i > 0) {
        i -= 1;
        const v = topo_buf[i];
        const g = arena[v].grad;
        if (arena[v].nc >= 1) {
            const c0: usize = @intCast(arena[v].c0);
            arena[c0].grad += arena[v].lg0 * g;
        }
        if (arena[v].nc >= 2) {
            const c1: usize = @intCast(arena[v].c1);
            arena[c1].grad += arena[v].lg1 * g;
        }
    }
}

// =========================================================================
// RNG (xorshift64)
// =========================================================================

var rng_state: u64 = 42;

fn rngNext() f64 {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return @as(f64, @floatFromInt(rng_state >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
}

fn gaussRandom(s: f64) f64 {
    const u1 = @max(rngNext(), 1e-300);
    const u2 = rngNext();
    return s * @sqrt(-2.0 * @log(u1)) * @cos(2.0 * math.pi * u2);
}

fn weightedChoice(w: []const f64) usize {
    var total: f64 = 0;
    for (w) |x| { total += x; }
    var r = rngNext() * total;
    for (w, 0..) |x, i| { r -= x; if (r <= 0) return i; }
    return w.len - 1;
}

// =========================================================================
// Forward helpers
// =========================================================================

fn linearFwd(out: []usize, w: []const usize, nout: usize, nin: usize, x: []const usize) void {
    var prods_buf: [4 * N_EMBD]usize = undefined;
    for (0..nout) |i| {
        for (0..nin) |j| {
            prods_buf[j] = valMul(w[i * nin + j], x[j]);
        }
        out[i] = sumValues(prods_buf[0..nin]);
    }
}

fn softmaxFwd(out: []usize, logits: []const usize, n: usize) void {
    var mx: f64 = arena[logits[0]].data;
    for (logits[1..n]) |l| { if (arena[l].data > mx) mx = arena[l].data; }

    var exps_buf: [MAX_CHARS]usize = undefined;
    for (0..n) |i| { exps_buf[i] = valExpV(valAddC(logits[i], -mx)); }
    const total = sumValues(exps_buf[0..n]);
    for (0..n) |i| { out[i] = valDiv(exps_buf[i], total); }
}

fn rmsnormFwd(out: []usize, x: []const usize, n: usize) void {
    var sq_buf: [N_EMBD]usize = undefined;
    for (0..n) |i| { sq_buf[i] = valMul(x[i], x[i]); }
    const ms = valDivC(sumValues(sq_buf[0..n]), @as(f64, @floatFromInt(n)));
    const scale = valPowC(valAddC(ms, 1e-5), -0.5);
    for (0..n) |i| { out[i] = valMul(x[i], scale); }
}

// =========================================================================
// Model storage
// =========================================================================

var wte: [MAX_CHARS][N_EMBD]usize = undefined;
var wpe: [BLOCK_SIZE][N_EMBD]usize = undefined;
var lm_head: [MAX_CHARS][N_EMBD]usize = undefined;
var attn_wq: [N_LAYER][N_EMBD * N_EMBD]usize = undefined;
var attn_wk: [N_LAYER][N_EMBD * N_EMBD]usize = undefined;
var attn_wv: [N_LAYER][N_EMBD * N_EMBD]usize = undefined;
var attn_wo: [N_LAYER][N_EMBD * N_EMBD]usize = undefined;
var mlp_fc1: [N_LAYER][4 * N_EMBD * N_EMBD]usize = undefined;
var mlp_fc2: [N_LAYER][N_EMBD * 4 * N_EMBD]usize = undefined;

var params_buf: [8192]usize = undefined;
var num_params: usize = 0;

fn allocMatrix(mat: []usize, nout: usize, nin: usize) void {
    for (0..nout) |i| {
        for (0..nin) |j| {
            const idx = valNew(gaussRandom(0.08));
            mat[i * nin + j] = idx;
            params_buf[num_params] = idx;
            num_params += 1;
        }
    }
}

// KV cache
var kv_keys: [N_LAYER][BLOCK_SIZE][N_EMBD]usize = undefined;
var kv_vals: [N_LAYER][BLOCK_SIZE][N_EMBD]usize = undefined;
var kv_len: [N_LAYER]usize = undefined;

fn kvReset() void { for (&kv_len) |*l| { l.* = 0; } }

// =========================================================================
// GPT forward
// =========================================================================

fn gptForward(logits_out: []usize, token_id: usize, pos_id: usize, vs: usize) void {
    var x: [N_EMBD]usize = undefined;
    var xn: [N_EMBD]usize = undefined;

    for (0..N_EMBD) |i| { x[i] = valAdd(wte[token_id][i], wpe[pos_id][i]); }
    rmsnormFwd(&xn, &x, N_EMBD);
    @memcpy(&x, &xn);

    for (0..N_LAYER) |li| {
        var x_res: [N_EMBD]usize = undefined;
        @memcpy(&x_res, &x);

        rmsnormFwd(&xn, &x, N_EMBD);
        @memcpy(&x, &xn);

        var q: [N_EMBD]usize = undefined;
        var k: [N_EMBD]usize = undefined;
        var v: [N_EMBD]usize = undefined;
        linearFwd(&q, &attn_wq[li], N_EMBD, N_EMBD, &x);
        linearFwd(&k, &attn_wk[li], N_EMBD, N_EMBD, &x);
        linearFwd(&v, &attn_wv[li], N_EMBD, N_EMBD, &x);

        const t = kv_len[li];
        @memcpy(&kv_keys[li][t], &k);
        @memcpy(&kv_vals[li][t], &v);
        kv_len[li] = t + 1;

        var x_attn: [N_EMBD]usize = undefined;
        var ai: usize = 0;
        for (0..N_HEAD) |h| {
            const hs = h * HEAD_DIM;
            const seq = kv_len[li];

            var al: [BLOCK_SIZE]usize = undefined;
            for (0..seq) |tt| {
                var pr: [HEAD_DIM]usize = undefined;
                for (0..HEAD_DIM) |d| {
                    pr[d] = valMul(q[hs + d], kv_keys[li][tt][hs + d]);
                }
                al[tt] = valDivC(sumValues(pr[0..HEAD_DIM]), @sqrt(@as(f64, @floatFromInt(HEAD_DIM))));
            }
            var aw: [BLOCK_SIZE]usize = undefined;
            softmaxFwd(&aw, al[0..seq], seq);

            for (0..HEAD_DIM) |d| {
                var terms: [BLOCK_SIZE]usize = undefined;
                for (0..seq) |tt| {
                    terms[tt] = valMul(aw[tt], kv_vals[li][tt][hs + d]);
                }
                x_attn[ai] = sumValues(terms[0..seq]);
                ai += 1;
            }
        }

        var xp: [N_EMBD]usize = undefined;
        linearFwd(&xp, &attn_wo[li], N_EMBD, N_EMBD, &x_attn);
        for (0..N_EMBD) |i| { x[i] = valAdd(xp[i], x_res[i]); }

        var x_res2: [N_EMBD]usize = undefined;
        @memcpy(&x_res2, &x);

        rmsnormFwd(&xn, &x, N_EMBD);
        @memcpy(&x, &xn);

        var h1: [4 * N_EMBD]usize = undefined;
        linearFwd(&h1, &mlp_fc1[li], 4 * N_EMBD, N_EMBD, &x);
        for (0..4 * N_EMBD) |i| { h1[i] = valRelu(h1[i]); }

        var h2: [N_EMBD]usize = undefined;
        linearFwd(&h2, &mlp_fc2[li], N_EMBD, 4 * N_EMBD, &h1);
        for (0..N_EMBD) |i| { x[i] = valAdd(h2[i], x_res2[i]); }
    }

    linearFwd(logits_out, &lm_head[0], vs, N_EMBD, &x);
}

// =========================================================================
// Dataset
// =========================================================================

var docs: [MAX_DOCS][MAX_DOC_LEN]u8 = undefined;
var doc_lens: [MAX_DOCS]usize = undefined;
var num_docs: usize = 0;
var uchars: [MAX_CHARS]u8 = undefined;
var num_uchars: usize = 0;
var char_to_idx: [256]usize = undefined;
var bos: usize = 0;
var vocab_size: usize = 0;

fn loadDataset() !void {
    var file = std.fs.cwd().openFile("input.txt", .{}) catch {
        print("Downloading names dataset...\n", .{});
        var child = std.process.Child.init(.{
            .argv = &.{ "curl", "-sL", "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt", "-o", "input.txt" },
        });
        _ = try child.spawnAndWait();
        file = try std.fs.cwd().openFile("input.txt", .{});
        return loadDataset();
    };
    defer file.close();

    var buf: [256]u8 = undefined;
    var reader = file.reader();
    while (true) {
        const line = reader.readUntilDelimiter(&buf, '\n') catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        var len = line.len;
        while (len > 0 and (line[len - 1] == '\r' or line[len - 1] == ' ')) len -= 1;
        if (len == 0) continue;
        if (num_docs >= MAX_DOCS) break;
        @memcpy(docs[num_docs][0..len], line[0..len]);
        doc_lens[num_docs] = len;
        num_docs += 1;
    }

    // Fisher-Yates shuffle
    var i: usize = num_docs - 1;
    while (i >= 1) : (i -= 1) {
        const j: usize = @intFromFloat(rngNext() * @as(f64, @floatFromInt(i + 1)));
        const tmp = docs[i];
        const tmp_len = doc_lens[i];
        docs[i] = docs[j];
        doc_lens[i] = doc_lens[j];
        docs[j] = tmp;
        doc_lens[j] = tmp_len;
        if (i == 0) break;
    }

    // Build char set
    var seen = [_]bool{false} ** 256;
    for (0..num_docs) |d| {
        for (0..doc_lens[d]) |c| { seen[docs[d][c]] = true; }
    }
    num_uchars = 0;
    for (0..256) |c| {
        if (seen[c]) { uchars[num_uchars] = @intCast(c); num_uchars += 1; }
    }
    @memset(&char_to_idx, 0);
    for (0..num_uchars) |ci| { char_to_idx[uchars[ci]] = ci; }
    bos = num_uchars;
    vocab_size = num_uchars + 1;
}

// =========================================================================
// Main
// =========================================================================

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    try loadDataset();
    try stdout.print("num docs: {d}\n", .{num_docs});
    try stdout.print("vocab size: {d}\n", .{vocab_size});

    // Init params (alphabetical key order)
    num_params = 0;
    for (0..N_LAYER) |li| {
        allocMatrix(&attn_wk[li], N_EMBD, N_EMBD);
        allocMatrix(&attn_wo[li], N_EMBD, N_EMBD);
        allocMatrix(&attn_wq[li], N_EMBD, N_EMBD);
        allocMatrix(&attn_wv[li], N_EMBD, N_EMBD);
        allocMatrix(&mlp_fc1[li], 4 * N_EMBD, N_EMBD);
        allocMatrix(&mlp_fc2[li], N_EMBD, 4 * N_EMBD);
    }
    for (0..vocab_size) |r| {
        for (0..N_EMBD) |c| {
            const idx = valNew(gaussRandom(0.08));
            lm_head[r][c] = idx;
            params_buf[num_params] = idx;
            num_params += 1;
        }
    }
    for (0..BLOCK_SIZE) |r| {
        for (0..N_EMBD) |c| {
            const idx = valNew(gaussRandom(0.08));
            wpe[r][c] = idx;
            params_buf[num_params] = idx;
            num_params += 1;
        }
    }
    for (0..vocab_size) |r| {
        for (0..N_EMBD) |c| {
            const idx = valNew(gaussRandom(0.08));
            wte[r][c] = idx;
            params_buf[num_params] = idx;
            num_params += 1;
        }
    }
    num_params_base = arena_counter;
    try stdout.print("num params: {d}\n", .{num_params});

    // Adam state
    var m_adam = [_]f64{0} ** 8192;
    var v_adam = [_]f64{0} ** 8192;

    // Training
    for (0..NUM_STEPS) |step| {
        arena_counter = num_params_base;

        const doc_idx = step % num_docs;
        const dlen = doc_lens[doc_idx];

        var tokens: [BLOCK_SIZE + 2]usize = undefined;
        tokens[0] = bos;
        for (0..dlen) |ci| { tokens[ci + 1] = char_to_idx[docs[doc_idx][ci]]; }
        tokens[dlen + 1] = bos;

        const n = if (BLOCK_SIZE < dlen + 1) BLOCK_SIZE else dlen + 1;

        kvReset();
        var losses: [BLOCK_SIZE]usize = undefined;

        for (0..n) |p| {
            var logits: [MAX_CHARS]usize = undefined;
            gptForward(logits[0..vocab_size], tokens[p], p, vocab_size);
            var probs: [MAX_CHARS]usize = undefined;
            softmaxFwd(&probs, logits[0..vocab_size], vocab_size);
            losses[p] = valNeg(valLogV(probs[tokens[p + 1]]));
        }

        const loss = valDivC(sumValues(losses[0..n]), @as(f64, @floatFromInt(n)));
        backward(loss);

        const lr_t = LR * (1.0 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(NUM_STEPS)));
        for (0..num_params) |pi| {
            const pidx = params_buf[pi];
            const g = arena[pidx].grad;
            m_adam[pi] = BETA1 * m_adam[pi] + (1 - BETA1) * g;
            v_adam[pi] = BETA2 * v_adam[pi] + (1 - BETA2) * g * g;
            const step_f: f64 = @floatFromInt(step + 1);
            const m_hat = m_adam[pi] / (1.0 - math.pow(f64, BETA1, step_f));
            const v_hat = v_adam[pi] / (1.0 - math.pow(f64, BETA2, step_f));
            arena[pidx].data -= lr_t * m_hat / (@sqrt(v_hat) + EPS_ADAM);
            arena[pidx].grad = 0;
        }

        try stdout.print("step {d:>4} / {d:>4} | loss {d:.4}\n", .{ step + 1, NUM_STEPS, arena[loss].data });
    }

    // Inference
    try stdout.print("\n--- inference (new, hallucinated names) ---\n", .{});

    for (0..20) |si| {
        arena_counter = num_params_base;
        kvReset();

        var tok = bos;
        var sample: [BLOCK_SIZE]u8 = undefined;
        var slen: usize = 0;

        for (0..BLOCK_SIZE) |p| {
            var logits: [MAX_CHARS]usize = undefined;
            gptForward(logits[0..vocab_size], tok, p, vocab_size);

            var scaled: [MAX_CHARS]usize = undefined;
            for (0..vocab_size) |ci| { scaled[ci] = valDivC(logits[ci], TEMPERATURE); }

            var probs: [MAX_CHARS]usize = undefined;
            softmaxFwd(&probs, scaled[0..vocab_size], vocab_size);

            var pd: [MAX_CHARS]f64 = undefined;
            for (0..vocab_size) |ci| { pd[ci] = arena[probs[ci]].data; }

            tok = weightedChoice(pd[0..vocab_size]);
            if (tok == bos) break;
            sample[slen] = uchars[tok];
            slen += 1;
        }

        try stdout.print("sample {d:>2}: {s}\n", .{ si + 1, sample[0..slen] });
    }
}
