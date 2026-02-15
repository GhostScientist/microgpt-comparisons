// microgpt.c
// The most atomic way to train and inference a GPT in pure C.
// Ported from Karpathy's microgpt.py — zero dependencies beyond libc + libm.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// -------------------------------------------------------------------------
// Configuration
// -------------------------------------------------------------------------

#define N_EMBD      16
#define N_HEAD      4
#define N_LAYER     1
#define BLOCK_SIZE  16
#define HEAD_DIM    (N_EMBD / N_HEAD)
#define NUM_STEPS   1000
#define LR          0.01
#define BETA1       0.85
#define BETA2       0.99
#define EPS_ADAM    1e-8
#define TEMPERATURE 0.5

#define ARENA_SIZE  2000000
#define MAX_DOCS    50000
#define MAX_DOC_LEN 64
#define MAX_CHARS   128

// -------------------------------------------------------------------------
// Arena-allocated Value (autograd node)
// -------------------------------------------------------------------------

typedef struct {
    double data;
    double grad;
    int children[2];
    double local_grads[2];
    int n_children;
} Value;

static Value arena[ARENA_SIZE];
static int arena_counter = 0;
static int num_params_base = 0;

static int val_new(double data) {
    if (arena_counter >= ARENA_SIZE) {
        fprintf(stderr, "Arena overflow at %d\n", arena_counter);
        exit(1);
    }
    int idx = arena_counter++;
    arena[idx].data = data;
    arena[idx].grad = 0.0;
    arena[idx].children[0] = -1;
    arena[idx].children[1] = -1;
    arena[idx].local_grads[0] = 0.0;
    arena[idx].local_grads[1] = 0.0;
    arena[idx].n_children = 0;
    return idx;
}

static int val_add(int a, int b) {
    int idx = val_new(arena[a].data + arena[b].data);
    arena[idx].children[0] = a;
    arena[idx].children[1] = b;
    arena[idx].local_grads[0] = 1.0;
    arena[idx].local_grads[1] = 1.0;
    arena[idx].n_children = 2;
    return idx;
}

static int val_mul(int a, int b) {
    int idx = val_new(arena[a].data * arena[b].data);
    arena[idx].children[0] = a;
    arena[idx].children[1] = b;
    arena[idx].local_grads[0] = arena[b].data;
    arena[idx].local_grads[1] = arena[a].data;
    arena[idx].n_children = 2;
    return idx;
}

static int val_add_const(int a, double c) {
    return val_add(a, val_new(c));
}

static int val_mul_const(int a, double c) {
    return val_mul(a, val_new(c));
}

static int val_neg(int a) {
    return val_mul_const(a, -1.0);
}

static int val_pow_const(int a, double n) {
    double ad = arena[a].data;
    int idx = val_new(pow(ad, n));
    arena[idx].children[0] = a;
    arena[idx].local_grads[0] = n * pow(ad, n - 1.0);
    arena[idx].n_children = 1;
    return idx;
}

static int val_div(int a, int b) {
    return val_mul(a, val_pow_const(b, -1.0));
}

static int val_div_const(int a, double c) {
    return val_mul_const(a, 1.0 / c);
}

static int val_exp_v(int a) {
    double e = exp(arena[a].data);
    int idx = val_new(e);
    arena[idx].children[0] = a;
    arena[idx].local_grads[0] = e;
    arena[idx].n_children = 1;
    return idx;
}

static int val_log_v(int a) {
    double ad = arena[a].data;
    int idx = val_new(log(ad));
    arena[idx].children[0] = a;
    arena[idx].local_grads[0] = 1.0 / ad;
    arena[idx].n_children = 1;
    return idx;
}

static int val_relu(int a) {
    double ad = arena[a].data;
    int idx = val_new(ad > 0.0 ? ad : 0.0);
    arena[idx].children[0] = a;
    arena[idx].local_grads[0] = ad > 0.0 ? 1.0 : 0.0;
    arena[idx].n_children = 1;
    return idx;
}

// -------------------------------------------------------------------------
// Backward pass: iterative DFS topological sort
// -------------------------------------------------------------------------

typedef struct { int node; int child_idx; } DFSFrame;

static unsigned char *visited_bits = NULL;
static int *topo_order = NULL;
static int topo_count = 0;
static DFSFrame *dfs_stack = NULL;

static inline bool visited_get(int idx) {
    return (visited_bits[idx / 8] >> (idx % 8)) & 1;
}

static inline void visited_set(int idx) {
    visited_bits[idx / 8] |= (unsigned char)(1 << (idx % 8));
}

static void backward(int root) {
    if (!visited_bits) {
        visited_bits = (unsigned char *)calloc((ARENA_SIZE + 7) / 8, 1);
        topo_order = (int *)malloc(ARENA_SIZE * sizeof(int));
        dfs_stack = (DFSFrame *)malloc(ARENA_SIZE * sizeof(DFSFrame));
    }

    memset(visited_bits, 0, ((size_t)arena_counter + 7) / 8);
    topo_count = 0;

    int sp = 0;
    dfs_stack[sp].node = root;
    dfs_stack[sp].child_idx = 0;
    sp++;

    while (sp > 0) {
        DFSFrame *frame = &dfs_stack[sp - 1];
        int node = frame->node;

        if (visited_get(node)) { sp--; continue; }

        if (frame->child_idx < arena[node].n_children) {
            int child = arena[node].children[frame->child_idx];
            frame->child_idx++;
            if (!visited_get(child)) {
                dfs_stack[sp].node = child;
                dfs_stack[sp].child_idx = 0;
                sp++;
            }
        } else {
            visited_set(node);
            topo_order[topo_count++] = node;
            sp--;
        }
    }

    for (int i = 0; i < topo_count; i++)
        arena[topo_order[i]].grad = 0.0;
    arena[root].grad = 1.0;

    for (int i = topo_count - 1; i >= 0; i--) {
        int v = topo_order[i];
        for (int c = 0; c < arena[v].n_children; c++) {
            arena[arena[v].children[c]].grad +=
                arena[v].local_grads[c] * arena[v].grad;
        }
    }
}

// -------------------------------------------------------------------------
// RNG
// -------------------------------------------------------------------------

static double gauss_random(double std_dev) {
    double u1 = drand48(), u2 = drand48();
    return std_dev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int weighted_random_choice(const double *w, int n) {
    double total = 0;
    for (int i = 0; i < n; i++) total += w[i];
    double r = drand48() * total;
    for (int i = 0; i < n; i++) { r -= w[i]; if (r <= 0) return i; }
    return n - 1;
}

// -------------------------------------------------------------------------
// Dataset
// -------------------------------------------------------------------------

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;
static char uchars_set[MAX_CHARS];
static int num_uchars = 0;
static int char_to_idx[256];
static int BOS, vocab_size;

static void load_dataset(void) {
    FILE *f = fopen("input.txt", "r");
    if (!f) {
        printf("Downloading names dataset...\n");
        system("curl -sL "
               "https://raw.githubusercontent.com/karpathy/makemore/"
               "refs/heads/master/names.txt -o input.txt");
        f = fopen("input.txt", "r");
        if (!f) { fprintf(stderr, "Failed to open input.txt\n"); exit(1); }
    }

    char line[256];
    while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1]=='\n' || line[len-1]=='\r' || line[len-1]==' '))
            len--;
        line[len] = '\0';
        if (len == 0) continue;
        if (len >= MAX_DOC_LEN) len = MAX_DOC_LEN - 1;
        memcpy(docs[num_docs], line, (size_t)len + 1);
        num_docs++;
    }
    fclose(f);

    for (int i = num_docs - 1; i >= 1; i--) {
        int j = (int)(drand48() * (double)(i + 1));
        char tmp[MAX_DOC_LEN];
        memcpy(tmp, docs[i], MAX_DOC_LEN);
        memcpy(docs[i], docs[j], MAX_DOC_LEN);
        memcpy(docs[j], tmp, MAX_DOC_LEN);
    }

    bool seen[256]; memset(seen, 0, sizeof(seen));
    for (int d = 0; d < num_docs; d++)
        for (int c = 0; docs[d][c]; c++)
            seen[(unsigned char)docs[d][c]] = true;

    num_uchars = 0;
    for (int c = 0; c < 256; c++)
        if (seen[c]) uchars_set[num_uchars++] = (char)c;

    memset(char_to_idx, 0xff, sizeof(char_to_idx));
    for (int i = 0; i < num_uchars; i++)
        char_to_idx[(unsigned char)uchars_set[i]] = i;

    BOS = num_uchars;
    vocab_size = num_uchars + 1;
}

// -------------------------------------------------------------------------
// Model parameters (arena indices)
// -------------------------------------------------------------------------

static int wte[MAX_CHARS][N_EMBD];
static int wpe[BLOCK_SIZE][N_EMBD];
static int lm_head_w[MAX_CHARS][N_EMBD];
static int attn_wq[N_LAYER][N_EMBD][N_EMBD];
static int attn_wk[N_LAYER][N_EMBD][N_EMBD];
static int attn_wv[N_LAYER][N_EMBD][N_EMBD];
static int attn_wo[N_LAYER][N_EMBD][N_EMBD];
static int mlp_fc1[N_LAYER][4 * N_EMBD][N_EMBD];
static int mlp_fc2[N_LAYER][N_EMBD][4 * N_EMBD];

static int *params_idx = NULL;
static int num_params = 0;

static void alloc_matrix(int *mat, int nout, int nin) {
    for (int i = 0; i < nout; i++)
        for (int j = 0; j < nin; j++) {
            int idx = val_new(gauss_random(0.08));
            mat[i * nin + j] = idx;
            params_idx[num_params++] = idx;
        }
}

static void init_params(void) {
    int total = 0;
    for (int i = 0; i < N_LAYER; i++)
        total += 4 * N_EMBD * N_EMBD + (4*N_EMBD)*N_EMBD + N_EMBD*(4*N_EMBD);
    total += 2 * vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD;

    params_idx = (int *)malloc((size_t)total * sizeof(int));
    num_params = 0;

    // Alphabetical key order
    for (int li = 0; li < N_LAYER; li++) {
        alloc_matrix(&attn_wk[li][0][0], N_EMBD, N_EMBD);
        alloc_matrix(&attn_wo[li][0][0], N_EMBD, N_EMBD);
        alloc_matrix(&attn_wq[li][0][0], N_EMBD, N_EMBD);
        alloc_matrix(&attn_wv[li][0][0], N_EMBD, N_EMBD);
        alloc_matrix(&mlp_fc1[li][0][0], 4 * N_EMBD, N_EMBD);
        alloc_matrix(&mlp_fc2[li][0][0], N_EMBD, 4 * N_EMBD);
    }
    alloc_matrix(&lm_head_w[0][0], vocab_size, N_EMBD);
    alloc_matrix(&wpe[0][0], BLOCK_SIZE, N_EMBD);
    alloc_matrix(&wte[0][0], vocab_size, N_EMBD);

    num_params_base = arena_counter;
    printf("num params: %d\n", num_params);
}

// -------------------------------------------------------------------------
// Forward helpers
// -------------------------------------------------------------------------

static int sum_values(const int *vals, int n) {
    int s = vals[0];
    for (int i = 1; i < n; i++) s = val_add(s, vals[i]);
    return s;
}

static void linear_forward(int *out, const int *w, int nout, int nin, const int *x) {
    for (int i = 0; i < nout; i++) {
        int prods[4 * N_EMBD];
        for (int j = 0; j < nin; j++) prods[j] = val_mul(w[i * nin + j], x[j]);
        out[i] = sum_values(prods, nin);
    }
}

static void softmax_forward(int *out, const int *logits, int n) {
    double mx = arena[logits[0]].data;
    for (int i = 1; i < n; i++)
        if (arena[logits[i]].data > mx) mx = arena[logits[i]].data;

    int exps[MAX_CHARS];
    for (int i = 0; i < n; i++) exps[i] = val_exp_v(val_add_const(logits[i], -mx));
    int total = sum_values(exps, n);
    for (int i = 0; i < n; i++) out[i] = val_div(exps[i], total);
}

static void rmsnorm_forward(int *out, const int *x, int n) {
    int sq[N_EMBD];
    for (int i = 0; i < n; i++) sq[i] = val_mul(x[i], x[i]);
    int ms = val_div_const(sum_values(sq, n), (double)n);
    int scale = val_pow_const(val_add_const(ms, 1e-5), -0.5);
    for (int i = 0; i < n; i++) out[i] = val_mul(x[i], scale);
}

// -------------------------------------------------------------------------
// KV cache
// -------------------------------------------------------------------------

static int kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_values[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_len[N_LAYER];

static void kv_reset(void) {
    for (int l = 0; l < N_LAYER; l++) kv_len[l] = 0;
}

// -------------------------------------------------------------------------
// GPT forward
// -------------------------------------------------------------------------

static void gpt_forward(int *logits_out, int token_id, int pos_id) {
    int x[N_EMBD], xn[N_EMBD];

    for (int i = 0; i < N_EMBD; i++)
        x[i] = val_add(wte[token_id][i], wpe[pos_id][i]);

    rmsnorm_forward(xn, x, N_EMBD);
    memcpy(x, xn, sizeof(x));

    for (int li = 0; li < N_LAYER; li++) {
        int x_res[N_EMBD];
        memcpy(x_res, x, sizeof(x));

        rmsnorm_forward(xn, x, N_EMBD);
        memcpy(x, xn, sizeof(x));

        int q[N_EMBD], k[N_EMBD], v[N_EMBD];
        linear_forward(q, &attn_wq[li][0][0], N_EMBD, N_EMBD, x);
        linear_forward(k, &attn_wk[li][0][0], N_EMBD, N_EMBD, x);
        linear_forward(v, &attn_wv[li][0][0], N_EMBD, N_EMBD, x);

        int t = kv_len[li];
        memcpy(kv_keys[li][t], k, sizeof(k));
        memcpy(kv_values[li][t], v, sizeof(v));
        kv_len[li] = t + 1;

        int x_attn[N_EMBD];
        int ai = 0;
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            int seq = kv_len[li];

            int al[BLOCK_SIZE];
            for (int tt = 0; tt < seq; tt++) {
                int pr[HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d++)
                    pr[d] = val_mul(q[hs + d], kv_keys[li][tt][hs + d]);
                al[tt] = val_div_const(sum_values(pr, HEAD_DIM), sqrt((double)HEAD_DIM));
            }
            int aw[BLOCK_SIZE];
            softmax_forward(aw, al, seq);

            for (int d = 0; d < HEAD_DIM; d++) {
                int terms[BLOCK_SIZE];
                for (int tt = 0; tt < seq; tt++)
                    terms[tt] = val_mul(aw[tt], kv_values[li][tt][hs + d]);
                x_attn[ai++] = sum_values(terms, seq);
            }
        }

        int xp[N_EMBD];
        linear_forward(xp, &attn_wo[li][0][0], N_EMBD, N_EMBD, x_attn);
        for (int i = 0; i < N_EMBD; i++) x[i] = val_add(xp[i], x_res[i]);

        int x_res2[N_EMBD];
        memcpy(x_res2, x, sizeof(x));

        rmsnorm_forward(xn, x, N_EMBD);
        memcpy(x, xn, sizeof(x));

        int h1[4 * N_EMBD];
        linear_forward(h1, &mlp_fc1[li][0][0], 4 * N_EMBD, N_EMBD, x);
        for (int i = 0; i < 4 * N_EMBD; i++) h1[i] = val_relu(h1[i]);

        int h2[N_EMBD];
        linear_forward(h2, &mlp_fc2[li][0][0], N_EMBD, 4 * N_EMBD, h1);
        for (int i = 0; i < N_EMBD; i++) x[i] = val_add(h2[i], x_res2[i]);
    }

    linear_forward(logits_out, &lm_head_w[0][0], vocab_size, N_EMBD, x);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main(void) {
    srand48(42);
    load_dataset();
    printf("num docs: %d\n", num_docs);
    printf("vocab size: %d\n", vocab_size);
    init_params();

    double *m_adam = (double *)calloc((size_t)num_params, sizeof(double));
    double *v_adam = (double *)calloc((size_t)num_params, sizeof(double));

    for (int step = 0; step < NUM_STEPS; step++) {
        arena_counter = num_params_base;

        const char *doc = docs[step % num_docs];
        int dlen = (int)strlen(doc);

        int tokens[BLOCK_SIZE + 2];
        tokens[0] = BOS;
        for (int i = 0; i < dlen; i++)
            tokens[i + 1] = char_to_idx[(unsigned char)doc[i]];
        tokens[dlen + 1] = BOS;

        int n = BLOCK_SIZE < (dlen + 1) ? BLOCK_SIZE : (dlen + 1);

        kv_reset();
        int losses[BLOCK_SIZE];

        for (int p = 0; p < n; p++) {
            int logits[MAX_CHARS];
            gpt_forward(logits, tokens[p], p);
            int probs[MAX_CHARS];
            softmax_forward(probs, logits, vocab_size);
            losses[p] = val_neg(val_log_v(probs[tokens[p + 1]]));
        }

        int loss = val_div_const(sum_values(losses, n), (double)n);
        backward(loss);

        double lr_t = LR * (1.0 - (double)step / (double)NUM_STEPS);
        for (int i = 0; i < num_params; i++) {
            int pi = params_idx[i];
            double g = arena[pi].grad;
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * g;
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * g * g;
            double mh = m_adam[i] / (1 - pow(BETA1, (double)(step + 1)));
            double vh = v_adam[i] / (1 - pow(BETA2, (double)(step + 1)));
            arena[pi].data -= lr_t * mh / (sqrt(vh) + EPS_ADAM);
            arena[pi].grad = 0.0;
        }

        printf("step %4d / %4d | loss %.4f\n", step + 1, NUM_STEPS, arena[loss].data);
    }

    printf("\n--- inference (new, hallucinated names) ---\n");

    for (int si = 0; si < 20; si++) {
        arena_counter = num_params_base;
        kv_reset();

        int tok = BOS;
        char sample[BLOCK_SIZE + 1];
        int slen = 0;

        for (int p = 0; p < BLOCK_SIZE; p++) {
            int logits[MAX_CHARS];
            gpt_forward(logits, tok, p);

            int scaled[MAX_CHARS];
            for (int i = 0; i < vocab_size; i++)
                scaled[i] = val_div_const(logits[i], TEMPERATURE);

            int probs[MAX_CHARS];
            softmax_forward(probs, scaled, vocab_size);

            double pd[MAX_CHARS];
            for (int i = 0; i < vocab_size; i++) pd[i] = arena[probs[i]].data;

            tok = weighted_random_choice(pd, vocab_size);
            if (tok == BOS) break;
            sample[slen++] = uchars_set[tok];
        }
        sample[slen] = '\0';
        printf("sample %2d: %s\n", si + 1, sample);
    }

    free(m_adam); free(v_adam); free(params_idx);
    free(visited_bits); free(topo_order); free(dfs_stack);
    return 0;
}
