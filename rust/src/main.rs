// microgpt — pure Rust, zero dependencies.
// Ported from Karpathy's microgpt.py via the Swift port.

use std::collections::HashSet;
use std::fs;
use std::process::Command;

// =========================================================================
// Autograd engine (arena-based)
// =========================================================================

struct ValueNode {
    data: f64,
    grad: f64,
    children: [i32; 2],
    local_grads: [f64; 2],
    n_children: u8,
}

struct Tape {
    nodes: Vec<ValueNode>,
}

impl Tape {
    fn new() -> Self { Tape { nodes: Vec::with_capacity(2_000_000) } }

    fn val(&mut self, data: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ValueNode { data, grad: 0.0, children: [-1, -1], local_grads: [0.0, 0.0], n_children: 0 });
        idx
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.nodes[a].data + self.nodes[b].data;
        let idx = self.val(data);
        self.nodes[idx].children = [a as i32, b as i32];
        self.nodes[idx].local_grads = [1.0, 1.0];
        self.nodes[idx].n_children = 2;
        idx
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let ad = self.nodes[a].data;
        let bd = self.nodes[b].data;
        let idx = self.val(ad * bd);
        self.nodes[idx].children = [a as i32, b as i32];
        self.nodes[idx].local_grads = [bd, ad];
        self.nodes[idx].n_children = 2;
        idx
    }

    fn add_const(&mut self, a: usize, c: f64) -> usize { let b = self.val(c); self.add(a, b) }
    fn mul_const(&mut self, a: usize, c: f64) -> usize { let b = self.val(c); self.mul(a, b) }
    fn neg(&mut self, a: usize) -> usize { self.mul_const(a, -1.0) }

    fn pow_const(&mut self, a: usize, n: f64) -> usize {
        let ad = self.nodes[a].data;
        let idx = self.val(ad.powf(n));
        self.nodes[idx].children[0] = a as i32;
        self.nodes[idx].local_grads[0] = n * ad.powf(n - 1.0);
        self.nodes[idx].n_children = 1;
        idx
    }

    fn div(&mut self, a: usize, b: usize) -> usize { let bp = self.pow_const(b, -1.0); self.mul(a, bp) }
    fn div_const(&mut self, a: usize, c: f64) -> usize { self.mul_const(a, 1.0 / c) }

    fn exp_v(&mut self, a: usize) -> usize {
        let e = self.nodes[a].data.exp();
        let idx = self.val(e);
        self.nodes[idx].children[0] = a as i32;
        self.nodes[idx].local_grads[0] = e;
        self.nodes[idx].n_children = 1;
        idx
    }

    fn log_v(&mut self, a: usize) -> usize {
        let ad = self.nodes[a].data;
        let idx = self.val(ad.ln());
        self.nodes[idx].children[0] = a as i32;
        self.nodes[idx].local_grads[0] = 1.0 / ad;
        self.nodes[idx].n_children = 1;
        idx
    }

    fn relu(&mut self, a: usize) -> usize {
        let ad = self.nodes[a].data;
        let idx = self.val(if ad > 0.0 { ad } else { 0.0 });
        self.nodes[idx].children[0] = a as i32;
        self.nodes[idx].local_grads[0] = if ad > 0.0 { 1.0 } else { 0.0 };
        self.nodes[idx].n_children = 1;
        idx
    }

    fn backward(&mut self, root: usize) {
        // Iterative topological sort
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut topo = Vec::with_capacity(n);
        let mut stack: Vec<(usize, u8)> = vec![(root, 0)];

        while let Some((node, ci)) = stack.last_mut() {
            let node = *node;
            if visited[node] { stack.pop(); continue; }
            let nc = self.nodes[node].n_children;
            if *ci < nc {
                let child = self.nodes[node].children[*ci as usize] as usize;
                *ci += 1;
                if !visited[child] { stack.push((child, 0)); }
            } else {
                visited[node] = true;
                topo.push(node);
                stack.pop();
            }
        }

        for &i in &topo { self.nodes[i].grad = 0.0; }
        self.nodes[root].grad = 1.0;

        for &v in topo.iter().rev() {
            let g = self.nodes[v].grad;
            for c in 0..self.nodes[v].n_children as usize {
                let child = self.nodes[v].children[c] as usize;
                let lg = self.nodes[v].local_grads[c];
                self.nodes[child].grad += lg * g;
            }
        }
    }

    fn sum_values(&mut self, vals: &[usize]) -> usize {
        let mut s = vals[0];
        for &v in &vals[1..] { s = self.add(s, v); }
        s
    }

    fn data(&self, i: usize) -> f64 { self.nodes[i].data }
}

// =========================================================================
// RNG (xorshift64)
// =========================================================================

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }

    fn gauss(&mut self, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn weighted_choice(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, &wi) in w.iter().enumerate() { r -= wi; if r <= 0.0 { return i; } }
        w.len() - 1
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn matrix(tape: &mut Tape, rng: &mut Rng, nout: usize, nin: usize) -> Vec<Vec<usize>> {
    (0..nout).map(|_| (0..nin).map(|_| tape.val(rng.gauss(0.08))).collect()).collect()
}

fn linear(tape: &mut Tape, x: &[usize], w: &[Vec<usize>]) -> Vec<usize> {
    w.iter().map(|row| {
        let prods: Vec<usize> = row.iter().zip(x).map(|(&wi, &xi)| tape.mul(wi, xi)).collect();
        tape.sum_values(&prods)
    }).collect()
}

fn softmax(tape: &mut Tape, logits: &[usize]) -> Vec<usize> {
    let mx = logits.iter().map(|&i| tape.data(i)).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<usize> = logits.iter().map(|&i| { let s = tape.add_const(i, -mx); tape.exp_v(s) }).collect();
    let total = tape.sum_values(&exps);
    exps.iter().map(|&e| tape.div(e, total)).collect()
}

fn rmsnorm(tape: &mut Tape, x: &[usize]) -> Vec<usize> {
    let sq: Vec<usize> = x.iter().map(|&i| tape.mul(i, i)).collect();
    let ms = tape.sum_values(&sq);
    let ms = tape.div_const(ms, x.len() as f64);
    let ms = tape.add_const(ms, 1e-5);
    let scale = tape.pow_const(ms, -0.5);
    x.iter().map(|&i| tape.mul(i, scale)).collect()
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    let mut rng = Rng::new(42);

    // Load dataset
    if !std::path::Path::new("input.txt").exists() {
        println!("Downloading names dataset...");
        Command::new("curl").args(["-sL",
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
            "-o", "input.txt"]).status().expect("curl failed");
    }

    let content = fs::read_to_string("input.txt").expect("cannot read input.txt");
    let mut docs: Vec<&str> = content.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    for i in (1..docs.len()).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        docs.swap(i, j);
    }
    println!("num docs: {}", docs.len());

    let mut chars_set: HashSet<char> = HashSet::new();
    for doc in &docs { for c in doc.chars() { chars_set.insert(c); } }
    let mut uchars: Vec<char> = chars_set.into_iter().collect();
    uchars.sort();
    let char_to_idx: std::collections::HashMap<char, usize> =
        uchars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    // Model parameters
    let n_embd = 16usize;
    let n_head = 4usize;
    let n_layer = 1usize;
    let block_size = 16usize;
    let head_dim = n_embd / n_head;

    let mut tape = Tape::new();
    let mut sd: std::collections::HashMap<String, Vec<Vec<usize>>> = std::collections::HashMap::new();

    sd.insert("wte".into(), matrix(&mut tape, &mut rng, vocab_size, n_embd));
    sd.insert("wpe".into(), matrix(&mut tape, &mut rng, block_size, n_embd));
    sd.insert("lm_head".into(), matrix(&mut tape, &mut rng, vocab_size, n_embd));
    for i in 0..n_layer {
        sd.insert(format!("layer{}.attn_wq", i), matrix(&mut tape, &mut rng, n_embd, n_embd));
        sd.insert(format!("layer{}.attn_wk", i), matrix(&mut tape, &mut rng, n_embd, n_embd));
        sd.insert(format!("layer{}.attn_wv", i), matrix(&mut tape, &mut rng, n_embd, n_embd));
        sd.insert(format!("layer{}.attn_wo", i), matrix(&mut tape, &mut rng, n_embd, n_embd));
        sd.insert(format!("layer{}.mlp_fc1", i), matrix(&mut tape, &mut rng, 4 * n_embd, n_embd));
        sd.insert(format!("layer{}.mlp_fc2", i), matrix(&mut tape, &mut rng, n_embd, 4 * n_embd));
    }

    let mut sorted_keys: Vec<String> = sd.keys().cloned().collect();
    sorted_keys.sort();
    let params: Vec<usize> = sorted_keys.iter()
        .flat_map(|k| sd[k].iter().flat_map(|row| row.iter().copied()))
        .collect();
    let num_params_base = tape.nodes.len();
    println!("num params: {}", params.len());

    // Training
    let lr = 0.01f64;
    let beta1 = 0.85f64;
    let beta2 = 0.99f64;
    let eps_adam = 1e-8f64;
    let mut m_adam = vec![0.0f64; params.len()];
    let mut v_adam = vec![0.0f64; params.len()];
    let num_steps = 1000;

    for step in 0..num_steps {
        tape.nodes.truncate(num_params_base);

        let doc = docs[step % docs.len()];
        let mut tokens: Vec<usize> = vec![bos];
        for c in doc.chars() { tokens.push(char_to_idx[&c]); }
        tokens.push(bos);
        let n = block_size.min(tokens.len() - 1);

        let mut kv_keys: Vec<Vec<Vec<usize>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut kv_vals: Vec<Vec<Vec<usize>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut losses = Vec::with_capacity(n);

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            // GPT forward
            let tok_emb = &sd["wte"][token_id];
            let pos_emb = &sd["wpe"][pos_id];
            let mut x: Vec<usize> = (0..n_embd).map(|i| tape.add(tok_emb[i], pos_emb[i])).collect();
            x = rmsnorm(&mut tape, &x);

            for li in 0..n_layer {
                let x_res = x.clone();
                x = rmsnorm(&mut tape, &x);
                let wq_key = format!("layer{}.attn_wq", li);
                let wk_key = format!("layer{}.attn_wk", li);
                let wv_key = format!("layer{}.attn_wv", li);
                let wo_key = format!("layer{}.attn_wo", li);
                let q = linear(&mut tape, &x, &sd[&wq_key]);
                let k = linear(&mut tape, &x, &sd[&wk_key]);
                let v = linear(&mut tape, &x, &sd[&wv_key]);

                kv_keys[li].push(k);
                kv_vals[li].push(v);

                let mut x_attn = Vec::with_capacity(n_embd);
                for h in 0..n_head {
                    let hs = h * head_dim;
                    let seq = kv_keys[li].len();
                    let mut attn_logits = Vec::with_capacity(seq);
                    for t in 0..seq {
                        let prods: Vec<usize> = (0..head_dim).map(|d|
                            tape.mul(q[hs + d], kv_keys[li][t][hs + d])
                        ).collect();
                        let s = tape.sum_values(&prods);
                        attn_logits.push(tape.div_const(s, (head_dim as f64).sqrt()));
                    }
                    let aw = softmax(&mut tape, &attn_logits);
                    for d in 0..head_dim {
                        let terms: Vec<usize> = (0..seq).map(|t|
                            tape.mul(aw[t], kv_vals[li][t][hs + d])
                        ).collect();
                        x_attn.push(tape.sum_values(&terms));
                    }
                }

                x = linear(&mut tape, &x_attn, &sd[&wo_key]);
                for i in 0..n_embd { x[i] = tape.add(x[i], x_res[i]); }

                let x_res2 = x.clone();
                x = rmsnorm(&mut tape, &x);
                let fc1_key = format!("layer{}.mlp_fc1", li);
                let fc2_key = format!("layer{}.mlp_fc2", li);
                x = linear(&mut tape, &x, &sd[&fc1_key]);
                for i in 0..x.len() { x[i] = tape.relu(x[i]); }
                x = linear(&mut tape, &x, &sd[&fc2_key]);
                for i in 0..n_embd { x[i] = tape.add(x[i], x_res2[i]); }
            }

            let logits = linear(&mut tape, &x, &sd["lm_head"]);
            let probs = softmax(&mut tape, &logits);
            let lp = tape.log_v(probs[target_id]);
            losses.push(tape.neg(lp));
        }

        let loss_sum = tape.sum_values(&losses);
        let loss = tape.div_const(loss_sum, n as f64);
        tape.backward(loss);

        let lr_t = lr * (1.0 - step as f64 / num_steps as f64);
        for i in 0..params.len() {
            let pi = params[i];
            let g = tape.nodes[pi].grad;
            m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * g;
            v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * g * g;
            let m_hat = m_adam[i] / (1.0 - beta1.powi(step as i32 + 1));
            let v_hat = v_adam[i] / (1.0 - beta2.powi(step as i32 + 1));
            tape.nodes[pi].data -= lr_t * m_hat / (v_hat.sqrt() + eps_adam);
            tape.nodes[pi].grad = 0.0;
        }

        println!("step {:4} / {:4} | loss {:.4}", step + 1, num_steps, tape.data(loss));
    }

    // Inference
    let temperature = 0.5f64;
    println!("\n--- inference (new, hallucinated names) ---");

    for si in 0..20 {
        tape.nodes.truncate(num_params_base);
        let mut kv_keys: Vec<Vec<Vec<usize>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut kv_vals: Vec<Vec<Vec<usize>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..block_size {
            let tok_emb = &sd["wte"][token_id];
            let pos_emb = &sd["wpe"][pos_id];
            let mut x: Vec<usize> = (0..n_embd).map(|i| tape.add(tok_emb[i], pos_emb[i])).collect();
            x = rmsnorm(&mut tape, &x);

            for li in 0..n_layer {
                let x_res = x.clone();
                x = rmsnorm(&mut tape, &x);
                let q = linear(&mut tape, &x, &sd[&format!("layer{}.attn_wq", li)]);
                let k = linear(&mut tape, &x, &sd[&format!("layer{}.attn_wk", li)]);
                let v = linear(&mut tape, &x, &sd[&format!("layer{}.attn_wv", li)]);
                kv_keys[li].push(k); kv_vals[li].push(v);

                let mut x_attn = Vec::with_capacity(n_embd);
                for h in 0..n_head {
                    let hs = h * head_dim;
                    let seq = kv_keys[li].len();
                    let mut al = Vec::with_capacity(seq);
                    for t in 0..seq {
                        let prods: Vec<usize> = (0..head_dim).map(|d| tape.mul(q[hs+d], kv_keys[li][t][hs+d])).collect();
                        let s = tape.sum_values(&prods);
                        al.push(tape.div_const(s, (head_dim as f64).sqrt()));
                    }
                    let aw = softmax(&mut tape, &al);
                    for d in 0..head_dim {
                        let terms: Vec<usize> = (0..seq).map(|t| tape.mul(aw[t], kv_vals[li][t][hs+d])).collect();
                        x_attn.push(tape.sum_values(&terms));
                    }
                }
                x = linear(&mut tape, &x_attn, &sd[&format!("layer{}.attn_wo", li)]);
                for i in 0..n_embd { x[i] = tape.add(x[i], x_res[i]); }

                let x_res2 = x.clone();
                x = rmsnorm(&mut tape, &x);
                x = linear(&mut tape, &x, &sd[&format!("layer{}.mlp_fc1", li)]);
                for i in 0..x.len() { x[i] = tape.relu(x[i]); }
                x = linear(&mut tape, &x, &sd[&format!("layer{}.mlp_fc2", li)]);
                for i in 0..n_embd { x[i] = tape.add(x[i], x_res2[i]); }
            }

            let logits = linear(&mut tape, &x, &sd["lm_head"]);
            let scaled: Vec<usize> = logits.iter().map(|&l| tape.div_const(l, temperature)).collect();
            let probs = softmax(&mut tape, &scaled);
            let prob_data: Vec<f64> = probs.iter().map(|&p| tape.data(p)).collect();
            token_id = rng.weighted_choice(&prob_data);
            if token_id == bos { break; }
            sample.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", si + 1, sample);
    }
}
