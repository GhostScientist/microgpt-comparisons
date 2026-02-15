// MicroGPT.java
// The most atomic way to train and inference a GPT in pure Java.
// Ported from Karpathy's microgpt.py — zero dependencies beyond java.*.

import java.io.*;
import java.net.*;
import java.util.*;

public class MicroGPT {

    static final class Value {
        double data;
        double grad;
        final Value[] children;
        final double[] localGrads;

        Value(double data) {
            this.data = data; this.children = new Value[0]; this.localGrads = new double[0];
        }
        Value(double data, Value[] children, double[] localGrads) {
            this.data = data; this.children = children; this.localGrads = localGrads;
        }

        Value add(Value o)  { return new Value(data + o.data, new Value[]{this, o}, new double[]{1, 1}); }
        Value add(double c) { return add(new Value(c)); }
        Value mul(Value o)  { return new Value(data * o.data, new Value[]{this, o}, new double[]{o.data, data}); }
        Value mul(double c) { return mul(new Value(c)); }
        Value neg()         { return mul(-1.0); }
        Value sub(Value o)  { return add(o.neg()); }
        Value sub(double c) { return add(-c); }
        Value div(Value o)  { return mul(o.powOf(-1)); }
        Value div(double c) { return mul(1.0 / c); }

        Value log() { return new Value(Math.log(data), new Value[]{this}, new double[]{1.0 / data}); }
        Value exp() { double e = Math.exp(data); return new Value(e, new Value[]{this}, new double[]{e}); }
        Value relu() { return new Value(Math.max(0, data), new Value[]{this}, new double[]{data > 0 ? 1.0 : 0.0}); }
        Value powOf(double n) { return new Value(Math.pow(data, n), new Value[]{this}, new double[]{n * Math.pow(data, n - 1)}); }

        void backward() {
            List<Value> topo = new ArrayList<>();
            Set<Value> visited = new HashSet<>();
            buildTopo(this, topo, visited);
            this.grad = 1.0;
            for (int i = topo.size() - 1; i >= 0; i--) {
                Value v = topo.get(i);
                for (int j = 0; j < v.children.length; j++)
                    v.children[j].grad += v.localGrads[j] * v.grad;
            }
        }
        private static void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
            if (visited.contains(v)) return;
            visited.add(v);
            for (Value c : v.children) buildTopo(c, topo, visited);
            topo.add(v);
        }
    }

    static Random rng = new Random(42);

    static double gaussRandom(double std) {
        return std * Math.sqrt(-2.0 * Math.log(rng.nextDouble())) * Math.cos(2.0 * Math.PI * rng.nextDouble());
    }

    static int weightedRandomChoice(double[] w) {
        double total = 0; for (double x : w) total += x;
        double r = rng.nextDouble() * total;
        for (int i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
        return w.length - 1;
    }

    static Value sumValues(Value[] v) {
        Value s = v[0]; for (int i = 1; i < v.length; i++) s = s.add(v[i]); return s;
    }

    static Value[][] matrix(int nout, int nin) {
        Value[][] m = new Value[nout][nin];
        for (int i = 0; i < nout; i++)
            for (int j = 0; j < nin; j++) m[i][j] = new Value(gaussRandom(0.08));
        return m;
    }

    static Value[] linear(Value[] x, Value[][] w) {
        Value[] out = new Value[w.length];
        for (int i = 0; i < w.length; i++) {
            Value[] p = new Value[x.length];
            for (int j = 0; j < x.length; j++) p[j] = w[i][j].mul(x[j]);
            out[i] = sumValues(p);
        }
        return out;
    }

    static Value[] softmax(Value[] logits) {
        double mx = Double.NEGATIVE_INFINITY;
        for (Value v : logits) if (v.data > mx) mx = v.data;
        Value[] exps = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) exps[i] = logits[i].sub(mx).exp();
        Value total = sumValues(exps);
        Value[] probs = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) probs[i] = exps[i].div(total);
        return probs;
    }

    static Value[] rmsnorm(Value[] x) {
        Value[] sq = new Value[x.length];
        for (int i = 0; i < x.length; i++) sq[i] = x[i].mul(x[i]);
        Value ms = sumValues(sq).div((double) x.length);
        Value scale = ms.add(1e-5).powOf(-0.5);
        Value[] out = new Value[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i].mul(scale);
        return out;
    }

    static final int nEmbd = 16, nHead = 4, nLayer = 1, blockSize = 16, headDim = nEmbd / nHead;
    static HashMap<String, Value[][]> stateDict = new HashMap<>();

    @SuppressWarnings("unchecked")
    static Value[] gpt(int tokenId, int posId, List<Value[]>[] keys, List<Value[]>[] values) {
        Value[] tokEmb = stateDict.get("wte")[tokenId];
        Value[] posEmb = stateDict.get("wpe")[posId];
        Value[] x = new Value[nEmbd];
        for (int i = 0; i < nEmbd; i++) x[i] = tokEmb[i].add(posEmb[i]);
        x = rmsnorm(x);

        for (int li = 0; li < nLayer; li++) {
            Value[] xRes = x;
            x = rmsnorm(x);
            Value[] q = linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] v = linear(x, stateDict.get("layer" + li + ".attn_wv"));
            keys[li].add(k); values[li].add(v);

            List<Value> xAttnList = new ArrayList<>();
            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;
                int seq = keys[li].size();
                Value[] attnLogits = new Value[seq];
                double sc = Math.sqrt((double) headDim);
                for (int t = 0; t < seq; t++) {
                    Value[] prods = new Value[headDim];
                    for (int j = 0; j < headDim; j++)
                        prods[j] = q[hs + j].mul(keys[li].get(t)[hs + j]);
                    attnLogits[t] = sumValues(prods).div(sc);
                }
                Value[] aw = softmax(attnLogits);
                for (int j = 0; j < headDim; j++) {
                    Value[] weighted = new Value[seq];
                    for (int t = 0; t < seq; t++) weighted[t] = aw[t].mul(values[li].get(t)[hs + j]);
                    xAttnList.add(sumValues(weighted));
                }
            }
            Value[] xAttn = xAttnList.toArray(new Value[0]);
            x = linear(xAttn, stateDict.get("layer" + li + ".attn_wo"));
            for (int i = 0; i < nEmbd; i++) x[i] = x[i].add(xRes[i]);

            Value[] xRes2 = x;
            x = rmsnorm(x);
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc1"));
            for (int i = 0; i < x.length; i++) x[i] = x[i].relu();
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc2"));
            for (int i = 0; i < nEmbd; i++) x[i] = x[i].add(xRes2[i]);
        }
        return linear(x, stateDict.get("lm_head"));
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        File inputFile = new File("input.txt");
        if (!inputFile.exists()) {
            System.out.println("Downloading names dataset...");
            URL url = new URL("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");
            try (InputStream in = url.openStream(); FileOutputStream out = new FileOutputStream(inputFile)) {
                byte[] buf = new byte[8192]; int n;
                while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            }
        }

        List<String> docs = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            String line;
            while ((line = br.readLine()) != null) { line = line.trim(); if (!line.isEmpty()) docs.add(line); }
        }
        for (int i = docs.size() - 1; i >= 1; i--) {
            int j = (int)(rng.nextDouble() * (i + 1));
            Collections.swap(docs, i, j);
        }
        System.out.println("num docs: " + docs.size());

        TreeSet<Character> charSet = new TreeSet<>();
        for (String doc : docs) for (char c : doc.toCharArray()) charSet.add(c);
        List<Character> uchars = new ArrayList<>(charSet);
        HashMap<Character, Integer> charToIdx = new HashMap<>();
        for (int i = 0; i < uchars.size(); i++) charToIdx.put(uchars.get(i), i);
        int BOS = uchars.size();
        int vocabSize = uchars.size() + 1;
        System.out.println("vocab size: " + vocabSize);

        stateDict.put("wte", matrix(vocabSize, nEmbd));
        stateDict.put("wpe", matrix(blockSize, nEmbd));
        stateDict.put("lm_head", matrix(vocabSize, nEmbd));
        for (int i = 0; i < nLayer; i++) {
            stateDict.put("layer" + i + ".attn_wq", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wk", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wv", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wo", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".mlp_fc1", matrix(4 * nEmbd, nEmbd));
            stateDict.put("layer" + i + ".mlp_fc2", matrix(nEmbd, 4 * nEmbd));
        }
        List<String> sortedKeys = new ArrayList<>(stateDict.keySet());
        Collections.sort(sortedKeys);
        List<Value> paramsList = new ArrayList<>();
        for (String key : sortedKeys) for (Value[] row : stateDict.get(key)) for (Value v : row) paramsList.add(v);
        Value[] params = paramsList.toArray(new Value[0]);
        System.out.println("num params: " + params.length);

        double lr = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
        double[] mAdam = new double[params.length], vAdam = new double[params.length];
        int numSteps = 1000;

        for (int step = 0; step < numSteps; step++) {
            String doc = docs.get(step % docs.size());
            int[] tokens = new int[doc.length() + 2];
            tokens[0] = BOS;
            for (int i = 0; i < doc.length(); i++) tokens[i + 1] = charToIdx.get(doc.charAt(i));
            tokens[doc.length() + 1] = BOS;
            int n = Math.min(blockSize, tokens.length - 1);

            List<Value[]>[] keys = new List[nLayer];
            List<Value[]>[] vals = new List[nLayer];
            for (int li = 0; li < nLayer; li++) { keys[li] = new ArrayList<>(); vals[li] = new ArrayList<>(); }

            Value[] losses = new Value[n];
            for (int posId = 0; posId < n; posId++) {
                Value[] logits = gpt(tokens[posId], posId, keys, vals);
                Value[] probs = softmax(logits);
                losses[posId] = probs[tokens[posId + 1]].log().neg();
            }
            Value loss = sumValues(losses).div((double) n);
            loss.backward();

            double lrT = lr * (1.0 - (double) step / numSteps);
            for (int i = 0; i < params.length; i++) {
                mAdam[i] = beta1 * mAdam[i] + (1 - beta1) * params[i].grad;
                vAdam[i] = beta2 * vAdam[i] + (1 - beta2) * params[i].grad * params[i].grad;
                double mHat = mAdam[i] / (1 - Math.pow(beta1, step + 1));
                double vHat = vAdam[i] / (1 - Math.pow(beta2, step + 1));
                params[i].data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
                params[i].grad = 0;
            }
            System.out.printf("step %4d / %4d | loss %.4f%n", step + 1, numSteps, loss.data);
        }

        double temperature = 0.5;
        System.out.println("\n--- inference (new, hallucinated names) ---");
        for (int si = 0; si < 20; si++) {
            List<Value[]>[] keys = new List[nLayer];
            List<Value[]>[] vals = new List[nLayer];
            for (int li = 0; li < nLayer; li++) { keys[li] = new ArrayList<>(); vals[li] = new ArrayList<>(); }
            int tokenId = BOS;
            StringBuilder sample = new StringBuilder();
            for (int posId = 0; posId < blockSize; posId++) {
                Value[] logits = gpt(tokenId, posId, keys, vals);
                Value[] scaled = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) scaled[i] = logits[i].div(temperature);
                Value[] probs = softmax(scaled);
                double[] pd = new double[probs.length];
                for (int i = 0; i < probs.length; i++) pd[i] = probs[i].data;
                tokenId = weightedRandomChoice(pd);
                if (tokenId == BOS) break;
                sample.append(uchars.get(tokenId));
            }
            System.out.printf("sample %2d: %s%n", si + 1, sample.toString());
        }
    }
}
