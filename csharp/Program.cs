// MicroGPT — pure C#, zero external dependencies.
// Ported from Karpathy's microgpt.py via the Swift port.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;

class Value {
    public double Data;
    public double Grad;
    public readonly Value[] Children;
    public readonly double[] LocalGrads;

    public Value(double data) { Data = data; Children = Array.Empty<Value>(); LocalGrads = Array.Empty<double>(); }
    Value(double data, Value[] ch, double[] lg) { Data = data; Children = ch; LocalGrads = lg; }

    public static Value operator +(Value a, Value b) => new(a.Data + b.Data, new[]{a,b}, new[]{1.0,1.0});
    public static Value operator +(Value a, double b) => a + new Value(b);
    public static Value operator +(double a, Value b) => new Value(a) + b;
    public static Value operator *(Value a, Value b) => new(a.Data * b.Data, new[]{a,b}, new[]{b.Data,a.Data});
    public static Value operator *(Value a, double b) => a * new Value(b);
    public static Value operator *(double a, Value b) => new Value(a) * b;
    public static Value operator -(Value a) => a * -1.0;
    public static Value operator -(Value a, Value b) => a + (-b);
    public static Value operator -(Value a, double b) => a + (-b);
    public static Value operator /(Value a, Value b) => a * b.Pow(-1);
    public static Value operator /(Value a, double b) => a * (1.0 / b);

    public Value Log() => new(Math.Log(Data), new[]{this}, new[]{1.0/Data});
    public Value Exp() { var e = Math.Exp(Data); return new(e, new[]{this}, new[]{e}); }
    public Value Relu() => new(Math.Max(0, Data), new[]{this}, new[]{Data > 0 ? 1.0 : 0.0});
    public Value Pow(double n) => new(Math.Pow(Data, n), new[]{this}, new[]{n * Math.Pow(Data, n - 1)});

    public void Backward() {
        var topo = new List<Value>();
        var visited = new HashSet<Value>(ReferenceEqualityComparer.Instance);
        void Build(Value v) {
            if (!visited.Add(v)) return;
            foreach (var c in v.Children) Build(c);
            topo.Add(v);
        }
        Build(this);
        Grad = 1.0;
        for (int i = topo.Count - 1; i >= 0; i--) {
            var v = topo[i];
            for (int j = 0; j < v.Children.Length; j++)
                v.Children[j].Grad += v.LocalGrads[j] * v.Grad;
        }
    }
}

class Program {
    static Random rng = new(42);

    static double GaussRandom(double std) {
        double u1 = rng.NextDouble(), u2 = rng.NextDouble();
        return std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    static int WeightedChoice(double[] w) {
        double total = w.Sum(), r = rng.NextDouble() * total;
        for (int i = 0; i < w.Length; i++) { r -= w[i]; if (r <= 0) return i; }
        return w.Length - 1;
    }

    static Value Sum(Value[] v) { var s = v[0]; for (int i = 1; i < v.Length; i++) s = s + v[i]; return s; }

    static Value[][] Matrix(int nout, int nin) {
        var m = new Value[nout][];
        for (int i = 0; i < nout; i++) {
            m[i] = new Value[nin];
            for (int j = 0; j < nin; j++) m[i][j] = new Value(GaussRandom(0.08));
        }
        return m;
    }

    static Value[] Linear(Value[] x, Value[][] w) {
        var o = new Value[w.Length];
        for (int i = 0; i < w.Length; i++) {
            var p = new Value[x.Length];
            for (int j = 0; j < x.Length; j++) p[j] = w[i][j] * x[j];
            o[i] = Sum(p);
        }
        return o;
    }

    static Value[] Softmax(Value[] logits) {
        double mx = logits.Max(v => v.Data);
        var exps = logits.Select(v => (v - mx).Exp()).ToArray();
        var total = Sum(exps);
        return exps.Select(e => e / total).ToArray();
    }

    static Value[] RmsNorm(Value[] x) {
        var sq = x.Select(v => v * v).ToArray();
        var ms = Sum(sq) / (double)x.Length;
        var scale = (ms + 1e-5).Pow(-0.5);
        return x.Select(v => v * scale).ToArray();
    }

    const int NEmbd = 16, NHead = 4, NLayer = 1, BlockSize = 16, HeadDim = NEmbd / NHead;
    static Dictionary<string, Value[][]> sd = new();

    static Value[] Gpt(int tokenId, int posId, List<Value[]>[] keys, List<Value[]>[] vals) {
        var tokEmb = sd["wte"][tokenId];
        var posEmb = sd["wpe"][posId];
        var x = new Value[NEmbd];
        for (int i = 0; i < NEmbd; i++) x[i] = tokEmb[i] + posEmb[i];
        x = RmsNorm(x);

        for (int li = 0; li < NLayer; li++) {
            var xRes = (Value[])x.Clone();
            x = RmsNorm(x);
            var q = Linear(x, sd[$"layer{li}.attn_wq"]);
            var k = Linear(x, sd[$"layer{li}.attn_wk"]);
            var v = Linear(x, sd[$"layer{li}.attn_wv"]);
            keys[li].Add(k); vals[li].Add(v);

            var xAttn = new List<Value>();
            for (int h = 0; h < NHead; h++) {
                int hs = h * HeadDim;
                int seq = keys[li].Count;
                var al = new Value[seq];
                double sc = Math.Sqrt(HeadDim);
                for (int t = 0; t < seq; t++) {
                    var prods = new Value[HeadDim];
                    for (int d = 0; d < HeadDim; d++) prods[d] = q[hs + d] * keys[li][t][hs + d];
                    al[t] = Sum(prods) / sc;
                }
                var aw = Softmax(al);
                for (int d = 0; d < HeadDim; d++) {
                    var terms = new Value[seq];
                    for (int t = 0; t < seq; t++) terms[t] = aw[t] * vals[li][t][hs + d];
                    xAttn.Add(Sum(terms));
                }
            }
            x = Linear(xAttn.ToArray(), sd[$"layer{li}.attn_wo"]);
            for (int i = 0; i < NEmbd; i++) x[i] = x[i] + xRes[i];

            var xRes2 = (Value[])x.Clone();
            x = RmsNorm(x);
            x = Linear(x, sd[$"layer{li}.mlp_fc1"]);
            for (int i = 0; i < x.Length; i++) x[i] = x[i].Relu();
            x = Linear(x, sd[$"layer{li}.mlp_fc2"]);
            for (int i = 0; i < NEmbd; i++) x[i] = x[i] + xRes2[i];
        }
        return Linear(x, sd["lm_head"]);
    }

    static void Main() {
        if (!File.Exists("input.txt")) {
            Console.WriteLine("Downloading names dataset...");
            using var client = new HttpClient();
            var data = client.GetStringAsync(
                "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt").Result;
            File.WriteAllText("input.txt", data);
        }

        var docs = File.ReadAllLines("input.txt").Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        for (int i = docs.Count - 1; i >= 1; i--) {
            int j = (int)(rng.NextDouble() * (i + 1));
            (docs[i], docs[j]) = (docs[j], docs[i]);
        }
        Console.WriteLine($"num docs: {docs.Count}");

        var charSet = new SortedSet<char>(docs.SelectMany(d => d));
        var uchars = charSet.ToList();
        var charToIdx = uchars.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);
        int BOS = uchars.Count, vocabSize = uchars.Count + 1;
        Console.WriteLine($"vocab size: {vocabSize}");

        sd["wte"] = Matrix(vocabSize, NEmbd);
        sd["wpe"] = Matrix(BlockSize, NEmbd);
        sd["lm_head"] = Matrix(vocabSize, NEmbd);
        for (int i = 0; i < NLayer; i++) {
            sd[$"layer{i}.attn_wq"] = Matrix(NEmbd, NEmbd);
            sd[$"layer{i}.attn_wk"] = Matrix(NEmbd, NEmbd);
            sd[$"layer{i}.attn_wv"] = Matrix(NEmbd, NEmbd);
            sd[$"layer{i}.attn_wo"] = Matrix(NEmbd, NEmbd);
            sd[$"layer{i}.mlp_fc1"] = Matrix(4 * NEmbd, NEmbd);
            sd[$"layer{i}.mlp_fc2"] = Matrix(NEmbd, 4 * NEmbd);
        }
        var sortedKeys = sd.Keys.OrderBy(k => k).ToList();
        var parameters = sortedKeys.SelectMany(k => sd[k].SelectMany(row => row)).ToArray();
        Console.WriteLine($"num params: {parameters.Length}");

        double lr = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
        var mAdam = new double[parameters.Length];
        var vAdam = new double[parameters.Length];
        int numSteps = 1000;

        for (int step = 0; step < numSteps; step++) {
            var doc = docs[step % docs.Count];
            var tokens = new int[doc.Length + 2];
            tokens[0] = BOS;
            for (int i = 0; i < doc.Length; i++) tokens[i + 1] = charToIdx[doc[i]];
            tokens[doc.Length + 1] = BOS;
            int n = Math.Min(BlockSize, tokens.Length - 1);

            var keys = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var vals = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var losses = new Value[n];

            for (int posId = 0; posId < n; posId++) {
                var logits = Gpt(tokens[posId], posId, keys, vals);
                var probs = Softmax(logits);
                losses[posId] = -(probs[tokens[posId + 1]].Log());
            }
            var loss = Sum(losses) / (double)n;
            loss.Backward();

            double lrT = lr * (1.0 - (double)step / numSteps);
            for (int i = 0; i < parameters.Length; i++) {
                mAdam[i] = beta1 * mAdam[i] + (1 - beta1) * parameters[i].Grad;
                vAdam[i] = beta2 * vAdam[i] + (1 - beta2) * parameters[i].Grad * parameters[i].Grad;
                double mHat = mAdam[i] / (1 - Math.Pow(beta1, step + 1));
                double vHat = vAdam[i] / (1 - Math.Pow(beta2, step + 1));
                parameters[i].Data -= lrT * mHat / (Math.Sqrt(vHat) + epsAdam);
                parameters[i].Grad = 0;
            }
            Console.WriteLine($"step {step+1,4} / {numSteps,4} | loss {loss.Data:F4}");
        }

        double temperature = 0.5;
        Console.WriteLine("\n--- inference (new, hallucinated names) ---");
        for (int si = 0; si < 20; si++) {
            var keys = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            var vals = Enumerable.Range(0, NLayer).Select(_ => new List<Value[]>()).ToArray();
            int tokenId = BOS;
            var sample = new List<char>();
            for (int posId = 0; posId < BlockSize; posId++) {
                var logits = Gpt(tokenId, posId, keys, vals);
                var scaled = logits.Select(l => l / temperature).ToArray();
                var probs = Softmax(scaled);
                var pd = probs.Select(p => p.Data).ToArray();
                tokenId = WeightedChoice(pd);
                if (tokenId == BOS) break;
                sample.Add(uchars[tokenId]);
            }
            Console.WriteLine($"sample {si+1,2}: {new string(sample.ToArray())}");
        }
    }
}
