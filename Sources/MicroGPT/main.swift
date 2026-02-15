// microgpt.swift
// The most atomic way to train and inference a GPT in pure Swift.
// Ported from Karpathy's microgpt.py — zero dependencies beyond Foundation.

import Foundation

// ---------------------------------------------------------------------------
// Math helpers (avoid name collision with Value instance methods)
// ---------------------------------------------------------------------------

private func _log(_ x: Double) -> Double { log(x) }
private func _exp(_ x: Double) -> Double { exp(x) }
private func _pow(_ base: Double, _ exp: Double) -> Double { pow(base, exp) }

// ---------------------------------------------------------------------------
// Seeded RNG
// ---------------------------------------------------------------------------

srand48(42)

func gaussRandom(std: Double) -> Double {
    let u1 = drand48()
    let u2 = drand48()
    return std * sqrt(-2.0 * _log(u1)) * cos(2.0 * .pi * u2)
}

func weightedRandomChoice(_ weights: [Double]) -> Int {
    let total = weights.reduce(0, +)
    var r = drand48() * total
    for (i, w) in weights.enumerated() {
        r -= w
        if r <= 0 { return i }
    }
    return weights.count - 1
}

// ---------------------------------------------------------------------------
// Autograd engine
// ---------------------------------------------------------------------------

final class Value {
    var data: Double
    var grad: Double = 0
    let children: [Value]
    let localGrads: [Double]

    init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
        self.data = data
        self.children = children
        self.localGrads = localGrads
    }

    func log() -> Value {
        Value(_log(data), children: [self], localGrads: [1.0 / data])
    }

    func exp() -> Value {
        let e = _exp(data)
        return Value(e, children: [self], localGrads: [e])
    }

    func relu() -> Value {
        Value(Swift.max(0, data), children: [self], localGrads: [data > 0 ? 1.0 : 0.0])
    }

    func pow(_ n: Double) -> Value {
        Value(_pow(data, n), children: [self], localGrads: [n * _pow(data, n - 1)])
    }

    func backward() {
        var topo: [Value] = []
        var visited = Set<ObjectIdentifier>()

        func buildTopo(_ v: Value) {
            let id = ObjectIdentifier(v)
            guard !visited.contains(id) else { return }
            visited.insert(id)
            for child in v.children {
                buildTopo(child)
            }
            topo.append(v)
        }

        buildTopo(self)
        self.grad = 1.0

        for v in topo.reversed() {
            for (child, lg) in zip(v.children, v.localGrads) {
                child.grad += lg * v.grad
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

func + (a: Value, b: Value) -> Value {
    Value(a.data + b.data, children: [a, b], localGrads: [1, 1])
}
func + (a: Value, b: Double) -> Value { a + Value(b) }
func + (a: Double, b: Value) -> Value { Value(a) + b }

func * (a: Value, b: Value) -> Value {
    Value(a.data * b.data, children: [a, b], localGrads: [b.data, a.data])
}
func * (a: Value, b: Double) -> Value { a * Value(b) }
func * (a: Double, b: Value) -> Value { Value(a) * b }

prefix func - (v: Value) -> Value { v * (-1.0) }
func - (a: Value, b: Value) -> Value { a + (-b) }
func - (a: Value, b: Double) -> Value { a + (-b) }
func - (a: Double, b: Value) -> Value { Value(a) + (-b) }

func / (a: Value, b: Value) -> Value { a * b.pow(-1) }
func / (a: Value, b: Double) -> Value { a * (1.0 / b) }
func / (a: Double, b: Value) -> Value { Value(a) * b.pow(-1) }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func sumValues(_ values: [Value]) -> Value {
    values.dropFirst().reduce(values[0]) { $0 + $1 }
}

func matrix(_ nout: Int, _ nin: Int, std: Double = 0.08) -> [[Value]] {
    (0..<nout).map { _ in (0..<nin).map { _ in Value(gaussRandom(std: std)) } }
}

// ===========================================================================
// Load dataset
// ===========================================================================

let inputPath = "input.txt"
if !FileManager.default.fileExists(atPath: inputPath) {
    print("Downloading names dataset...")
    let url = URL(string: "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt")!
    let data = try Data(contentsOf: url)
    try data.write(to: URL(fileURLWithPath: inputPath))
}

let content = try String(contentsOfFile: inputPath, encoding: .utf8)
var docs = content
    .split(separator: "\n")
    .map { $0.trimmingCharacters(in: .whitespaces) }
    .filter { !$0.isEmpty }

// Seeded Fisher-Yates shuffle
for i in stride(from: docs.count - 1, through: 1, by: -1) {
    let j = Int(drand48() * Double(i + 1))
    docs.swapAt(i, j)
}
print("num docs: \(docs.count)")

// ===========================================================================
// Tokenizer
// ===========================================================================

let uchars: [Character] = Set(docs.joined()).sorted()
let charToIdx = Dictionary(uniqueKeysWithValues: uchars.enumerated().map { ($1, $0) })
let BOS = uchars.count
let vocabSize = uchars.count + 1
print("vocab size: \(vocabSize)")

// ===========================================================================
// Model parameters
// ===========================================================================

let nEmbd = 16
let nHead = 4
let nLayer = 1
let blockSize = 16
let headDim = nEmbd / nHead

var stateDict: [String: [[Value]]] = [
    "wte": matrix(vocabSize, nEmbd),
    "wpe": matrix(blockSize, nEmbd),
    "lm_head": matrix(vocabSize, nEmbd),
]

for i in 0..<nLayer {
    stateDict["layer\(i).attn_wq"] = matrix(nEmbd, nEmbd)
    stateDict["layer\(i).attn_wk"] = matrix(nEmbd, nEmbd)
    stateDict["layer\(i).attn_wv"] = matrix(nEmbd, nEmbd)
    stateDict["layer\(i).attn_wo"] = matrix(nEmbd, nEmbd)
    stateDict["layer\(i).mlp_fc1"] = matrix(4 * nEmbd, nEmbd)
    stateDict["layer\(i).mlp_fc2"] = matrix(nEmbd, 4 * nEmbd)
}

let params: [Value] = stateDict
    .sorted { $0.key < $1.key }
    .flatMap { $0.value.flatMap { $0 } }
print("num params: \(params.count)")

// ===========================================================================
// Model architecture
// ===========================================================================

func linear(_ x: [Value], _ w: [[Value]]) -> [Value] {
    w.map { row in sumValues(zip(row, x).map { $0 * $1 }) }
}

func softmax(_ logits: [Value]) -> [Value] {
    let maxVal = logits.map(\.data).max()!
    let exps = logits.map { ($0 - maxVal).exp() }
    let total = sumValues(exps)
    return exps.map { $0 / total }
}

func rmsnorm(_ x: [Value]) -> [Value] {
    let ms = sumValues(x.map { $0 * $0 }) / Double(x.count)
    let scale = (ms + 1e-5).pow(-0.5)
    return x.map { $0 * scale }
}

func gpt(
    _ tokenId: Int,
    _ posId: Int,
    _ keys: inout [[[Value]]],
    _ values: inout [[[Value]]]
) -> [Value] {
    let tokEmb = stateDict["wte"]![tokenId]
    let posEmb = stateDict["wpe"]![posId]
    var x = zip(tokEmb, posEmb).map { $0 + $1 }
    x = rmsnorm(x)

    for li in 0..<nLayer {
        // --- multi-head attention ---
        let xResidual = x
        x = rmsnorm(x)
        let q = linear(x, stateDict["layer\(li).attn_wq"]!)
        let k = linear(x, stateDict["layer\(li).attn_wk"]!)
        let v = linear(x, stateDict["layer\(li).attn_wv"]!)

        keys[li].append(k)
        values[li].append(v)

        var xAttn: [Value] = []
        for h in 0..<nHead {
            let hs = h * headDim
            let qH = Array(q[hs..<hs + headDim])
            let kH = keys[li].map { Array($0[hs..<hs + headDim]) }
            let vH = values[li].map { Array($0[hs..<hs + headDim]) }

            let attnLogits = kH.map { kT in
                sumValues(zip(qH, kT).map { $0 * $1 }) / sqrt(Double(headDim))
            }
            let attnWeights = softmax(attnLogits)

            let headOut = (0..<headDim).map { j in
                sumValues((0..<vH.count).map { t in attnWeights[t] * vH[t][j] })
            }
            xAttn.append(contentsOf: headOut)
        }

        x = linear(xAttn, stateDict["layer\(li).attn_wo"]!)
        x = zip(x, xResidual).map { $0 + $1 }

        // --- MLP ---
        let xResidual2 = x
        x = rmsnorm(x)
        x = linear(x, stateDict["layer\(li).mlp_fc1"]!)
        x = x.map { $0.relu() }
        x = linear(x, stateDict["layer\(li).mlp_fc2"]!)
        x = zip(x, xResidual2).map { $0 + $1 }
    }

    return linear(x, stateDict["lm_head"]!)
}

// ===========================================================================
// Training (Adam)
// ===========================================================================

let learningRate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let epsAdam = 1e-8
var mAdam = [Double](repeating: 0, count: params.count)
var vAdam = [Double](repeating: 0, count: params.count)
let numSteps = 1000

for step in 0..<numSteps {
    let doc = docs[step % docs.count]
    let tokens = [BOS] + doc.map { charToIdx[$0]! } + [BOS]
    let n = min(blockSize, tokens.count - 1)

    var keys = (0..<nLayer).map { _ in [[Value]]() }
    var vals = (0..<nLayer).map { _ in [[Value]]() }
    var losses: [Value] = []

    for posId in 0..<n {
        let tokenId = tokens[posId]
        let targetId = tokens[posId + 1]
        let logits = gpt(tokenId, posId, &keys, &vals)
        let probs = softmax(logits)
        losses.append(-probs[targetId].log())
    }

    let loss = sumValues(losses) / Double(n)
    loss.backward()

    let lrT = learningRate * (1.0 - Double(step) / Double(numSteps))

    for i in 0..<params.count {
        mAdam[i] = beta1 * mAdam[i] + (1 - beta1) * params[i].grad
        vAdam[i] = beta2 * vAdam[i] + (1 - beta2) * params[i].grad * params[i].grad
        let mHat = mAdam[i] / (1 - _pow(beta1, Double(step + 1)))
        let vHat = vAdam[i] / (1 - _pow(beta2, Double(step + 1)))
        params[i].data -= lrT * mHat / (sqrt(vHat) + epsAdam)
        params[i].grad = 0
    }

    print(String(format: "step %4d / %4d | loss %.4f", step + 1, numSteps, loss.data))
}

// ===========================================================================
// Inference
// ===========================================================================

let temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")

for sampleIdx in 0..<20 {
    var keys = (0..<nLayer).map { _ in [[Value]]() }
    var vals = (0..<nLayer).map { _ in [[Value]]() }
    var tokenId = BOS
    var sample: [Character] = []

    for posId in 0..<blockSize {
        let logits = gpt(tokenId, posId, &keys, &vals)
        let probs = softmax(logits.map { $0 / temperature })
        tokenId = weightedRandomChoice(probs.map(\.data))

        if tokenId == BOS { break }
        sample.append(uchars[tokenId])
    }

    print(String(format: "sample %2d: %@", sampleIdx + 1, String(sample)))
}
