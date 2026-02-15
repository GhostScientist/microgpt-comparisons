//
//  MicroGPTRunner.swift
//  microgpt-watchos Watch App
//
//  Port of the training + inference flow from Sources/MicroGPT/main.swift
//  adapted for button-triggered execution with progress callbacks.
//

import Foundation

private func _log(_ x: Double) -> Double { log(x) }
private func _exp(_ x: Double) -> Double { exp(x) }
private func _pow(_ base: Double, _ exp: Double) -> Double { pow(base, exp) }

private func gaussRandom(std: Double) -> Double {
    let u1 = drand48()
    let u2 = drand48()
    return std * sqrt(-2.0 * _log(u1)) * cos(2.0 * .pi * u2)
}

private func weightedRandomChoice(_ weights: [Double]) -> Int {
    let total = weights.reduce(0, +)
    var r = drand48() * total
    for (i, w) in weights.enumerated() {
        r -= w
        if r <= 0 { return i }
    }
    return max(0, weights.count - 1)
}

private final class MGValue {
    var data: Double
    var grad: Double = 0
    let children: [MGValue]
    let localGrads: [Double]

    init(_ data: Double, children: [MGValue] = [], localGrads: [Double] = []) {
        self.data = data
        self.children = children
        self.localGrads = localGrads
    }

    func log() -> MGValue {
        MGValue(_log(data), children: [self], localGrads: [1.0 / data])
    }

    func exp() -> MGValue {
        let e = _exp(data)
        return MGValue(e, children: [self], localGrads: [e])
    }

    func relu() -> MGValue {
        MGValue(Swift.max(0, data), children: [self], localGrads: [data > 0 ? 1.0 : 0.0])
    }

    func pow(_ n: Double) -> MGValue {
        MGValue(_pow(data, n), children: [self], localGrads: [n * _pow(data, n - 1)])
    }

    func backward() {
        var topo: [MGValue] = []
        var visited = Set<ObjectIdentifier>()

        func buildTopo(_ value: MGValue) {
            let id = ObjectIdentifier(value)
            guard !visited.contains(id) else { return }
            visited.insert(id)
            for child in value.children {
                buildTopo(child)
            }
            topo.append(value)
        }

        buildTopo(self)
        grad = 1.0
        for value in topo.reversed() {
            for (child, localGrad) in zip(value.children, value.localGrads) {
                child.grad += localGrad * value.grad
            }
        }
    }
}

private func + (lhs: MGValue, rhs: MGValue) -> MGValue {
    MGValue(lhs.data + rhs.data, children: [lhs, rhs], localGrads: [1, 1])
}
private func + (lhs: MGValue, rhs: Double) -> MGValue { lhs + MGValue(rhs) }
private func + (lhs: Double, rhs: MGValue) -> MGValue { MGValue(lhs) + rhs }

private func * (lhs: MGValue, rhs: MGValue) -> MGValue {
    MGValue(lhs.data * rhs.data, children: [lhs, rhs], localGrads: [rhs.data, lhs.data])
}
private func * (lhs: MGValue, rhs: Double) -> MGValue { lhs * MGValue(rhs) }
private func * (lhs: Double, rhs: MGValue) -> MGValue { MGValue(lhs) * rhs }

private prefix func - (value: MGValue) -> MGValue { value * (-1.0) }
private func - (lhs: MGValue, rhs: MGValue) -> MGValue { lhs + (-rhs) }
private func - (lhs: MGValue, rhs: Double) -> MGValue { lhs + (-rhs) }
private func - (lhs: Double, rhs: MGValue) -> MGValue { MGValue(lhs) + (-rhs) }

private func / (lhs: MGValue, rhs: MGValue) -> MGValue { lhs * rhs.pow(-1) }
private func / (lhs: MGValue, rhs: Double) -> MGValue { lhs * (1.0 / rhs) }
private func / (lhs: Double, rhs: MGValue) -> MGValue { MGValue(lhs) * rhs.pow(-1) }

private func sumValues(_ values: [MGValue]) -> MGValue {
    values.dropFirst().reduce(values[0]) { $0 + $1 }
}

private func matrix(_ nout: Int, _ nin: Int, std: Double = 0.08) -> [[MGValue]] {
    (0..<nout).map { _ in (0..<nin).map { _ in MGValue(gaussRandom(std: std)) } }
}

private enum MicroGPTRunnerError: Error, LocalizedError {
    case noTrainingData
    case noVocabulary

    var errorDescription: String? {
        switch self {
        case .noTrainingData:
            return "No training data was available."
        case .noVocabulary:
            return "Vocabulary could not be constructed."
        }
    }
}

final class MicroGPTRunner {
    struct Config {
        var nEmbd = 16
        var nHead = 4
        var nLayer = 1
        var blockSize = 16
        var learningRate = 0.01
        var beta1 = 0.85
        var beta2 = 0.99
        var epsAdam = 1e-8
        var numSteps = 1000
        var temperature = 0.5
    }

    private let config: Config
    private let namesURL = URL(string: "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt")!

    init(config: Config = Config()) {
        self.config = config
    }

    func run(progress: @escaping (String) async -> Void) async throws -> String {
        srand48(42)
        await progress("Loading dataset...")
        let content = try loadDatasetContent()
        var docs = content
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !docs.isEmpty else {
            throw MicroGPTRunnerError.noTrainingData
        }

        if docs.count > 1 {
            for i in stride(from: docs.count - 1, through: 1, by: -1) {
                let j = Int(drand48() * Double(i + 1))
                docs.swapAt(i, j)
            }
        }

        let uniqueCharacters = Array(Set(docs.joined())).sorted()
        guard !uniqueCharacters.isEmpty else {
            throw MicroGPTRunnerError.noVocabulary
        }

        let charToIndex = Dictionary(uniqueKeysWithValues: uniqueCharacters.enumerated().map { ($1, $0) })
        let bos = uniqueCharacters.count
        let vocabSize = uniqueCharacters.count + 1
        let headDim = config.nEmbd / config.nHead

        var stateDict: [String: [[MGValue]]] = [
            "wte": matrix(vocabSize, config.nEmbd),
            "wpe": matrix(config.blockSize, config.nEmbd),
            "lm_head": matrix(vocabSize, config.nEmbd),
        ]

        for layerIndex in 0..<config.nLayer {
            stateDict["layer\(layerIndex).attn_wq"] = matrix(config.nEmbd, config.nEmbd)
            stateDict["layer\(layerIndex).attn_wk"] = matrix(config.nEmbd, config.nEmbd)
            stateDict["layer\(layerIndex).attn_wv"] = matrix(config.nEmbd, config.nEmbd)
            stateDict["layer\(layerIndex).attn_wo"] = matrix(config.nEmbd, config.nEmbd)
            stateDict["layer\(layerIndex).mlp_fc1"] = matrix(4 * config.nEmbd, config.nEmbd)
            stateDict["layer\(layerIndex).mlp_fc2"] = matrix(config.nEmbd, 4 * config.nEmbd)
        }

        let params: [MGValue] = stateDict
            .sorted { $0.key < $1.key }
            .flatMap { $0.value.flatMap { $0 } }

        func linear(_ x: [MGValue], _ w: [[MGValue]]) -> [MGValue] {
            w.map { row in sumValues(zip(row, x).map { $0 * $1 }) }
        }

        func softmax(_ logits: [MGValue]) -> [MGValue] {
            let maxVal = logits.map(\.data).max() ?? 0
            let exps = logits.map { ($0 - maxVal).exp() }
            let total = sumValues(exps)
            return exps.map { $0 / total }
        }

        func rmsnorm(_ x: [MGValue]) -> [MGValue] {
            let ms = sumValues(x.map { $0 * $0 }) / Double(x.count)
            let scale = (ms + 1e-5).pow(-0.5)
            return x.map { $0 * scale }
        }

        func gpt(
            _ tokenId: Int,
            _ posId: Int,
            _ keys: inout [[[MGValue]]],
            _ values: inout [[[MGValue]]]
        ) -> [MGValue] {
            let tokEmb = stateDict["wte"]![tokenId]
            let posEmb = stateDict["wpe"]![posId]
            var x = zip(tokEmb, posEmb).map { $0 + $1 }
            x = rmsnorm(x)

            for layerIndex in 0..<config.nLayer {
                let xResidual = x
                x = rmsnorm(x)
                let q = linear(x, stateDict["layer\(layerIndex).attn_wq"]!)
                let k = linear(x, stateDict["layer\(layerIndex).attn_wk"]!)
                let v = linear(x, stateDict["layer\(layerIndex).attn_wv"]!)

                keys[layerIndex].append(k)
                values[layerIndex].append(v)

                var xAttn: [MGValue] = []
                for headIndex in 0..<config.nHead {
                    let hs = headIndex * headDim
                    let qH = Array(q[hs..<hs + headDim])
                    let kH = keys[layerIndex].map { Array($0[hs..<hs + headDim]) }
                    let vH = values[layerIndex].map { Array($0[hs..<hs + headDim]) }

                    let attnLogits = kH.map { keyStep in
                        sumValues(zip(qH, keyStep).map { $0 * $1 }) / sqrt(Double(headDim))
                    }
                    let attnWeights = softmax(attnLogits)

                    let headOut = (0..<headDim).map { j in
                        sumValues((0..<vH.count).map { t in attnWeights[t] * vH[t][j] })
                    }
                    xAttn.append(contentsOf: headOut)
                }

                x = linear(xAttn, stateDict["layer\(layerIndex).attn_wo"]!)
                x = zip(x, xResidual).map { $0 + $1 }

                let xResidual2 = x
                x = rmsnorm(x)
                x = linear(x, stateDict["layer\(layerIndex).mlp_fc1"]!)
                x = x.map { $0.relu() }
                x = linear(x, stateDict["layer\(layerIndex).mlp_fc2"]!)
                x = zip(x, xResidual2).map { $0 + $1 }
            }

            return linear(x, stateDict["lm_head"]!)
        }

        var mAdam = [Double](repeating: 0, count: params.count)
        var vAdam = [Double](repeating: 0, count: params.count)

        await progress("Training: step 0 / \(config.numSteps)")

        for step in 0..<config.numSteps {
            let doc = docs[step % docs.count]
            let tokens = [bos] + doc.map { charToIndex[$0]! } + [bos]
            let n = min(config.blockSize, tokens.count - 1)

            var keys = (0..<config.nLayer).map { _ in [[MGValue]]() }
            var values = (0..<config.nLayer).map { _ in [[MGValue]]() }
            var losses: [MGValue] = []

            for posId in 0..<n {
                let tokenId = tokens[posId]
                let targetId = tokens[posId + 1]
                let logits = gpt(tokenId, posId, &keys, &values)
                let probs = softmax(logits)
                losses.append(-probs[targetId].log())
            }

            let loss = sumValues(losses) / Double(n)
            loss.backward()

            let lrT = config.learningRate * (1.0 - Double(step) / Double(config.numSteps))
            for i in 0..<params.count {
                mAdam[i] = config.beta1 * mAdam[i] + (1 - config.beta1) * params[i].grad
                vAdam[i] = config.beta2 * vAdam[i] + (1 - config.beta2) * params[i].grad * params[i].grad
                let mHat = mAdam[i] / (1 - _pow(config.beta1, Double(step + 1)))
                let vHat = vAdam[i] / (1 - _pow(config.beta2, Double(step + 1)))
                params[i].data -= lrT * mHat / (sqrt(vHat) + config.epsAdam)
                params[i].grad = 0
            }

            await progress(String(format: "step %4d / %4d | loss %.4f", step + 1, config.numSteps, loss.data))
        }

        await progress("Inference...")

        var keys = (0..<config.nLayer).map { _ in [[MGValue]]() }
        var values = (0..<config.nLayer).map { _ in [[MGValue]]() }
        var tokenId = bos
        var sample: [Character] = []

        for posId in 0..<config.blockSize {
            let logits = gpt(tokenId, posId, &keys, &values)
            let probs = softmax(logits.map { $0 / config.temperature })
            tokenId = weightedRandomChoice(probs.map(\.data))
            if tokenId == bos { break }
            sample.append(uniqueCharacters[tokenId])
        }

        let inference = String(sample)
        await progress("Inference result: \(inference.isEmpty ? "(empty)" : inference)")
        return inference
    }

    private func loadDatasetContent() throws -> String {
        if let localURL = Bundle.main.url(forResource: "input", withExtension: "txt") {
            return try String(contentsOf: localURL, encoding: .utf8)
        }

        let data = try Data(contentsOf: namesURL)
        return String(decoding: data, as: UTF8.self)
    }
}
