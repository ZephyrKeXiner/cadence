//
//  AttentionPerfTest.swift
//  Cadence
//
//  Created by Codex on 22/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum AttentionPerfTest {
    static func run() {
        print("=== Attention perf test ===")

        let seqLen = 1024
        let dK = 64
        let warmupRuns = 3
        let timedRuns = 10

        let qData = makeData(count: seqLen * dK, seed: 0x1234_5678)
        let kData = makeData(count: seqLen * dK, seed: 0x8765_4321)
        let vData = makeData(count: seqLen * dK, seed: 0xCAFE_BABE)

        let graph = MPSGraph()
        let qPh = graph.floatPlaceholder(shape: [seqLen, dK], name: "q")
        let kPh = graph.floatPlaceholder(shape: [seqLen, dK], name: "k")
        let vPh = graph.floatPlaceholder(shape: [seqLen, dK], name: "v")
        let dKTensor = graph.floatScalar(Float(dK))
        let output = Attention.apply(graph: graph, Q: qPh, K: kPh, V: vPh, d_k: dKTensor)

        let feeds = [
            qPh: TensorUtils.data(from: qData, shape: [seqLen, dK]),
            kPh: TensorUtils.data(from: kData, shape: [seqLen, dK]),
            vPh: TensorUtils.data(from: vData, shape: [seqLen, dK]),
        ]

        for _ in 0 ..< warmupRuns {
            _ = graph.run(
                with: Device.shared.commandQueue,
                feeds: feeds,
                targetTensors: [output],
                targetOperations: nil
            )
        }

        var lastGPUResult: [MPSGraphTensor: MPSGraphTensorData] = [:]
        let gpuAverageMs = averageMilliseconds(runs: timedRuns) {
            lastGPUResult = graph.run(
                with: Device.shared.commandQueue,
                feeds: feeds,
                targetTensors: [output],
                targetOperations: nil
            )
        }

        let gpuOutput = TensorUtils.readFloats(
            from: lastGPUResult[output]!,
            count: seqLen * dK
        )

        var cpuOutput = [Float]()
        let cpuAverageMs = averageMilliseconds(runs: timedRuns) {
            cpuOutput = cpuAttention(
                q: qData,
                k: kData,
                v: vData,
                seqLen: seqLen,
                dK: dK
            )
        }

        let maxDiff = zip(gpuOutput, cpuOutput).map { abs($0 - $1) }.max() ?? 0
        let speedup = cpuAverageMs / gpuAverageMs

        print("shape: [\(seqLen), \(dK)]")
        print("warmup runs: \(warmupRuns)")
        print("timed runs: \(timedRuns)")
        print(String(format: "GPU avg: %.3f ms", gpuAverageMs))
        print(String(format: "CPU avg: %.3f ms", cpuAverageMs))
        print(String(format: "speedup (CPU/GPU): %.2fx", speedup))
        print("GPU output (row 0):", Array(gpuOutput[0 ..< min(dK, 8)]))
        print("CPU output (row 0):", Array(cpuOutput[0 ..< min(dK, 8)]))
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func cpuAttention(
        q: [Float],
        k: [Float],
        v: [Float],
        seqLen: Int,
        dK: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * dK)
        let scale = 1.0 / sqrt(Float(dK))

        for i in 0 ..< seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            for j in 0 ..< seqLen {
                var dot: Float = 0
                for kIndex in 0 ..< dK {
                    dot += q[i * dK + kIndex] * k[j * dK + kIndex]
                }
                scores[j] = dot * scale
            }

            let maxScore = scores.max() ?? 0
            var sumExp: Float = 0
            for j in 0 ..< seqLen {
                scores[j] = exp(scores[j] - maxScore)
                sumExp += scores[j]
            }
            for j in 0 ..< seqLen {
                scores[j] /= sumExp
            }

            for valueIndex in 0 ..< dK {
                var sum: Float = 0
                for j in 0 ..< seqLen {
                    sum += scores[j] * v[j * dK + valueIndex]
                }
                out[i * dK + valueIndex] = sum
            }
        }

        return out
    }

    static func averageMilliseconds(runs: Int, _ block: () -> Void) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< runs {
            block()
        }
        let end = DispatchTime.now().uptimeNanoseconds
        let totalMilliseconds = Double(end - start) / 1_000_000
        return totalMilliseconds / Double(runs)
    }

    static func makeData(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1
            let value = Float((state >> 32) & 0xFFFF) / Float(0xFFFF)
            return value * 2 - 1
        }
    }
}
