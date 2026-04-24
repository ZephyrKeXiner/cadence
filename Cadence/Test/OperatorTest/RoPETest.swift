//
//  RoPETest.swift
//  Cadence
//
//  Created by Haotian Gong on 20/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class RoPETest {
    func run() {
        print("=== RoPE test ===")

        let seqLen = 4
        let nHeads = 2
        let headDim = 8

        // 随机输入 x: [seq, n_heads, headDim]
        let totalX = seqLen * nHeads * headDim
        let xData: [Float] = (0 ..< totalX).map { Float($0 % 5) - 2 }

        let (cosData, sinData) = RoPE.precomputeCosSin(seqLen: seqLen, headDim: headDim)

        // ─── 构图 ───
        let graph = MPSGraph()
        let xPh = graph.floatPlaceholder(shape: [seqLen, nHeads, headDim], name: "x")
        let cosPh = graph.floatPlaceholder(shape: [seqLen, headDim], name: "cos")
        let sinPh = graph.floatPlaceholder(shape: [seqLen, headDim], name: "sin")

        let output = RoPE.apply(graph: graph, x: xPh, cos: cosPh, sin: sinPh, headDim: headDim)

        // ─── 执行 ───
        let xTD = TensorUtils.data(from: xData, shape: [seqLen, nHeads, headDim])
        let cosTD = TensorUtils.data(from: cosData, shape: [seqLen, headDim])
        let sinTD = TensorUtils.data(from: sinData, shape: [seqLen, headDim])

        let results = graph.run(
            with: Device.shared.commandQueue,
            feeds: [xPh: xTD, cosPh: cosTD, sinPh: sinTD],
            targetTensors: [output],
            targetOperations: nil
        )
        let gpuResult = TensorUtils.readFloats(from: results[output]!, count: totalX)

        // ─── CPU 参考 ───
        let cpuResult = cpuRoPE(
            x: xData,
            cos: cosData,
            sin: sinData,
            seqLen: seqLen,
            nHeads: nHeads,
            headDim: headDim
        )

        // ─── 对比 ───
        print("Input x (first head, all positions):")
        for m in 0 ..< seqLen {
            let start = m * nHeads * headDim
            print("  pos \(m):", Array(xData[start ..< start + headDim]))
        }
        print("GPU output (first head, all positions):")
        for m in 0 ..< seqLen {
            let start = m * nHeads * headDim
            let row = Array(gpuResult[start ..< start + headDim]).map { String(format: "%.3f", $0) }
            print("  pos \(m):", row.joined(separator: ", "))
        }

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    /// CPU 参考实现
    func cpuRoPE(
        x: [Float],
        cos: [Float],
        sin: [Float],
        seqLen: Int,
        nHeads: Int,
        headDim: Int
    ) -> [Float] {
        let half = headDim / 2
        var out = [Float](repeating: 0, count: seqLen * nHeads * headDim)

        for m in 0 ..< seqLen {
            for h in 0 ..< nHeads {
                for i in 0 ..< headDim {
                    let xIdx = m * nHeads * headDim + h * headDim + i
                    let csIdx = m * headDim + i // cos/sin 表索引

                    // rotate_half(x)[i] =
                    //   i < half  →  -x[i + half]
                    //   i >= half →   x[i - half]
                    let rotatedVal: Float
                    if i < half {
                        let pairedIdx = m * nHeads * headDim + h * headDim + (i + half)
                        rotatedVal = -x[pairedIdx]
                    } else {
                        let pairedIdx = m * nHeads * headDim + h * headDim + (i - half)
                        rotatedVal = x[pairedIdx]
                    }

                    out[xIdx] = x[xIdx] * cos[csIdx] + rotatedVal * sin[csIdx]
                }
            }
        }
        return out
    }
}
