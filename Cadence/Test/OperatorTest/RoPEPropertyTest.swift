//
//  RoPEPropertyTest.swift
//  Cadence
//
//  Created by Haotian Gong on 21/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class RoPEPropertyTest {
    func run() {
        print("=== RoPE property test: length preservation ===")

        let seqLen = 10
        let nHeads = 4
        let headDim = 16

        // 1. 随机生成 x
        let totalX = seqLen * nHeads * headDim
        let xData: [Float] = (0 ..< totalX).map { _ in Float.random(in: -1 ... 1) }

        // 2. 预计算 cos/sin
        let (cosData, sinData) = RoPE.precomputeCosSin(seqLen: seqLen, headDim: headDim)

        // 3. 过 RoPE（构图 + 执行）
        let graph = MPSGraph()
        let xPh = graph.floatPlaceholder(shape: [seqLen, nHeads, headDim], name: "x")
        let cosPh = graph.floatPlaceholder(shape: [seqLen, headDim], name: "cos")
        let sinPh = graph.floatPlaceholder(shape: [seqLen, headDim], name: "sin")
        let output = RoPE.apply(graph: graph, x: xPh, cos: cosPh, sin: sinPh, headDim: headDim)

        let results = graph.run(
            with: Device.shared.commandQueue,
            feeds: [
                xPh: TensorUtils.data(from: xData, shape: [seqLen, nHeads, headDim]),
                cosPh: TensorUtils.data(from: cosData, shape: [seqLen, headDim]),
                sinPh: TensorUtils.data(from: sinData, shape: [seqLen, headDim]),
            ],
            targetTensors: [output],
            targetOperations: nil
        )
        let xRoped = TensorUtils.readFloats(from: results[output]!, count: totalX)

        // ─── 你的任务 ───
        // 4. 计算 xData 和 xRoped 中，每个 [position, head] 的 16 维向量的 L2 长度
        //    提示：对 m in 0..<seqLen, h in 0..<nHeads，
        //         起始 index = m * nHeads * headDim + h * headDim
        //         取连续 headDim 个元素，算 sqrt(sum of squares)

        let lengthsBefore: [Float] = computeLengths(
            data: xData,
            seqLen: seqLen,
            nHead: nHeads,
            headDim: headDim
        )
        let lengthsAfter: [Float] = computeLengths(
            data: xRoped,
            seqLen: seqLen,
            nHead: nHeads,
            headDim: headDim
        )

        // 5. 对比
        let maxDiff = zip(lengthsBefore, lengthsAfter).map { abs($0 - $1) }.max() ?? 0
        print("Max length diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS (RoPE preserves length)" : "❌ FAIL")

        // 额外：打印前几个长度对比，直观感受
        print("Before:", Array(lengthsBefore[0 ..< 5]))
        print("After: ", Array(lengthsAfter[0 ..< 5]))
    }

    func computeLengths(data: [Float], seqLen: Int, nHead: Int, headDim: Int) -> [Float] {
        var length = [Float](repeating: 0, count: seqLen * nHead)
        for m in 0 ..< seqLen {
            for h in 0 ..< nHead {
                let index = m * nHead * headDim + h * headDim
                let xSeq: [Float] = (index ..< index + headDim).map { data[$0] }

                let xSquared: [Float] = xSeq.map { $0 * $0 }
                let l2 = sqrt(xSquared.reduce(0, +))
                length[m * h] = l2
            }
        }
        return length
    }
}
