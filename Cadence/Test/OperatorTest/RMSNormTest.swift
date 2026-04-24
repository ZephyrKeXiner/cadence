//
//  RMSNormTest.swift
//  Cadence
//
//  Created by Haotian Gong on 19/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class RMSNormTest {
    func run() {
        print("=== RMSNorm test ===")

        let batch = 2
        // 构造一个简单输入：3 个词，每个 8 维
        let seqLen = 3
        let dim = 8
        let eps: Float = 1e-6

        // 随机输入（用固定种子保证可重复）
        let xData: [Float] = (0 ..< batch * seqLen * dim).map { Float($0 % 7) - 3 }
        // gamma 初始化成全 1（就是纯归一化，没缩放）
        let gammaData = [Float](repeating: 1.0, count: dim)

        // ─── 构图 ───
        let graph = MPSGraph()
        let xPh = graph.floatPlaceholder(shape: [batch, seqLen, dim], name: "x")
        // 把 gamma 做成 constant（模拟"这是加载好的权重"）
        let gammaConst = graph.floatConstant(gammaData, shape: [dim])

        let result = RMSNorm.apply(graph: graph, x: xPh, gamma: gammaConst, eps: eps)

        // ─── 执行 ───
        let xTensorData = TensorUtils.data(from: xData, shape: [batch, seqLen, dim])
        let runResults = graph.run(
            with: Device.shared.commandQueue,
            feeds: [xPh: xTensorData],
            targetTensors: [result.output, result.meanSquared, result.invRms],
            targetOperations: nil
        )

        guard let outTensorData = runResults[result.output] else {
            print("❌ no output"); return
        }
        let meanSquared = TensorUtils.readFloats(from: runResults[result.meanSquared]!, count: batch * seqLen)
        let invRms = TensorUtils.readFloats(from: runResults[result.invRms]!, count: batch * seqLen)
        let gpuResult = TensorUtils.readFloats(from: outTensorData, count: batch * seqLen * dim)

        print(Array(meanSquared[0 ..< seqLen]))
        print(Array(invRms[0 ..< seqLen]))
        // ─── CPU 参考实现 ───
        let cpuResult = cpuRMSNorm(
            x: xData,
            gamma: gammaData,
            seqLen: batch * seqLen,
            dim: dim,
            eps: eps
        )

        // ─── 对比 ───
        print("Input x (row 0):     ", Array(xData[0 ..< dim]))
        print("GPU output (row 0):  ", Array(gpuResult[0 ..< dim]))
        print("CPU output (row 0):  ", Array(cpuResult[0 ..< dim]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    /// CPU 参考实现：逐行对 [seqLen, dim] 做 RMSNorm
    func cpuRMSNorm(
        x: [Float],
        gamma: [Float],
        seqLen: Int,
        dim: Int,
        eps: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * dim)
        for i in 0 ..< seqLen {
            // 取出第 i 行
            let start = i * dim
            let row = Array(x[start ..< start + dim])

            // 算 mean(x²)
            let meanSq = row.map { $0 * $0 }.reduce(0, +) / Float(dim)
            let invRms = 1.0 / sqrt(meanSq + eps)

            // 逐元素：x * invRms * gamma
            for j in 0 ..< dim {
                out[start + j] = row[j] * invRms * gamma[j]
            }
        }
        return out
    }
}
