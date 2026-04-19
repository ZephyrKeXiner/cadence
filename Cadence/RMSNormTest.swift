//
//  RMSNormTest.swift
//  Cadence
//
//  Created by 龚浩天 on 19/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class RMSNormTest {
    func run() {
        print("=== RMSNorm test ===")

        // 构造一个简单输入：3 个词，每个 8 维
        let seqLen = 3
        let dim = 8
        let eps: Float = 1e-6

        // 随机输入（用固定种子保证可重复）
        let xData: [Float] = (0 ..< seqLen * dim).map { Float($0 % 7) - 3 }
        // gamma 初始化成全 1（就是纯归一化，没缩放）
        let gammaData = [Float](repeating: 1.0, count: dim)

        // ─── 构图 ───
        let graph = MPSGraph()
        let xPh = graph.floatPlaceholder(shape: [seqLen, dim], name: "x")
        // 把 gamma 做成 constant（模拟"这是加载好的权重"）
        let gammaConst = graph.floatConstant(gammaData, shape: [dim])

        let output = RMSNorm.apply(graph: graph, x: xPh, gamma: gammaConst, eps: eps)

        // ─── 执行 ───
        let xTensorData = TensorUtils.data(from: xData, shape: [seqLen, dim])
        let results = graph.run(
            with: Device.shared.commandQueue,
            feeds: [xPh: xTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        guard let outTensorData = results[output] else {
            print("❌ no output"); return
        }
        let gpuResult = TensorUtils.readFloats(from: outTensorData, count: seqLen * dim)

        // ─── CPU 参考实现 ───
        let cpuResult = cpuRMSNorm(
            x: xData,
            gamma: gammaData,
            seqLen: seqLen,
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
