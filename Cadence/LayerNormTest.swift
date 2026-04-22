//
//  LayerNormTest.swift
//  Cadence
//
//  Created by 龚浩天 on 21/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum LayerNormTest {
    static func run() {
        let seqLen = 4
        let dim = 8

        let xData: [Float] = (0 ..< (seqLen * dim)).map { Float($0 % 11) - 5 }
        let gammaData: [Float] = (0 ..< dim).map { Float($0) + 0.5 * 0.1 }
        let betaData: [Float] = (0 ..< dim).map { Float($0) * 0.05 }

        let graph = MPSGraph()
        let xPh = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: dim)],
            dataType: MPSDataType.float32,
            name: "x"
        )
        let gammaPh = graph.placeholder(
            shape: [NSNumber(value: dim)],
            dataType: MPSDataType.float32,
            name: "gamma"
        )
        let betaPh = graph.placeholder(
            shape: [NSNumber(value: dim)],
            dataType: MPSDataType.float32,
            name: "beta"
        )

        let xTensorData: MPSGraphTensorData = TensorUtils.data(from: xData, shape: [seqLen, dim])
        let gammaTensorData: MPSGraphTensorData = TensorUtils.data(from: gammaData, shape: [dim])
        let betaTensorData: MPSGraphTensorData = TensorUtils.data(from: betaData, shape: [dim])

        let output = LayerNorm.apply(graph: graph, x: xPh, gamma: gammaPh, beta: betaPh)

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [xPh: xTensorData, gammaPh: gammaTensorData, betaPh: betaTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        let gpuResult = TensorUtils.readFloats(from: result[output]!, count: seqLen * dim)
        let cpuResult = cpuLayerNorm(
            x: xData, gamma: gammaData, beta: betaData,
            seqLen: seqLen, dim: dim, eps: 1e-6
        )
        print("GPU output (row 0):", Array(gpuResult[0 ..< dim]))
        print("CPU output (row 0):", Array(cpuResult[0 ..< dim]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func cpuLayerNorm(
        x: [Float], gamma: [Float], beta: [Float],
        seqLen: Int, dim: Int, eps: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * dim)
        for i in 0 ..< seqLen {
            let start = i * dim
            let row = Array(x[start ..< start + dim])

            let mean = row.reduce(0, +) / Float(dim)
            let centered = row.map { $0 - mean }
            let variance = centered.map { $0 * $0 }.reduce(0, +) / Float(dim)
            let invStd = 1.0 / sqrt(variance + eps)

            for j in 0 ..< dim {
                out[start + j] = centered[j] * invStd * gamma[j] + beta[j]
            }
        }
        return out
    }
}
