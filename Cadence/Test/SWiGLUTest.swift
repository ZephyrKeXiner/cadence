//
//  SWiGLUTest.swift
//  Cadence
//
//  Created by Haotian Gong on 22/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum SWiGLUTest {
    static func run() {
        let d_model = 4
        let d_ff = 16
        let seqLen = 3
        let graph = MPSGraph()

        let xData: [Float] = (0 ..< seqLen * d_model).map { Float($0) / Float(seqLen * d_model) }
        let wGate: [Float] = (0 ..< d_ff * d_model).map { Float($0) / Float(d_ff * d_model) }
        let wUp: [Float] = (0 ..< d_ff * d_model).map { Float($0) / Float(d_ff * d_model) }
        let wDown: [Float] = (0 ..< d_model * d_ff).map { Float($0) / Float(d_ff * d_model) }

        let xPh = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: d_model)],
            dataType: .float32,
            name: "x_ph"
        )
        let wGatePh = graph.placeholder(
            shape: [NSNumber(value: d_ff), NSNumber(value: d_model)],
            dataType: .float32,
            name: "wGate_ph"
        )
        let wUpPh = graph.placeholder(
            shape: [NSNumber(value: d_ff), NSNumber(value: d_model)],
            dataType: .float32,
            name: "wUp_ph"
        )
        let wDownPh = graph.placeholder(
            shape: [NSNumber(value: d_model), NSNumber(value: d_ff)],
            dataType: .float32,
            name: "wDown_ph"
        )

        let xTensorData = TensorUtils.data(from: xData, shape: [seqLen, d_model])
        let wGateTensorData = TensorUtils.data(from: wGate, shape: [d_ff, d_model])
        let wUpTensorData = TensorUtils.data(from: wUp, shape: [d_ff, d_model])
        let wDownTensorData = TensorUtils.data(from: wDown, shape: [d_model, d_ff])

        let output = SWiGLU.apply(graph: graph, x: xPh, wGate: wGatePh, wUp: wUpPh, wDown: wDownPh)

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [xPh: xTensorData, wGatePh: wGateTensorData, wUpPh: wUpTensorData,
                    wDownPh: wDownTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        let gpuResult = TensorUtils.readFloats(from: result[output]!, count: seqLen * d_model)

        let cpuResult = cpuSWiGLU(
            x: xData, wGate: wGate, wUp: wUp, wDown: wDown,
            seqLen: seqLen, dModel: d_model, dFF: d_ff
        )

        print("GPU output: ", Array(gpuResult[0 ..< seqLen * d_model]))
        print("CPU output: ", Array(cpuResult[0 ..< seqLen * d_model]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func cpuSWiGLU(
        x: [Float],
        wGate: [Float],
        wUp: [Float],
        wDown: [Float],
        seqLen: Int,
        dModel: Int,
        dFF: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * dModel)

        for i in 0 ..< seqLen {
            // 1. 同时算 gate 和 up（两个 matmul 合并成一个循环，省得重复遍历 x）
            var gate = [Float](repeating: 0, count: dFF)
            var up = [Float](repeating: 0, count: dFF)
            for f in 0 ..< dFF {
                var sumGate: Float = 0
                var sumUp: Float = 0
                for k in 0 ..< dModel {
                    let xVal = x[i * dModel + k]
                    sumGate += wGate[f * dModel + k] * xVal
                    sumUp += wUp[f * dModel + k] * xVal
                }
                gate[f] = sumGate
                up[f] = sumUp
            }

            // 2. hidden = silu(gate) * up，逐元素
            //    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            var hidden = [Float](repeating: 0, count: dFF)
            for f in 0 ..< dFF {
                let g = gate[f]
                let silu = g / (1.0 + exp(-g))
                hidden[f] = silu * up[f]
            }

            // 3. out[i, j] = Σ_f wDown[j, f] * hidden[f]
            for j in 0 ..< dModel {
                var sum: Float = 0
                for f in 0 ..< dFF {
                    sum += wDown[j * dFF + f] * hidden[f]
                }
                out[i * dModel + j] = sum
            }
        }

        return out
    }
}
