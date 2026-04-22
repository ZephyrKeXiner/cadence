//
//  SWiGLUTest.swift
//  Cadence
//
//  Created by 龚浩天 on 22/4/26.
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

        print("GPU output: ", Array(gpuResult[0 ..< seqLen * d_model]))
    }
}
