//
//  SWiGLU.swift
//  Cadence
//
//  Created by 龚浩天 on 22/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum SWiGLU {
    static func apply(
        graph: MPSGraph,
        x: MPSGraphTensor,
        wGate: MPSGraphTensor,
        wUp: MPSGraphTensor,
        wDown: MPSGraphTensor
    ) -> MPSGraphTensor {
        let wGateT = graph.transposeTensor(wGate, dimension: 0, withDimension: 1, name: "wGate_transpose")
        let gate = graph.matrixMultiplication(primary: x, secondary: wGateT, name: "gate")
        let wUpT = graph.transposeTensor(wUp, dimension: 0, withDimension: 1, name: "wUp_transpose")
        let wDownT = graph.transposeTensor(wDown, dimension: 0, withDimension: 1, name: "wDown_transpose")
        let up = graph.matrixMultiplication(primary: x, secondary: wUpT, name: "up")

        let hidden = graph.multiplication(SiLU(x: gate), up, name: "hidden")
        return graph.matrixMultiplication(primary: hidden, secondary: wDownT, name: "output")

        func SiLU(x: MPSGraphTensor) -> MPSGraphTensor {
            let xSig = graph.sigmoid(with: x, name: "x_sigmoid")
            return graph.multiplication(x, xSig, name: "x_silu")
        }
    }
}
