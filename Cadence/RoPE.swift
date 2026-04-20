//
//  RoPE.swift
//  Cadence
//
//  Created by 龚浩天 on 19/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum RoPE {
    static func precomputeCosSin(
        seqLen: Int,
        headDim: Int,
        base: Float = 10000
    ) -> (cos: [Float], sin: [Float]) {
        precondition(headDim % 2 == 0, "headDim must be odd")
        let half = headDim / 2

        var theta = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            theta[i] = pow(base, -Float(2 * i) / Float(headDim))
        }

        var cosTable = [Float](repeating: 0, count: seqLen * headDim)
        var sinTable = [Float](repeating: 0, count: seqLen * headDim)

        for m in 0 ..< seqLen {
            for i in 0 ..< half {
                let angle = Float(m) * theta[i]
                let c = cos(angle)
                let s = sin(angle)

                cosTable[m * headDim + i] = c
                sinTable[m * headDim + i] = s
                cosTable[m * headDim + i + half] = c
                sinTable[m * headDim + i + half] = s
            }
        }
        return (cosTable, sinTable)
    }

    static func apply(
        graph: MPSGraph,
        x: MPSGraphTensor,
        cos: MPSGraphTensor,
        sin: MPSGraphTensor,
        headDim: Int
    ) -> MPSGraphTensor {
        let half = headDim / 2
        let lastAxis = x.shape!.count - 1

        let cos3d = graph.expandDims(cos, axis: 1, name: "cos_expanded")
        let sin3d = graph.expandDims(sin, axis: 1, name: "sin_expanded")

        let xLeft = graph.sliceTensor(x, dimension: lastAxis, start: 0, length: half, name: "x_left")
        let xRight = graph.sliceTensor(x, dimension: lastAxis, start: half, length: half, name: "x_right")

        let negRight = graph.negative(with: xRight, name: "neg_right")
        let rotated = graph.concatTensors([negRight, xLeft], dimension: lastAxis, name: "rotate_half")

        let term1 = graph.multiplication(x, cos3d, name: "term1")
        let term2 = graph.multiplication(rotated, sin3d, name: "term2")

        return graph.addition(term1, term2, name: "term_sum")
    }
}
