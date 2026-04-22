//
//  Attention.swift
//  Cadence
//
//  Created by Haotian Gong on 22/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum Attention {
    static func apply(
        graph: MPSGraph,
        Q: MPSGraphTensor,
        K: MPSGraphTensor,
        V: MPSGraphTensor,
        d_k: MPSGraphTensor
    ) -> MPSGraphTensor {
        let KT = graph.transposeTensor(K, dimension: 0, withDimension: 1, name: "K_transpose")
        let QKMul = graph.matrixMultiplication(primary: Q, secondary: KT, name: "QK_mul")
        let rsqrtDk = graph.reciprocalSquareRoot(d_k, name: "dk_rsqrt")
        let scores = graph.multiplication(QKMul, rsqrtDk, name: "scores")

        let weight = graph.softMax(with: scores, axis: 1, name: "weight_softmax")
        return graph.matrixMultiplication(primary: weight, secondary: V, name: "output")
    }
}
