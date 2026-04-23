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

    static func applyMultiHead(
        graph: MPSGraph,
        Q: MPSGraphTensor,
        K: MPSGraphTensor,
        V: MPSGraphTensor,
        mask: MPSGraphTensor,
        headDim: Int
    ) -> MPSGraphTensor {
        let headDimTensor = graph.constant(Double(headDim), dataType: .float32)
        let KT = graph.transposeTensor(K, dimension: 1, withDimension: 2, name: "K_transpose")
        let QKMul = graph.matrixMultiplication(primary: Q, secondary: KT, name: "QK_mul")
        let rsqrtHeadDim = graph.reciprocalSquareRoot(headDimTensor, name: "headDim_rsqrt")
        let scores = graph.multiplication(QKMul, rsqrtHeadDim, name: "scores")
        let masked = graph.addition(mask, scores, name: "scores_with_mask")

        let weight = graph.softMax(with: masked, axis: 2, name: "weight_softmax")
        return graph.matrixMultiplication(primary: weight, secondary: V, name: "output")
    }

    static func applyGQA(
        graph: MPSGraph,
        Q: MPSGraphTensor, // [n_heads,    seq, head_dim]
        K: MPSGraphTensor, // [n_kv_heads, seq, head_dim]
        V: MPSGraphTensor, // [n_kv_heads, seq, head_dim]
        mask: MPSGraphTensor,
        nHeads: Int,
        nKvHeads: Int,
        seqLen: Int,
        headDim: Int
    ) -> MPSGraphTensor {
        let K_expanded = graph.expandDims(K, axis: 1, name: "K_expanded")
        let V_expanded = graph.expandDims(V, axis: 1, name: "V_expanded")
        let nRep = nHeads / nKvHeads

        let K_broadcast = graph.broadcast(
            K_expanded,
            shape: [NSNumber(value: nKvHeads), NSNumber(value: nRep), NSNumber(value: seqLen),
                    NSNumber(value: headDim)],
            name: "K_broadcast"
        )
        let V_broadcast = graph.broadcast(
            V_expanded,
            shape: [NSNumber(value: nKvHeads), NSNumber(value: nRep), NSNumber(value: seqLen),
                    NSNumber(value: headDim)],
            name: "V_broadcast"
        )

        let KFinal = graph.reshape(
            K_broadcast,
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            name: "K_final"
        )
        let VFinal = graph.reshape(
            V_broadcast,
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            name: "V_final"
        )

        return applyMultiHead(graph: graph, Q: Q, K: KFinal, V: VFinal, mask: mask, headDim: headDim)
    }
}
