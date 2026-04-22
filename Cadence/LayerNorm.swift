//
//  LayerNorm.swift
//  Cadence
//
//  Created by 龚浩天 on 21/4/26.
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

enum LayerNorm {
    static func apply(
        graph: MPSGraph,
        x: MPSGraphTensor,
        gamma: MPSGraphTensor,
        beta: MPSGraphTensor,
        eps: Double = 1e-6
    ) -> MPSGraphTensor {
        let lastAxis = NSNumber(value: x.shape!.count - 1)
        let epsTensor = graph.constant(eps, dataType: .float32)

        let mean = graph.mean(of: x, axes: [lastAxis], name: "x_mean")
        let variance = graph.mean(
            of: graph.square(with: graph.subtraction(x, mean, name: "x_sub"), name: "x_sub"),
            axes: [lastAxis],
            name: "x_var"
        )

        let x_norm = graph.multiplication(
            graph.subtraction(x, mean, name: "x_sub_mul"),
            graph.reciprocalSquareRoot(graph.addition(variance, epsTensor, name: "var_eps_add"), name: "rsr"),
            name: "x_norm"
        )

        let output_mul = graph.multiplication(x_norm, gamma, name: "output_mul")
        return graph.addition(output_mul, beta, name: "output")
    }
}
