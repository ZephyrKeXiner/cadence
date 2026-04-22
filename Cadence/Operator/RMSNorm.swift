//
//  RMSNorm.swift
//  Cadence
//
//  Created by Haotian Gong on 19/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum RMSNorm {
    static func apply(
        graph: MPSGraph,
        x: MPSGraphTensor,
        gamma: MPSGraphTensor,
        eps: Float = 1e-6
    ) -> (
        output: MPSGraphTensor,
        meanSquared: MPSGraphTensor,
        invRms: MPSGraphTensor
    ) {
        let xSquared = graph.square(with: x, name: "x_squared")

        let lastAxis = NSNumber(value: x.shape!.count - 1)
        let meanSquared = graph.mean(
            of: xSquared,
            axes: [lastAxis],
            name: "mean_squared"
        )

        let epsTensor = graph.floatScalar(eps)
        let variance = graph.addition(meanSquared, epsTensor, name: "variance")

        let invRms = graph.reciprocalSquareRoot(variance, name: "inv_rms")

        let normalized = graph.multiplication(x, invRms, name: "normalized")

        return (graph.multiplication(normalized, gamma, name: "scaled"), meanSquared, invRms)
    }
}
