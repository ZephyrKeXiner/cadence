//
//  LayerNormTest.swift
//  Cadence
//
//  Created by 龚浩天 on 21/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class LayerNormTest {
    static func run() {
        let seq: Int = 4
        let dim: Int = 8
        let gamma: Int = 1
        
        let xArray: [Float] = (0..<(seq * dim)).map { _ in Float.random(in: -1.0..<1.0) }
        let graph = MPSGraph()
        let x = graph.placeholder(shape: [NSNumber(value: seq * dim)], dataType: MPSDataType.float32, name: "x")
        // TODO: need completing
//        let gamma = graph.
        
        let output = LayerNorm.apply(graph: graph, x: x, gamma: <#T##MPSGraphTensor#>, beta: <#T##MPSGraphTensor#>)
        
        
    }
}
