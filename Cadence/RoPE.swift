//
//  RoPE.swift
//  Cadence
//
//  Created by 龚浩天 on 19/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum RoPE {
    static func precomputerConsin(seqLen: Int, headDim: Int, base:Float = 10000 ) -> (cos: [Float], sin: [Float]) {
        precondition(headDim % 2 == 0, "headDim must be odd")
        let half = headDim / 2
        
        var theta = [Float](repeating: 0, count: half)
        for i in 0..<half {
            theta[i] = pow(base, -Float(2 * i) / Float(headDim))
        }
        
        var cosTable = [Float](repeating: 0, count: seqLen * headDim)
        var sinTable = [Float](repeating: 0, count: seqLen * headDim)
        
        for m in 0..<seqLen {
            for i in 0..<half {
                let angle = Float(m) * theta[i]
                let c = cos(angle)
                let s = sin(angle)
                
                cosTable[m*headDim + i] = c
                sinTable[m*headDim + i] = s
                cosTable[m * headDim + i + half] = c
                sinTable[m * headDim + i + half] = s
            }
        }
        return (cosTable, sinTable)
    }
}
