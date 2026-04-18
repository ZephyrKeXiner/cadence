//
//  MatmulTest.swift
//  Cadence
//
//  Created by 龚浩天 on 18/4/26.
//

import Foundation
import SwiftUI
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

enum MatmulTest {
    
    static func run() {
        print("=== MPSGraph matmul test ===")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No devices")
            return
        }
        
        print(device.name)
        guard let commandQueue = device.makeCommandQueue() else {
            print("❌ Can't create command queue")
            return
        }
        
        let M = 4, N = 4, K = 4
        let aData: [Float] = (0..<M*K).map { Float($0) }
        let bData: [Float] = (0..<K*N).map { Float($0) * 0.5 }
        
        let graph = MPSGraph()
        
        let aPlaceholder = graph.placeholder(
            shape: [NSNumber(value: M), NSNumber(value: K)],
            dataType: .float32,
            name: "A"
        )
        
        let bPlaceholder = graph.placeholder(
            shape: [NSNumber(value: K), NSNumber(value: N)],
            dataType: .float32,
            name: "B"
        )
        
        let cTensor = graph.matrixMultiplication(primary: aPlaceholder, secondary: bPlaceholder, name: "C")
        
        let aTensorData = tensorData(from: aData,
                                      shape: [M, K],
                                      device: device)
        let bTensorData = tensorData(from: bData,
                                      shape: [K, N],
                                      device: device)
    }
    
    
    static func tensorData(from array: [Float],
                           shape: [Int],
                           device: MTLDevice) -> MPSGraphTensorData {
        let data = array.withUnsafeBufferPointer {Data(buffer: $0)}
        return MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: data, shape: shape.map {NSNumber(value: $0)}, dataType: .float32)
        
    }
}
