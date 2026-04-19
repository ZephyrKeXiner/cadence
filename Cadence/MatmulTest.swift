//
//  MatmulTest.swift
//  Cadence
//
//  Created by 龚浩天 on 18/4/26.
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import SwiftUI

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
        let aData: [Float] = (0 ..< M * K).map { Float($0) }
        let bData: [Float] = (0 ..< K * N).map { Float($0) * 0.5 }

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

        let aTensorData = tensorData(
            from: aData,
            shape: [M, K],
            device: device
        )
        let bTensorData = tensorData(
            from: bData,
            shape: [K, N],
            device: device
        )

        let result = graph.run(
            with: commandQueue,
            feeds: [aPlaceholder: aTensorData, bPlaceholder: bTensorData],
            targetTensors: [cTensor],
            targetOperations: nil
        )

        guard let cTensorData = result[cTensor] else {
            print("❌ No output")
            return
        }
        let gpuResult = readFloats(from: cTensorData, count: M * N)

        let cpuResult = cpuMatmul(a: aData, b: bData, M: M, N: N, K: K)

        print("A:"); printMatrix(aData, rows: M, cols: K)
        print("B:"); printMatrix(bData, rows: K, cols: N)
        print("GPU C:"); printMatrix(gpuResult, rows: M, cols: N)
        print("CPU C:"); printMatrix(cpuResult, rows: M, cols: N)

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func tensorData(
        from array: [Float],
        shape: [Int],
        device: MTLDevice
    ) -> MPSGraphTensorData {
        let data = array.withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device),
            data: data,
            shape: shape.map { NSNumber(value: $0) },
            dataType: .float32
        )
    }

    static func readFloats(
        from tensorData: MPSGraphTensorData,
        count: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBytes { ptr in
            tensorData.mpsndarray().readBytes(ptr.baseAddress!, strideBytes: nil)
        }
        return result
    }

    static func cpuMatmul(
        a: [Float],
        b: [Float],
        M: Int,
        N: Int,
        K: Int
    ) -> [Float] {
        var c = [Float](repeating: 0, count: M * N)
        for i in 0 ..< M {
            for j in 0 ..< N {
                var sum: Float = 0
                for k in 0 ..< K {
                    sum += a[i * K + k] * b[k * N + j]
                }
                c[i * N + j] = sum
            }
        }
        return c
    }

    static func printMatrix(_ data: [Float], rows: Int, cols: Int) {
        for i in 0 ..< rows {
            let row = (0 ..< cols).map { String(format: "%6.2f", data[i * cols + $0]) }
            print("  [" + row.joined(separator: ", ") + "]")
        }
    }
}
