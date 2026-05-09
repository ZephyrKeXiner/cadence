//
//  TensorUtils.swift
//  Cadence
//
//  Created by Haotian Gong on 19/4/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

enum TensorUtils {
    static func data(from array: [Float], shape: [Int]) -> MPSGraphTensorData {
        let bytes = array.withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(
            device: Device.shared.graphDevice,
            data: bytes,
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

    static func nsShape(_ shape: [Int]) -> [NSNumber] {
        shape.map { NSNumber(value: $0) }
    }

    static func readBFloats(from tensorData: MPSGraphTensorData, count: Int) -> [Float] {
        var raw = [UInt16](repeating: 0, count: count)
        raw.withUnsafeMutableBytes { ptr in
            tensorData.mpsndarray().readBytes(ptr.baseAddress!, strideBytes: nil)
        }
        return raw.map { bits16 in
            Float(bitPattern: UInt32(bits16) << 16)
        }
    }
}

extension MPSGraph {
    func floatPlaceholder(shape: [Int], name: String) -> MPSGraphTensor {
        placeholder(
            shape: TensorUtils.nsShape(shape),
            dataType: .float32,
            name: name
        )
    }

    func floatConstant(_ values: [Float], shape: [Int]) -> MPSGraphTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return constant(
            data,
            shape: TensorUtils.nsShape(shape),
            dataType: .float32
        )
    }

    func floatScalar(_ value: Float) -> MPSGraphTensor {
        constant(Double(value), dataType: .float32)
    }
}
