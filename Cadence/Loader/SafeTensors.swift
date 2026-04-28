//
//  SafeTensors.swift
//  Cadence
//
//  Created by Haotian Gong on 26/4/26.
//

import Foundation

final class SafeTensors {
    private struct RawTensorInfo: Decodable {
        let dtype: String
        let shape: [Int]
        let byteRange: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case byteRange = "data_offsets"
        }
    }

    struct TensorInfo {
        let dtype: String
        let shape: [Int]
        let absoluteRange: Range<Int>
    }

    let tensors: [String: TensorInfo]
    private let data: Data

    init(filePath: String) throws {
        let url = URL(filePath: filePath)
        data = try Data(contentsOf: url, options: .alwaysMapped)
        let headerSize = data[0 ..< 8].withUnsafeBytes {
            $0.load(as: UInt64.self)
        }

        let headerData = data[8 ..< 8 + Int(headerSize)]
        let object = try JSONSerialization.jsonObject(with: headerData)
        guard var dict = object as? [String: Any] else {
            fatalError("Can't convert object to dictionary")
        }
        dict.removeValue(forKey: "__metadata__")

        let filteredData = try JSONSerialization.data(
            withJSONObject: dict,
            options: []
        )
        let binaryStart = 8 + Int(headerSize)
        let raw = try JSONDecoder().decode([String: RawTensorInfo].self, from: filteredData)

        tensors = raw.mapValues { r in
            TensorInfo(
                dtype: r.dtype,
                shape: r.shape,
                absoluteRange: (binaryStart + r.byteRange[0]) ..< (binaryStart + r.byteRange[1])
            )
        }
    }

    func loadAsFloat32(_ name: String) -> [Float] {
        guard let info = tensors[name] else {
            fatalError("Tensor '\(name)' not found in safetensors file")
        }
        let bytes = data[info.absoluteRange]

        switch info.dtype {
        case "F32":
            return bytes.withUnsafeBytes { buf in
                Array(buf.bindMemory(to: Float.self))
            }

        case "BF16":
            return bytes.withUnsafeBytes { buf in
                buf.bindMemory(to: UInt16.self).map { bits16 in
                    Float(bitPattern: UInt32(bits16) << 16)
                }
            }

        default:
            fatalError("Unsupported dtype '\(info.dtype)' for tensor '\(name)' (only F32/BF16 supported now)")
        }
    }
}
