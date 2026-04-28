//
//  SafeTensorsRouter.swift
//  Cadence
//
//  Created by Haotian Gong on 28/4/26.
//

import Foundation

final class SafeTensorsRouter {
    enum RouterError: Error {
        case tensorNotFound(String)
    }

    private struct SafeTensorIndex: Decodable {
        struct MetaData: Decodable {
            let totalSize: Int

            enum CodingKeys: String, CodingKey {
                case totalSize = "total_size"
            }
        }

        let metadata: MetaData?
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case metadata
            case weightMap = "weight_map"
        }
    }

    let indexData: Data
    var openShardCount: Int
    let weightMap: [String: String]
    let baseDir: URL
    private var shardCache: [String: SafeTensors]

    init(indexPath: String) throws {
        let url = URL(filePath: indexPath)
        baseDir = url.deletingLastPathComponent()
        indexData = try Data(contentsOf: url)

        let json = try JSONDecoder().decode(SafeTensorIndex.self, from: indexData)
        weightMap = json.weightMap
        openShardCount = 0
        shardCache = [:]
    }

    private func routeToSafeTensors(_ modelWeightName: String) throws -> SafeTensors {
        guard let shardFile = weightMap[modelWeightName] else {
            throw RouterError.tensorNotFound(modelWeightName)
        }

        if let cached = shardCache[shardFile] {
            return cached
        }

        let fullPathURL = baseDir.appendingPathComponent(shardFile)
        let safeTensor = try SafeTensors(filePath: fullPathURL.path)
        shardCache[shardFile] = safeTensor
        openShardCount = shardCache.count
        return safeTensor
    }

    func tensorInfo(_ name: String) -> SafeTensors.TensorInfo? {
        guard let safeTensor = try? routeToSafeTensors(name) else {
            return nil
        }
        return safeTensor.tensors[name]
    }

    func loadAsFloat32(_ name: String) -> [Float]? {
        guard let safeTensor = try? routeToSafeTensors(name) else {
            return nil
        }
        return safeTensor.loadAsFloat32(name)
    }
}
