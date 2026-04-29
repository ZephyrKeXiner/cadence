//
//  Qwen3Config.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation

struct Qwen3Config: Decodable {
    let eosTokenId: Int
    let headDim: Int
    let hiddenAct: String
    let hiddenSize: Int
    let initializerRange: Float
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let numAttentionHeads: Int
    let numHiddenLayers: Int
    let numKeyValueHeads: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let tieWordEmbeddings: Bool
    let vocabSize: Int

    static func load(from path: String) throws -> Qwen3Config {
        let url = URL(filePath: path)
        let data = try Data(contentsOf: url)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Qwen3Config.self, from: data)
    }
}
