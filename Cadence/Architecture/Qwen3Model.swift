//
//  Qwen3Model.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class Qwen3Model {
    let config: Qwen3Config
    let embedTokens: MPSGraphTensorData
    let layers: [Qwen3DecoderLayer]
    let finalNorm: MPSGraphTensorData

    init(modelDir: String) throws {
        config = try Qwen3Config.load(from: modelDir + "/config.json")
        let router = try SafeTensorsRouter(indexPath: modelDir + "/model.safetensors.index.json")

        guard let embed = router.loadAsGPU("model.embed_tokens.weight") else {
            fatalError("missing model.embed_tokens.weight")
        }
        embedTokens = embed

        guard let norm = router.loadAsGPU("model.norm.weight") else {
            fatalError("missing model.norm.weight")
        }
        finalNorm = norm

        layers = (0 ..< config.numHiddenLayers).map {
            Qwen3DecoderLayer(layerIdx: $0, router: router)
        }
    }
}
