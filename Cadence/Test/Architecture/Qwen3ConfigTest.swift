//
//  Qwen3ConfigTest.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation

enum Qwen3ConfigTest {
    static func run() {
        let config: Qwen3Config
        do {
            config = try Qwen3Config.load(from: TestPaths.modelDir + "/config.json")
        } catch {
            fatalError("config 加载失败：\(error)")
        }
        print(config)
        assert(config.hiddenSize == 2560)
        assert(config.numHiddenLayers == 36)
        assert(config.tieWordEmbeddings == true)
    }
}
