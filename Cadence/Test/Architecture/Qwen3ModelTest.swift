//
//  Qwen3ModelTest.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation

enum Qwen3ModelTest {
    static func run() {
        print("开始加载 Qwen3Model...")
        let t0 = Date()
        let model: Qwen3Model
        do {
            model = try Qwen3Model(modelDir: TestPaths.modelDir)
        } catch {
            fatalError("加载失败：\(error)")
        }
        let elapsed = Date().timeIntervalSince(t0)
        print("耗时 \(String(format: "%.2f", elapsed))s")

        assert(model.config.hiddenSize == 2560)
        assert(model.config.numHiddenLayers == 36)
        assert(model.layers.count == 36)
        assert(model.embedTokens.count == 151_936 * 2560)
        assert(model.finalNorm.count == 2560)
        assert(model.layers[0].qProj.count == 4096 * 2560)
        assert(model.layers[35].qProj.count == 4096 * 2560)

        print(
            "✅ Qwen3Model 加载完成 (config: hidden=\(model.config.hiddenSize), layers=\(model.config.numHiddenLayers))"
        )
    }
}
