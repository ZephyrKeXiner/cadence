//
//  Qwen3ModelTest.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

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

        // ─── config ───
        assert(model.config.hiddenSize == 2560)
        assert(model.config.numHiddenLayers == 36)
        assert(model.layers.count == 36)

        /// ─── shape helpers ───
        /// MPSGraphTensorData.shape 是 [NSNumber]，转 [Int] 方便比较
        func shape(_ td: MPSGraphTensorData) -> [Int] {
            td.shape.map(\.intValue)
        }

        // ─── 全局 weights ───
        assert(shape(model.embedTokens) == [151_936, 2560], "embedTokens 形状错")
        assert(shape(model.finalNorm) == [2560], "finalNorm 形状错")
        assert(model.embedTokens.dataType == .bFloat16, "embedTokens 应该是 BF16")

        // ─── 第 0 层的关键 weights ───
        let l0 = model.layers[0]
        assert(shape(l0.inputLayernorm) == [2560])
        assert(shape(l0.postAttentionLayernorm) == [2560])
        assert(shape(l0.qProj) == [4096, 2560])
        assert(shape(l0.kProj) == [1024, 2560])
        assert(shape(l0.vProj) == [1024, 2560])
        assert(shape(l0.oProj) == [2560, 4096])
        assert(shape(l0.qNorm) == [128])
        assert(shape(l0.kNorm) == [128])
        assert(shape(l0.gateProj) == [9728, 2560])
        assert(shape(l0.upProj) == [9728, 2560])
        assert(shape(l0.downProj) == [2560, 9728])

        // ─── 最后一层（验证跨 shard 加载）───
        let l35 = model.layers[35]
        assert(shape(l35.qProj) == [4096, 2560])
        assert(shape(l35.downProj) == [2560, 9728])

        // ─── dtype 抽查 ───
        // weights 应该全是 BF16，norm 也是 BF16（按 SafeTensorsRealTest 的 dtype 分布）
        assert(l0.qProj.dataType == .bFloat16, "qProj 应该是 BF16")
        assert(l0.inputLayernorm.dataType == .bFloat16, "inputLayernorm 应该是 BF16")

        print(
            "✅ Qwen3Model 加载完成 (hidden=\(model.config.hiddenSize), layers=\(model.config.numHiddenLayers), dtype=BF16)"
        )
        print("提示：在 Activity Monitor 看 Cadence 进程内存，应该 ≈ 8 GB")
    }
}
