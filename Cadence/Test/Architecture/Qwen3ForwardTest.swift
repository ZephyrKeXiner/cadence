//
//  Qwen3ForwardTest.swift
//  Cadence
//
//  Created by Haotian Gong on 9/5/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum Qwen3ForwardTest {
    static func run() {
        let model: Qwen3Model
        do {
            model = try Qwen3Model(modelDir: TestPaths.modelDir)
        } catch {
            fatalError("Qwen3Model 加载失败：\(error)")
        }

        let forward = Qwen3Forward(model: model)
        runSanityChecks(forward: forward, model: model)
        runDumpComparison(forward: forward, model: model)

        print("✅ Qwen3Forward.embedLookup 全部测试通过")
    }

    /// 内部一致性检查：不依赖外部 dump。
    /// 输入故意让 row 0 和 row 3 是同一个 token 来验证「相同 token → 相同 embedding 行」。
    /// 这个测试能 catch「索引失序 / dtype 错位」这类机制 bug。
    private static func runSanityChecks(forward: Qwen3Forward, model: Qwen3Model) {
        let input: [Int32] = [9707, 11, 1879, 9707]
        let output = forward.embedLookup(input: input)
        let outputShape = output.shape.map(\.intValue)

        assert(outputShape == [input.count, model.config.hiddenSize], "embedLookup 输出 shape 错：\(outputShape)")
        assert(output.dataType == .bFloat16, "embedLookup 输出 dtype 应该是 BF16")

        let values = TensorUtils.readBFloats(
            from: output,
            count: input.count * model.config.hiddenSize
        )
        assert(!values.contains { $0.isNaN }, "embedLookup 输出含 NaN")

        let row0 = row(values, index: 0, width: model.config.hiddenSize)
        let row1 = row(values, index: 1, width: model.config.hiddenSize)
        let row3 = row(values, index: 3, width: model.config.hiddenSize)

        assert(maxAbsDiff(row0, row3) == 0, "相同 token 的 embedding 行应该完全一致")
        assert(maxAbsDiff(row0, row1) > 0, "不同 token 的 embedding 行不应该完全一致")

        print("  ✓ Sanity checks (shape, dtype, NaN, row 一致性) 通过")
    }

    /// 与 PyTorch ground truth 对比：catch「值算错」类 bug（off-by-one、bf16 解码错等）。
    /// 输入与 dump_layer0.py 完全一致。
    private static func runDumpComparison(forward: Qwen3Forward, model: Qwen3Model) {
        let input: [Int32] = [9707, 11, 1879, 0]
        let output = forward.embedLookup(input: input)
        let values = TensorUtils.readBFloats(
            from: output,
            count: input.count * model.config.hiddenSize
        )

        let diff: Float
        do {
            diff = try DumpUtils.compareDump(values, "layer0_embed.bin")
        } catch {
            fatalError("compareDump 失败：\(error)")
        }

        // bf16 vs fp32：理论上 bit-exact（embed lookup 没有任何浮点运算，只取行 + cast）
        // 给 1e-6 容忍度防止极端情况
        let pass = diff < 1e-6
        print("  \(pass ? "✓" : "❌") vs PyTorch dump: max diff = \(diff)")
        assert(pass, "embed lookup 数值与 PyTorch 不一致")
    }

    private static func row(_ values: [Float], index: Int, width: Int) -> ArraySlice<Float> {
        values[(index * width) ..< ((index + 1) * width)]
    }

    private static func maxAbsDiff(_ lhs: ArraySlice<Float>, _ rhs: ArraySlice<Float>) -> Float {
        zip(lhs, rhs)
            .map { abs($0 - $1) }
            .max() ?? 0
    }
}
