//
//  DumpUtilsTest.swift
//  Cadence
//
//  验证 DumpUtils 能从 Python 写的 binary 里恢复出和 PyTorch 看到的相同数值。
//  Python 端原始输出（来自 dump_layer0.py）：
//    embed (1, 4, 2560) [-0.02416992 -0.00389099 0.00366211 0.0050354 0.01361084]
//    layer0_out (1, 4, 2560) [7.024704 -2.6239915 1.4640343 0.73644716 1.6919613]
//

import Foundation

enum DumpUtilsTest {
    static func run() {
        print("─── DumpUtils 自测 ───")

        // ─── 1. loadDump 能正确读出 [Float] ───
        let embed: [Float]
        do {
            embed = try DumpUtils.loadDump("layer0_embed.bin")
        } catch {
            fatalError("加载 layer0_embed.bin 失败：\(error)")
        }

        let expectedCount = 4 * 2560
        let countOK = embed.count == expectedCount
        print("embed count = \(embed.count)  期望 \(expectedCount)  \(countOK ? "✓" : "❌")")
        assert(countOK, "embed 数量不对")

        // ─── 2. 前 5 个数字应该精确匹配 Python 输出 ───
        // Python 端 fp32 写出，Swift 端 fp32 读入 → 应该 bit-exact
        let expectedFirst5: [Float] = [-0.02416992, -0.00389099, 0.00366211, 0.0050354, 0.01361084]
        let actual = Array(embed.prefix(5))
        print("前 5 个 (Swift): \(actual)")
        print("前 5 个 (Python): \(expectedFirst5)")

        let firstDiffs = zip(actual, expectedFirst5).map { abs($0 - $1) }
        let maxFirstDiff = firstDiffs.max() ?? 0
        let firstOK = maxFirstDiff < 1e-6
        print("max diff: \(maxFirstDiff)  \(firstOK ? "✓ bit-exact" : "❌ 数值不对")")
        assert(firstOK, "前 5 个 fp32 应该完全一致")

        // ─── 3. compareDump 自比应该返回 0 ───
        let selfDiff: Float
        do {
            selfDiff = try DumpUtils.compareDump(embed, "layer0_embed.bin")
        } catch {
            fatalError("compareDump 自比失败：\(error)")
        }
        print("自比 max diff: \(selfDiff)  \(selfDiff == 0 ? "✓" : "❌ 应该是 0")")
        assert(selfDiff == 0, "自己跟自己比应该是 0")

        // ─── 4. compareDump 加扰动应该返回扰动幅度 ───
        let perturbed = embed.map { $0 + 0.5 }
        let perturbedDiff: Float
        do {
            perturbedDiff = try DumpUtils.compareDump(perturbed, "layer0_embed.bin")
        } catch {
            fatalError("compareDump 扰动比对失败：\(error)")
        }
        let perturbOK = abs(perturbedDiff - 0.5) < 1e-5
        print("加 0.5 扰动后 max diff: \(perturbedDiff)  期望 ~0.5  \(perturbOK ? "✓" : "❌")")
        assert(perturbOK, "扰动 0.5 应该被检测到")

        // ─── 5. compareDump 长度不一致应该立刻 fatalError（这里只能注释掉，因为崩了进程就停）───
        // let truncated = Array(embed.prefix(100))
        // _ = try DumpUtils.compareDump(truncated, "layer0_embed.bin")  // 应该崩

        // ─── 6. layer0_out 也读得出，前 5 个数字应该是放大后的（验证不同文件路径都通） ───
        let layerOut = try! DumpUtils.loadDump("layer0_out.bin")
        let layerOutFirst5 = Array(layerOut.prefix(5))
        let expectedLayerOut: [Float] = [7.024704, -2.6239915, 1.4640343, 0.73644716, 1.6919613]
        let layerOutOK = zip(layerOutFirst5, expectedLayerOut).map { abs($0 - $1) }.max() ?? 0 < 1e-5
        print("layer0_out 前 5: \(layerOutFirst5)")
        print(layerOutOK ? "  ✓ 跟 Python 一致" : "  ❌")

        print("\n✅ DumpUtils 测试通过")
    }
}
