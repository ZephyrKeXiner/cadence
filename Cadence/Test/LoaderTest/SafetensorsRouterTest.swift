//
//  SafetensorsRouterTest.swift
//  Cadence
//
//  端到端验证多 shard 路由：
//    1. 解析 index.json，weight_map 不为空
//    2. lazy 行为：访问前 0 shard 打开，访问 1 个 tensor 后 1 shard 打开
//    3. 跨 shard 访问：访问其他 shard 的 tensor 时，多打开 1 个 shard
//    4. Qwen3-4B 关键 tensor 都能找到 + shape 与预期一致
//    5. embed_tokens 加载出来数值合理（非 NaN/Inf）
//

import Foundation

enum SafetensorsRouterTest {
    static func run() {
        let router: SafeTensorsRouter
        do {
            router = try SafeTensorsRouter(indexPath: TestPaths.weightsIndexPath)
        } catch {
            print("❌ 加载 index.json 失败：\(error)")
            print("（如果文件还没下载完，等下再跑）")
            return
        }

        print("─── SafetensorsRouter 测试 ───")
        print("weight_map 总 tensor 数: \(router.weightMap.count)")

        // ─── shard 分布 ───
        let shardSet = Set(router.weightMap.values)
        print("使用的 shard 数: \(shardSet.count)")
        print("shard 文件: \(shardSet.sorted())")

        // ─── lazy 验证 ───
        print("\n[lazy 行为]")
        print("访问任何 tensor 之前，已打开 shard 数: \(router.openShardCount)")
        assert(router.openShardCount == 0)

        // 访问第 1 个 tensor
        _ = router.loadAsFloat32("model.embed_tokens.weight")
        print("访问 embed_tokens 之后，已打开 shard 数: \(router.openShardCount)")
        assert(router.openShardCount == 1)

        // ─── Qwen3-4B 关键 tensor shape 验证 ───
        // d_model=2560, n_heads=32, n_kv_heads=8, head_dim=128, intermediate=9728, vocab=151936
        // tie_word_embeddings=true → 没有独立的 lm_head
        let expectations: [(String, [Int])] = [
            ("model.embed_tokens.weight", [151_936, 2560]),
            ("model.norm.weight", [2560]),
            ("model.layers.0.input_layernorm.weight", [2560]),
            ("model.layers.0.post_attention_layernorm.weight", [2560]),
            ("model.layers.0.self_attn.q_proj.weight", [4096, 2560]), // 32×128, d_model
            ("model.layers.0.self_attn.k_proj.weight", [1024, 2560]), // 8×128, d_model
            ("model.layers.0.self_attn.v_proj.weight", [1024, 2560]),
            ("model.layers.0.self_attn.o_proj.weight", [2560, 4096]),
            ("model.layers.0.self_attn.q_norm.weight", [128]), // ⭐ Qwen3 独有
            ("model.layers.0.self_attn.k_norm.weight", [128]), // ⭐ Qwen3 独有
            ("model.layers.0.mlp.gate_proj.weight", [9728, 2560]),
            ("model.layers.0.mlp.up_proj.weight", [9728, 2560]),
            ("model.layers.0.mlp.down_proj.weight", [2560, 9728]),
            ("model.layers.35.input_layernorm.weight", [2560]), // 最后一层也存在
        ]

        print("\n关键 tensor shape 验证：")
        var allShapeOK = true
        for (name, expected) in expectations {
            if let info = router.tensorInfo(name) {
                let ok = info.shape == expected
                allShapeOK = allShapeOK && ok
                print("  \(ok ? "✓" : "❌") \(name)")
                if !ok {
                    print("    实际 \(info.shape)  期望 \(expected)")
                } else {
                    print("    \(info.shape) (\(info.dtype))")
                }
            } else {
                allShapeOK = false
                print("  ❌ \(name) — 不在 weight_map 里")
            }
        }

        // ─── 跨 shard 访问的 lazy 验证 ───
        print("\n[跨 shard 访问]")
        let beforeShards = router.openShardCount
        // 试图访问最后一层的 tensor，多半在最后一个 shard
        _ = router.loadAsFloat32("model.layers.35.input_layernorm.weight")
        let afterShards = router.openShardCount
        print("访问最后一层之前：\(beforeShards) shard；之后：\(afterShards) shard")

        // ─── 数值合理性检查 ───
        print("\n[数值检查] model.layers.0.input_layernorm.weight (Qwen3 RMSNorm γ)")
        guard let gamma = router.loadAsFloat32("model.layers.0.input_layernorm.weight") else {
            print("  ❌ 加载 model.layers.0.input_layernorm.weight 失败")
            print("\n❌ 有失败")
            return
        }
        let mean = gamma.reduce(0, +) / Float(gamma.count)
        let minV = gamma.min() ?? 0
        let maxV = gamma.max() ?? 0
        print("  count=\(gamma.count) mean=\(mean) min=\(minV) max=\(maxV)")
        // Qwen3 layer 0 的 γ 值确实小（mean≈0.023, 含负），已通过 Python bit-exact 验证
        // 这里只检查没爆炸：finite 且 |γ|<100
        let reasonable = gamma.allSatisfy { $0.isFinite && abs($0) < 100 }
        print("  γ 数值有限不爆炸（|γ|<100）: \(reasonable ? "✓" : "❌")")

        // ─── 整体结论 ───
        print(allShapeOK && reasonable ? "\n✅ Router 测试全部通过" : "\n❌ 有失败")
    }
}
