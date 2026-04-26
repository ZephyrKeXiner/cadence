//
//  SafeTensorsRealTest.swift
//  Cadence
//
//  在真 Qwen3-4B safetensors 文件上验证 loader：
//    1. 解析 header 不挂
//    2. tensor 数量在合理范围（4B 模型大概几百到上千个 tensor）
//    3. dtype 是 BF16（验证 BF16 路径走通）
//    4. 关键 tensor 的 shape 与 Qwen3 config 一致
//    5. 加载一个 RMSNorm 的 gamma 向量看数值是否合理（应该都接近 1.0）
//

import Foundation

enum SafeTensorsRealTest {
    static func run() {
        // Qwen3-4B 共 3 个 shard
        let shardPath = TestPaths.weightsShardPath(1, of: 3)

        let st: SafeTensors
        do {
            st = try SafeTensors(filePath: shardPath)
        } catch {
            print("❌ 加载失败：\(error)")
            print("（如果文件不存在，等下载完再跑）")
            return
        }

        print("─── SafeTensors 真 Qwen3 测试 ───")

        // ─── 1. tensor 总数 + 抽样名字 ───
        let names = st.tensors.keys.sorted()
        print("Shard 1 含 tensor 数：\(names.count)")
        print("前 10 个名字：")
        for name in names.prefix(10) {
            let info = st.tensors[name]!
            print("  \(name)  dtype=\(info.dtype)  shape=\(info.shape)")
        }

        // ─── 2. dtype 分布 ───
        var dtypeCount: [String: Int] = [:]
        for (_, info) in st.tensors {
            dtypeCount[info.dtype, default: 0] += 1
        }
        print("\ndtype 分布：\(dtypeCount)")

        // ─── 3. 关键 tensor shape 验证 ───
        // Qwen3-4B config: d_model=2560, n_heads=32, n_kv_heads=8, head_dim=128, intermediate=9728,
        // vocab=151936
        // 注意：HF 卷积约定是 [out, in]
        let expectations: [(String, [Int])] = [
            ("model.embed_tokens.weight", [151_936, 2560]),
            ("model.layers.0.input_layernorm.weight", [2560]),
            ("model.layers.0.self_attn.q_proj.weight", [2560, 2560]), // n_heads*head_dim, d_model
            ("model.layers.0.self_attn.k_proj.weight", [1024, 2560]), // n_kv*head_dim, d_model
            ("model.layers.0.self_attn.v_proj.weight", [1024, 2560]),
            ("model.layers.0.mlp.gate_proj.weight", [9728, 2560]),
            ("model.layers.0.mlp.up_proj.weight", [9728, 2560]),
            ("model.layers.0.mlp.down_proj.weight", [2560, 9728]),
        ]

        print("\n关键 tensor shape 验证：")
        for (name, expected) in expectations {
            if let info = st.tensors[name] {
                let ok = info.shape == expected
                print("  \(name)")
                print("    实际 \(info.shape)  期望 \(expected)  \(ok ? "✓" : "❌")")
            } else {
                print("  \(name) — 在 shard 1 里找不到（可能在 shard 2/3）")
            }
        }

        // ─── 4. 加载一个 RMSNorm gamma 验证 BF16 数值合理性 ───
        // input_layernorm 是预 attention 的 RMSNorm gamma，应该都在 1.0 附近
        let gammaName = "model.layers.0.input_layernorm.weight"
        if let info = st.tensors[gammaName] {
            print("\n加载 \(gammaName) (\(info.dtype)):")
            let gamma = st.loadAsFloat32(gammaName)
            print("  count: \(gamma.count)")
            print("  前 10 个: \(Array(gamma.prefix(10)))")
            let mean = gamma.reduce(0, +) / Float(gamma.count)
            let minV = gamma.min() ?? 0
            let maxV = gamma.max() ?? 0
            print("  mean=\(mean)  min=\(minV)  max=\(maxV)")
            // 训练好的 RMSNorm γ 一般在 [0.5, 2.0] 范围内
            let allReasonable = gamma.allSatisfy { $0 > 0 && $0 < 10 }
            print("  数值是否合理（>0 且 <10）: \(allReasonable ? "✓" : "❌")")
        }

        // ─── 5. 加载一个 weight matrix 头几个数看 BF16 解码 ───
        let qProjName = "model.layers.0.self_attn.q_proj.weight"
        if let info = st.tensors[qProjName] {
            print("\n加载 \(qProjName) (\(info.dtype)) 头 8 个 fp32:")
            let q = st.loadAsFloat32(qProjName)
            print("  total elements: \(q.count) (\(info.shape[0]) × \(info.shape[1]))")
            print("  前 8 个: \(Array(q.prefix(8)))")
            // weight 通常是小幅 normal 分布，不应该有 NaN / Inf
            let hasNaN = q.contains { $0.isNaN }
            let hasInf = q.contains { $0.isInfinite }
            print("  含 NaN: \(hasNaN ? "❌" : "✓ 无")  含 Inf: \(hasInf ? "❌" : "✓ 无")")
        }

        print("\n✅ SafeTensors 真文件测试完成")
    }
}
