//
//  BPETest.swift
//  Cadence
//
//  测试 Vocab.bpe(_:) 核心 BPE 算法
//  注意：bpe 的输入必须是已经 byte-shadow 编码过的字符串
//        （所以"hello"是合法输入，但" hello"应该写成 "Ġhello"）
//

import Foundation

enum BPETest {
    static func run() {
        let basePath = "/Users/sakruhnab1/Documents/Cadence/Models/Qwen3.5-4B"

        let vocab: Vocab
        do {
            vocab = try Vocab(
                vocabPath: basePath + "/vocab.json",
                mergesPath: basePath + "/merges.txt"
            )
        } catch {
            print("❌ Vocab 加载失败：\(error)")
            return
        }

        print("─── BPE 自测 ───")

        /// 用一个闭包统一打印 + 验证
        /// 验证规则：每个返回的 sub-token 都应该在 vocab 里能查到
        /// （如果查不到，说明 bpe 算法把序列切到了 vocab 不认识的形态，肯定 bug）
        func test(_ word: String, label: String = "") {
            let tokens = vocab.bpe(word)
            let ids = tokens.map { vocab.tokenToId[$0] }
            let allFound = ids.allSatisfy { $0 != nil }
            let mark = allFound ? "✓" : "❌ 有 token 不在 vocab"
            let ext = label.isEmpty ? "" : "  // \(label)"
            print("bpe(\"\(word)\") → \(tokens)  ids=\(ids)  \(mark)\(ext)")
        }

        // ─── 边界 case ───
        print("\n[边界]")
        test("", label: "空字符串")
        test("a", label: "单字符")

        // ─── 常见英文（高频词，期望合并到 1 个 token） ───
        print("\n[常见英文]")
        test("Ġthe", label: "「 the」是顶级高频")
        test("Ġand", label: "「 and」")
        test("Ġhello", label: "「 hello」常见")
        test("hello", label: "句首 hello（没前导空格）")

        // ─── 长一点的英文 ───
        print("\n[长词]")
        test("Ġattention", label: "本项目的核心词")
        test("Ġtransformer")

        // ─── 中文（UTF-8 多字节，先经 byte-shadow 编码）───
        print("\n[中文]")
        let zh1 = ByteShadowMap.encode("你好")
        print("「你好」shadow: \"\(zh1)\"")
        test(zh1, label: "你好")

        let zh2 = ByteShadowMap.encode("人工智能")
        print("「人工智能」shadow: \"\(zh2)\"")
        test(zh2, label: "人工智能")

        // ─── 罕见 / 乱码（不应该合成 1 个 token，会被拆成多片） ───
        print("\n[罕见]")
        test("Ġxqzkj", label: "不太可能合并的乱码")
        test("Ġqwxz", label: "辅音堆")

        // ─── emoji（4 字节 UTF-8，看会拆几片） ───
        print("\n[emoji]")
        let emoji = ByteShadowMap.encode("👋")
        print("「👋」shadow: \"\(emoji)\" (\(emoji.count) chars)")
        test(emoji, label: "挥手 emoji")

        // ─── 单调性检查：合并后 token 数 <= 字符数 ───
        print("\n[单调性]")
        let stressInput = "Ġhellohellohello"
        let stressTokens = vocab.bpe(stressInput)
        print("bpe(\"\(stressInput)\") 字符数=\(stressInput.count) → token 数=\(stressTokens.count)")
        assert(stressTokens.count <= stressInput.count, "BPE 输出 token 数不应超过输入字符数")
        let joined = stressTokens.joined()
        print("拼回字符串: \"\(joined)\"  \(joined == stressInput ? "✓ 无损" : "❌ 拼不回去")")
        assert(joined == stressInput, "BPE 必须无损：拼接所有 token 应该等于原输入")

        print("\n✅ BPE 自测完成")
    }
}
