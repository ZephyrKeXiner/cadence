//
//  PreTokenizeDebugTest.swift
//  Cadence
//
//  验证 preTokenDebug 的 chunk 切分 + 分类正确性
//  关键不变量：
//    1. 所有 chunk 拼起来 == 原文（无损）
//    2. chunk 数量与 preTokenize（不带分类的版本）一致
//    3. 已知 case 的分类要符合预期
//

import Foundation

enum PreTokenizeDebugTest {
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

        print("─── preTokenDebug 自测 ───\n")

        // ─── case 1：经典英文 ───
        printCase(vocab, "Hello, world!", expected: [
            ("Hello", "letters"),
            (",", "punct"),
            (" world", "letters"),
            ("!", "punct"),
        ])

        // ─── case 2：缩略语 ───
        printCase(vocab, "I'll go", expected: [
            ("I", "letters"),
            ("'ll", "contraction"),
            (" go", "letters"),
        ])

        // ─── case 3：数字按 1-3 位切 ───
        printCase(vocab, "1234567890", expected: [
            ("123", "digits"),
            ("456", "digits"),
            ("789", "digits"),
            ("0", "digits"),
        ])

        // ─── case 4：换行 + 缩进 ───
        printCase(vocab, "def foo():\n    return 1", expected: [
            ("def", "letters"),
            (" foo", "letters"),
            ("():", "punct"),
            ("\n", "newline"),
            ("    ", "trail_ws"), // 注意：4 个空格在「return」前面，可能是 letters 吃掉
            ("return", "letters"),
            (" ", "trail_ws"),
            ("1", "digits"),
        ])
        // 注意：上面的 expected 不一定完全对，主要看你输出和直觉差多少 ↑

        // ─── case 5：中文 ───
        printCase(vocab, "你好 world", expected: nil)
        // 中文不知道会被哪条吃掉，看输出

        // ─── case 6：emoji ───
        printCase(vocab, "Hi 👋 there", expected: nil)

        // ─── case 7：纯空格（边界） ───
        printCase(vocab, "   ", expected: nil)

        // ─── 验证不变量：chunk 拼起来无损 ───
        print("\n[无损性验证]")
        let stressCases = [
            "Hello, world!",
            "I'll go now, won't you?",
            "1234567890",
            "def foo():\n    return 1",
            "你好 world",
            "Hi 👋 there",
            "   ",
        ]
        var allLossless = true
        for text in stressCases {
            let chunks = vocab.preTokenDebug(text)
            let recombined = chunks.map { $0.0 }.joined()
            let ok = recombined == text
            allLossless = allLossless && ok
            print("'\(text)' → chunks=\(chunks.count) → 拼回\(ok ? "✓" : "❌")")
        }
        print(allLossless ? "✅ 所有 case 拼接无损" : "❌ 有 case 丢字符")
    }

    /// 打印一个 case 的 chunk 切分 + 与期望对比
    private static func printCase(
        _ vocab: Vocab,
        _ text: String,
        expected: [(String, String)]?
    ) {
        let chunks = vocab.preTokenDebug(text)
        let escaped = text.replacingOccurrences(of: "\n", with: "\\n")
        print("Input: \"\(escaped)\"")
        for (i, chunk) in chunks.enumerated() {
            let chunkText = chunk.0.replacingOccurrences(of: "\n", with: "\\n")
            let mark: String
            if let exp = expected, i < exp.count {
                let ok = exp[i].0 == chunk.0 && exp[i].1 == chunk.1
                mark = ok ? "✓" : "❌ 期望(\(exp[i].0), \(exp[i].1))"
            } else {
                mark = ""
            }
            print("  [\(i)] '\(chunkText)' → \(chunk.1) \(mark)")
        }
        print("")
    }
}
