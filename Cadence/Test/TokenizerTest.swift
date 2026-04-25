//
//  TokenizerTest.swift
//  Cadence
//
//  端到端 encode / decode 测试。
//  关键不变量：encode → decode 应该等于原文（无损往返）。
//

import Foundation

enum TokenizerTest {
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

        print("─── Tokenizer 端到端测试 ───")

        let cases: [String] = [
            "Hello, world!",
            "你好世界",
            "  leading spaces",
            "trailing spaces  ",
            "1234567890",
            "👋 emoji",
            "Mix English and 中文 in one sentence.",
            "def foo():\n    return 1",
            "I'll go now, won't you?", // 测缩略
            "",
        ]

        var allPass = true
        for text in cases {
            let ids = vocab.encode(text)
            let back = vocab.decode(ids)
            let ok = back == text
            allPass = allPass && ok
            let preview = ids.prefix(10).map(String.init).joined(separator: ",")
            let suffix = ids.count > 10 ? "...(共\(ids.count))" : ""
            print("\"\(text)\"")
            print("  → ids: [\(preview)\(suffix)]")
            print("  → back: \"\(back)\"  \(ok ? "✓" : "❌")")
        }

        print(allPass ? "\n✅ 全部往返一致" : "\n❌ 有 case 失败")
    }
}
