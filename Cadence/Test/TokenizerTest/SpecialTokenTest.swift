//
//  SpecialTokenTest.swift
//  Cadence
//
//  验证 special token 处理：
//    1. 纯 special token 直接映射到对应 id
//    2. 混合文本：special 段直接 → id，normal 段走完整 BPE pipeline
//    3. 没有 special token 时行为与旧版完全一致（回归保护）
//    4. encode → decode round-trip 完整保留 special token 字面值
//

import Foundation

enum SpecialTokenTest {
    static func run() {
        let basePath = "/Users/sakruhnab1/Documents/Cadence/Models/Qwen3.5-4B"
        let vocab: Vocab
        do {
            vocab = try Vocab(
                vocabPath: basePath + "/vocab.json",
                mergesPath: basePath + "/merges.txt",
                specialTokens: Vocab.qwenSpecialTokens
            )
        } catch {
            print("❌ Vocab 加载失败：\(error)")
            return
        }

        print("─── Special Token 测试 ───\n")

        var allPass = true

        // case 1: 纯 special token
        let ids1 = vocab.encode("<|im_start|>")
        let ok1 = ids1 == [151_644]
        allPass = allPass && ok1
        print("encode(\"<|im_start|>\") = \(ids1)  期望 [151644]  \(ok1 ? "✓" : "❌")")

        let ids2 = vocab.encode("<|im_end|>")
        let ok2 = ids2 == [151_645]
        allPass = allPass && ok2
        print("encode(\"<|im_end|>\") = \(ids2)  期望 [151645]  \(ok2 ? "✓" : "❌")")

        let ids3 = vocab.encode("<|endoftext|>")
        let ok3 = ids3 == [151_643]
        allPass = allPass && ok3
        print("encode(\"<|endoftext|>\") = \(ids3)  期望 [151643]  \(ok3 ? "✓" : "❌")")

        // case 2: 完整 chat 模板片段
        let chat = "<|im_start|>user\n你好<|im_end|>"
        let chatIds = vocab.encode(chat)
        let okHead = chatIds.first == 151_644
        let okTail = chatIds.last == 151_645
        allPass = allPass && okHead && okTail
        print("\nencode(\"\\(chat)\")")
        print("  → \(chatIds)")
        print("  头 = \(chatIds.first ?? -1) 期望 151644  \(okHead ? "✓" : "❌")")
        print("  尾 = \(chatIds.last ?? -1)  期望 151645  \(okTail ? "✓" : "❌")")

        // case 3: round-trip
        let chatBack = vocab.decode(chatIds)
        let okBack = chatBack == chat
        allPass = allPass && okBack
        print("  decode 回原文 = \"\(chatBack)\"  \(okBack ? "✓ 无损" : "❌ 不匹配原文")")

        // case 4: 多个 special 紧挨
        let stacked = "<|im_start|><|im_end|>"
        let stackedIds = vocab.encode(stacked)
        let okStacked = stackedIds == [151_644, 151_645]
        allPass = allPass && okStacked
        print(
            "\nencode(\"<|im_start|><|im_end|>\") = \(stackedIds)  期望 [151644, 151645]  \(okStacked ? "✓" : "❌")"
        )

        // case 5: 回归保护 —— 普通文本行为不变
        let plain = "Hello, world!"
        let plainIds = vocab.encode(plain)
        let plainExpected = [9419, 11, 1814, 0]
        let okPlain = plainIds == plainExpected
        allPass = allPass && okPlain
        print("\n回归: encode(\"Hello, world!\") = \(plainIds)")
        print("  期望 \(plainExpected)  \(okPlain ? "✓" : "❌")")

        // case 6: special 前后有普通文本
        let mixed = "Before<|im_start|>middle<|im_end|>After"
        let mixedIds = vocab.encode(mixed)
        let mixedBack = vocab.decode(mixedIds)
        let okMixed = mixedBack == mixed
        allPass = allPass && okMixed
        print("\nencode(\"\\(mixed)\")")
        print("  → \(mixedIds)")
        print("  decode 回 = \"\(mixedBack)\"  \(okMixed ? "✓ 往返一致" : "❌")")
        // 检查 special id 在中间出现
        let containsImStart = mixedIds.contains(151_644)
        let containsImEnd = mixedIds.contains(151_645)
        print("  含 <|im_start|>(151644): \(containsImStart ? "✓" : "❌")")
        print("  含 <|im_end|>(151645): \(containsImEnd ? "✓" : "❌")")
        allPass = allPass && containsImStart && containsImEnd

        print(allPass ? "\n✅ Special token 全部 PASS" : "\n❌ 有 case 失败")
    }
}
