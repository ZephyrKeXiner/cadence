//
//  ByteShadowMapTest.swift
//  Cadence
//

import Foundation

enum ByteShadowMapTest {
    static func run() {
        print("─── ByteShadowMap 自测 ───")

        // 1. 关键 byte 的 shadow 应该是熟悉的字符
        let spaceShadow = ByteShadowMap.byteToShadow[0x20]!
        print("byte 0x20 (space) → \(spaceShadow)  (应该是 Ġ, U+0120)")
        assert(spaceShadow == "Ġ", "space 应该映射到 Ġ")

        let newlineShadow = ByteShadowMap.byteToShadow[0x0A]!
        print("byte 0x0A (newline) → \(newlineShadow)  (应该是 Ċ, U+010A)")

        let nullShadow = ByteShadowMap.byteToShadow[0x00]!
        print("byte 0x00 (null) → \(nullShadow)  (应该是 Ā, U+0100)")

        // 2. 可打印 ASCII 映射到自己
        let aShadow = ByteShadowMap.byteToShadow[0x61]!
        print("byte 0x61 ('a') → \(aShadow)  (应该是 a)")
        assert(aShadow == "a")

        // 3. 256 个 byte 全部都有 shadow
        var countMapped = 0
        for b in UInt8(0) ... UInt8(255) {
            if ByteShadowMap.byteToShadow[b] != nil { countMapped += 1 }
            if b == 255 { break }
        }
        print("覆盖的 byte 数量：\(countMapped) / 256")
        assert(countMapped == 256, "应该覆盖全部 256 个 byte")

        // 4. shadow 集合没有冲突（256 个 byte → 256 个不同 shadow）
        let shadows = Set(ByteShadowMap.byteToShadow.values)
        print("不同 shadow 字符的数量：\(shadows.count) / 256")
        assert(shadows.count == 256, "shadow 应该互不相同")

        // 5. 往返：encode → decode 应该得到原文
        let cases = ["hello", " hello", "hello world", "你好", "👋 emoji"]
        for s in cases {
            let shadow = ByteShadowMap.encode(s)
            let back = ByteShadowMap.decode(shadow)
            let ok = back == s
            print("\"\(s)\" → shadow: \"\(shadow)\" → back: \"\(back)\"  \(ok ? "✓" : "✗")")
            assert(ok, "往返应该完整")
        }

        print("✅ ByteShadowMap PASS")
    }
}
