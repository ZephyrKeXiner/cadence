//
//  ByteShadowMap.swift
//  Cadence
//
//  Created by Haotian Gong on 20/4/26.
//

import Foundation

enum ByteShadowMap {
    static let byteToShadow: [UInt8: Character] = {
        // Step 1: 收集「自己映射自己」的 byte（本身就可打印的 ASCII / Latin-1）
        var selfMapping = Set<UInt8>()
        for b in UInt8(33) ... UInt8(126) {
            selfMapping.insert(b)
        } // '!' .. '~'
        for b in UInt8(161) ... UInt8(172) {
            selfMapping.insert(b)
        } // '¡' .. '¬'
        for b in UInt8(174) ... UInt8(255) {
            selfMapping.insert(b)
        } // '®' .. 'ÿ'

        var map: [UInt8: Character] = [:]

        // 可打印的 byte：shadow 就是 byte 值对应的 unicode（0 ~ 255 恰好是 Latin-1 码点）
        for b in selfMapping {
            map[b] = Character(UnicodeScalar(b))
        }

        // 不可打印的 byte：按 byte 值顺序，依次分配 U+0100, U+0101, U+0102, ...
        var nextShadow: UInt32 = 256
        for b in UInt8(0) ... UInt8(255) {
            if !selfMapping.contains(b) {
                map[b] = Character(UnicodeScalar(nextShadow)!)
                nextShadow += 1
            }
            if b == 255 { break } // 防止 UInt8 溢出
        }

        return map
    }()

    /// 反向：shadow 字符 → byte
    static let shadowToByte: [Character: UInt8] = {
        var reverse: [Character: UInt8] = [:]
        for (byte, char) in byteToShadow {
            reverse[char] = byte
        }
        return reverse
    }()

    /// 把任意字符串的 UTF-8 字节序列转成 shadow 字符串
    /// 例："hello" → "hello"（全部 ASCII 自映射）
    ///     " hello" → "Ġhello"（空格变 Ġ）
    ///     "你好" → "ä½ å¥½"（3字节的「你」→ 3 个 shadow 字符）
    static func encode(_ text: String) -> String {
        var result = ""
        for byte in text.utf8 {
            result.append(byteToShadow[byte]!)
        }
        return result
    }

    /// 反向：shadow 字符串 → 原始字节 → UTF-8 String
    /// 例："Ġhello" → bytes [0x20, 0x68, ...] → " hello"
    static func decode(_ shadow: String) -> String {
        var bytes: [UInt8] = []
        for ch in shadow {
            if let b = shadowToByte[ch] {
                bytes.append(b)
            }
            // 未知字符忽略（正常情况下不会出现）
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
