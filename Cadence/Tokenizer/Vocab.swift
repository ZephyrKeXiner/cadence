//
//  Vocab.swift
//  Cadence
//
//  Created by Haotian Gong on 25/4/26.
//

import Foundation

struct Pair: Hashable {
    let first: String
    let second: String
}

final class Vocab {
    /// Qwen3 chat 推理需要的最关键 3 个 special token
    /// （未来可以扩展成读 tokenizer_config.json 拿全部 32 个）
    static let qwenSpecialTokens: [String: Int] = [
        "<|endoftext|>": 151_643,
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
    ]

    let tokenToId: [String: Int] // "Ġhello" → 24082       (从 vocab.json 来)
    let idToToken: [Int: String] // 24082 → "Ġhello"       (反向，decode 用；含 specialTokens)
    let bpeRanks: [Pair: Int] // ("Ġ","t") → 3          (从 merges.txt 来)
    let specialTokens: [String: Int] // "<|im_start|>" → 151644
    private let specialTokenRegex: NSRegularExpression? // 预编译，匹配任一 special token

    init(
        vocabPath: String,
        mergesPath: String,
        specialTokens: [String: Int]
    ) throws {
        // ── vocab.json ──
        let vocabData = try Data(contentsOf: URL(filePath: vocabPath))
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        tokenToId = vocab

        // ── 反向 + merge specialTokens（让 decode 透明支持）──
        var reverse: [Int: String] = [:]
        reverse.reserveCapacity(vocab.count + specialTokens.count)
        for (token, id) in vocab {
            reverse[id] = token
        }
        for (token, id) in specialTokens {
            reverse[id] = token
        }
        idToToken = reverse

        // ── merges.txt ──
        let mergesString = try String(contentsOfFile: mergesPath, encoding: .utf8)
        let lines = mergesString.split(separator: "\n").map(String.init)
        bpeRanks = Dictionary(uniqueKeysWithValues: lines.enumerated().compactMap { index, line -> (
            Pair,
            Int
        )? in
            let parts = line.split(separator: " ").map(String.init)
            guard parts.count == 2 else {
                return nil
            }

            let pair = Pair(first: parts[0], second: parts[1])

            return (pair, index)
        })

        // ── 预编译 special token regex（按长度降序避免短的吞长的）──
        self.specialTokens = specialTokens
        if specialTokens.isEmpty {
            specialTokenRegex = nil
        } else {
            let sorted = specialTokens.keys.sorted { $0.count > $1.count }
            let pattern = sorted
                .map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            specialTokenRegex = try NSRegularExpression(pattern: pattern)
        }
    }

    /// 文本 → token id 列表
    /// pipeline: split-by-special → 对每段：special 直接查 / normal 走 (preTokenize → byte-shadow → bpe → vocab)
    func encode(_ text: String) -> [Int] {
        let segments = splitBySpecialTokens(text)
        var ids: [Int] = []
        for seg in segments {
            if seg.isSpecial {
                guard let id = specialTokens[seg.text] else {
                    fatalError("Special token '\(seg.text)' not registered in specialTokens dict")
                }
                ids.append(id)
            } else {
                for chunk in preTokenize(seg.text) {
                    let shadow = ByteShadowMap.encode(chunk)
                    for tok in bpe(shadow) {
                        guard let id = tokenToId[tok] else {
                            fatalError("Can't convert token '\(tok)' (from chunk '\(chunk)') to id")
                        }
                        ids.append(id)
                    }
                }
            }
        }
        return ids
    }

    /// 把 text 切成 [(text, isSpecial)] 段交替。
    /// 没有 specialTokenRegex 时整段视为 normal。
    private func splitBySpecialTokens(_ text: String) -> [(text: String, isSpecial: Bool)] {
        guard let regex = specialTokenRegex else {
            return text.isEmpty ? [] : [(text, false)]
        }
        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            range: NSRange(location: 0, length: nsText.length)
        )

        var result: [(text: String, isSpecial: Bool)] = []
        var lastEnd = 0
        for match in matches {
            let r = match.range
            // 中间的 normal 段
            if r.location > lastEnd {
                let normal = nsText.substring(
                    with: NSRange(location: lastEnd, length: r.location - lastEnd)
                )
                result.append((normal, false))
            }
            // special 段
            result.append((nsText.substring(with: r), true))
            lastEnd = r.location + r.length
        }
        // 末尾剩余的 normal 段
        if lastEnd < nsText.length {
            let tail = nsText.substring(
                with: NSRange(location: lastEnd, length: nsText.length - lastEnd)
            )
            result.append((tail, false))
        }
        return result
    }

    /// token id 列表 → 文本
    /// pipeline: id 反查 → concat → byte-shadow decode
    func decode(_ ids: [Int]) -> String {
        var shadow = ""
        for id in ids {
            if let tok = idToToken[id] {
                shadow += tok
            }
        }
        return ByteShadowMap.decode(shadow)
    }

    /// Qwen / cl100k 风格的 pre-tokenizer regex
    /// 把整段文本切成「词 / 标点 / 数字 / 空白」chunk
    private static let preTokenizeRegex: NSRegularExpression = {
        let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()

    private func preTokenize(_ text: String) -> [String] {
        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let matches = Self.preTokenizeRegex.matches(in: text, options: [], range: fullRange)
        return matches.map { nsText.substring(with: $0.range) }
    }

    func preTokenDebug(_ text: String) -> [(String, String)] {
        let nsText = text as NSString
        var pos = 0
        var result: [(String, String)] = []
        while pos < nsText.length {
            var matched = false
            for (name, pattern) in preTokenizePatterns {
                let searchRange = NSRange(location: pos, length: nsText.length - pos)
                if let match = pattern.firstMatch(in: text, range: searchRange),
                   match.range.location == pos
                {
                    matched = true
                    if let range = Range(match.range, in: text) {
                        let chunk = String(text[range])
                        result.append((chunk, name))
                    }
                    pos += match.range.length
                    break
                }
            }
            if !matched {
                fatalError("No pattern matched at pos=\(pos), text='\(nsText.substring(from: pos))'")
            }
        }
        return result
    }

    private let preTokenizePatterns: [(name: String, pattern: NSRegularExpression)] = [
        ("contraction", try! NSRegularExpression(pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)"#)),
        ("letters", try! NSRegularExpression(pattern: #"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"#)),
        ("digits", try! NSRegularExpression(pattern: #"\p{N}"#)),
        ("punct", try! NSRegularExpression(pattern: #" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"#)),
        ("newline", try! NSRegularExpression(pattern: #"\s*[\r\n]+"#)),
        ("trail_ws", try! NSRegularExpression(pattern: #"\s+(?!\S)"#)),
        ("whitespace", try! NSRegularExpression(pattern: #"\s+"#)),
    ]

    func bpe(_ word: String) -> [String] {
        var chars = word.map { String($0) }

        while chars.count >= 2 {
            var bestPair: Pair?
            var bestRank = Int.max
            for i in 0 ..< chars.count - 1 {
                let pair = Pair(first: chars[i], second: chars[i + 1])
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard let best = bestPair else { break }

            var next: [String] = []
            next.reserveCapacity(chars.count)
            var i = 0
            while i < chars.count {
                if i + 1 < chars.count, chars[i] == best.first, chars[i + 1] == best.second {
                    next.append(chars[i] + chars[i + 1])
                    i += 2
                } else {
                    next.append(chars[i])
                    i += 1
                }
            }
            chars = next
        }

        return chars
    }
}
