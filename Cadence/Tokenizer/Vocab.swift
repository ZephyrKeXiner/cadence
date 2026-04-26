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
    let tokenToId: [String: Int] // "Ġhello" → 24082       (从 vocab.json 来)
    let idToToken: [Int: String] // 24082 → "Ġhello"       (反向，decode 要用)
    let bpeRanks: [Pair: Int] // ("Ġ","t") → 3          (从 merges.txt 来)

    init(vocabPath: String, mergesPath: String) throws {
        let vocabData = try Data(contentsOf: URL(filePath: vocabPath))
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        tokenToId = vocab
        idToToken = Dictionary(uniqueKeysWithValues: vocab.map { token, id in (id, token) })

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
    }

    /// 文本 → token id 列表
    /// pipeline: pre-tokenize → byte-shadow → bpe → vocab lookup
    func encode(_ text: String) -> [Int] {
        let chunks = preTokenize(text)
        var ids: [Int] = []
        for chunk in chunks {
            let shadow = ByteShadowMap.encode(chunk)
            let subTokens = bpe(shadow)
            for tok in subTokens {
                if let id = tokenToId[tok] {
                    ids.append(id)
                } else {
                    fatalError("Can't convert token '\(tok)' (from chunk '\(chunk)') to id")
                }
            }
        }
        return ids
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
        let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()

    private func preTokenize(_ text: String) -> [String] {
        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let matches = Self.preTokenizeRegex.matches(in: text, options: [], range: fullRange)
        return matches.map { nsText.substring(with: $0.range) }
    }

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
