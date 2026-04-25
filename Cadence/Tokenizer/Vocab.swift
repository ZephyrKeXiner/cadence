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
