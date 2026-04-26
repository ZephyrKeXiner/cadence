//
//  VocabTest.swift
//  Cadence
//
//  Created by Haotian Gong on 25/4/26.
//

import Foundation

enum VocabTest {
    static func run() {
        guard let vocab = try? Vocab(
            vocabPath: TestPaths.vocabPath,
            mergesPath: TestPaths.mergesPath,
            specialTokens: Vocab.qwenSpecialTokens
        ) else {
            fatalError("Failed to read files")
        }

        print(vocab.tokenToId.count)
        print(vocab.bpeRanks.count)
        print(vocab.bpeRanks[Pair(first: "Ġ", second: "Ġ")] as Any)
        print(vocab.bpeRanks[Pair(first: "i", second: "n")] as Any)
    }
}
