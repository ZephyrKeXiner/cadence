//
//  Qwen3ForwardTest.swift
//  Cadence
//
//  Created by Haotian Gong on 9/5/26.
//

import Foundation

enum Qwen3ForwardTest {
    static func run() {
        let input: [Int32] = [9707, 11, 1879, 0]
        let forward = Qwen3Forward()

        print(forward.embedLookup(input: input))
    }
}
