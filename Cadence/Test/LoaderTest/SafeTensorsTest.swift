//
//  SafeTensorsTest.swift
//  Cadence
//
//  Created by Haotian Gong on 26/4/26.
//

import Foundation

enum SafeTensorsTest {
    static func run() {
        let st: SafeTensors
        do {
            st =
                try SafeTensors(
                    filePath: "/Users/sakruhnab1/Documents/Cadence/Models/fixtures/tiny.safetensors"
                )
        } catch {
            print("❌ load failed: \(error)")
            return
        }

        print("─── SafeTensors fixture 测试 ───")
        print("tensors:", st.tensors.keys.sorted())

        // embed: shape [4, 5], expected [0, 1, 2, ..., 19]
        let embed = st.loadAsFloat32("embed")
        print("embed:", embed)
        let expected = (0 ..< 20).map { Float($0) }
        print(embed == expected ? "✅ embed 正确" : "❌ embed 不匹配")

        // q_proj: shape [3, 4], expected [0.0, 0.1, 0.2, ..., 1.1]
        let qProj = st.loadAsFloat32("q_proj")
        print("q_proj:", qProj)
        let qExpected = (0 ..< 12).map { Float($0) * 0.1 }
        let maxDiff = zip(qProj, qExpected).map { abs($0 - $1) }.max() ?? 0
        print("q_proj max diff: \(maxDiff)  \(maxDiff < 1e-6 ? "✅" : "❌")")
    }
}
