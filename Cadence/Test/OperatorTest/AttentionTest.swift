//
//  AttentionTest.swift
//  Cadence
//
//  Created by Haotian Gong on 22/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

enum AttentionTest {
    static func run() {
        let seqLen = 4
        let d_k = 8
        let graph = MPSGraph()

        let qData: [Float] = (0 ..< seqLen * d_k).map { Float($0) / Float(seqLen * d_k) }
        let kData: [Float] = (0 ..< seqLen * d_k).map { Float(($0 * 3) % 31) / 31.0 }
        let vData: [Float] = (0 ..< seqLen * d_k).map { Float(($0 * 7) % 29) / 29.0 }

        let qPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: d_k)],
            dataType: .float32,
            name: "qPlaceHolder"
        )
        let kPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: d_k)],
            dataType: .float32,
            name: "kPlaceHolder"
        )
        let vPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: d_k)],
            dataType: .float32,
            name: "vPlaceHolder"
        )
        let d_kTensor: MPSGraphTensor = graph.constant(Double(d_k), dataType: .float32)

        let qTensorData = TensorUtils.data(from: qData, shape: [seqLen, d_k])
        let kTensorData = TensorUtils.data(from: kData, shape: [seqLen, d_k])
        let vTensorData = TensorUtils.data(from: vData, shape: [seqLen, d_k])

        let output = Attention.apply(graph: graph, Q: qPh, K: kPh, V: vPh, d_k: d_kTensor)

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [qPh: qTensorData, kPh: kTensorData, vPh: vTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        let gpuResult = TensorUtils.readFloats(from: result[output]!, count: seqLen * d_k)

        let cpuResult = cpuAttention(
            Q: qData, K: kData, V: vData,
            seqLen: seqLen, dK: d_k
        )

        print("GPU output (row 0):", Array(gpuResult[0 ..< d_k]))
        print("CPU output (row 0):", Array(cpuResult[0 ..< d_k]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func runMulti() {
        let nHeads = 2
        let seqLen = 4
        let headDim = 8
        let graph = MPSGraph()

        let qData: [Float] = (0 ..< nHeads * seqLen * headDim)
            .map { Float($0) / Float(nHeads * seqLen * headDim) }
        let kData: [Float] = (0 ..< nHeads * seqLen * headDim).map { Float(($0 * 3) % 31) / 31.0 }
        let vData: [Float] = (0 ..< nHeads * seqLen * headDim).map { Float(($0 * 7) % 29) / 29.0 }

        let qPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "qPlaceHolder"
        )
        let kPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "kPlaceHolder"
        )
        let vPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "vPlaceHolder"
        )
        let maskPh: MPSGraphTensor = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: seqLen)],
            dataType: .float32,
            name: "maskPlaceHolder"
        )

        let qTensorData = TensorUtils.data(from: qData, shape: [nHeads, seqLen, headDim])
        let kTensorData = TensorUtils.data(from: kData, shape: [nHeads, seqLen, headDim])
        let vTensorData = TensorUtils.data(from: vData, shape: [nHeads, seqLen, headDim])
        let maskTensorData = createMask(seqLen: seqLen)

        let output = Attention.applyMultiHead(
            graph: graph,
            Q: qPh,
            K: kPh,
            V: vPh,
            mask: maskPh,
            headDim: headDim
        )

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [qPh: qTensorData, kPh: kTensorData, vPh: vTensorData, maskPh: maskTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        let gpuResult = TensorUtils.readFloats(from: result[output]!, count: nHeads * seqLen * headDim)

        let cpuResult = cpuMultiHeadAttention(
            Q: qData, K: kData, V: vData,
            nHeads: nHeads, seqLen: seqLen, headDim: headDim
        )

        print("GPU output (head 0, row 0):", Array(gpuResult[0 ..< headDim]))
        print("CPU output (head 0, row 0):", Array(cpuResult[0 ..< headDim]))

        // 也打印 head 1 row 2，多看一个位置确保 head 维正确
        let h1r2Start = 1 * seqLen * headDim + 2 * headDim
        print("GPU output (head 1, row 2):", Array(gpuResult[h1r2Start ..< h1r2Start + headDim]))
        print("CPU output (head 1, row 2):", Array(cpuResult[h1r2Start ..< h1r2Start + headDim]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    static func createMask(seqLen: Int) -> MPSGraphTensorData {
        var mask = [Float](repeating: 0, count: seqLen * seqLen)
        for i in 0 ..< seqLen {
            for j in (i + 1) ..< seqLen {
                mask[i * seqLen + j] = -1e9
            }
        }
        return TensorUtils.data(from: mask, shape: [seqLen, seqLen])
    }

    /// CPU 参考：Multi-Head Causal Attention
    /// 输入布局：Q/K/V = [nHeads, seqLen, headDim]，连续存储
    /// 输出布局：[nHeads, seqLen, headDim]
    static func cpuMultiHeadAttention(
        Q: [Float],
        K: [Float],
        V: [Float],
        nHeads: Int,
        seqLen: Int,
        headDim: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        let scale = 1.0 / sqrt(Float(headDim))
        let headStride = seqLen * headDim

        for h in 0 ..< nHeads {
            let base = h * headStride // 当前 head 的起始偏移

            for i in 0 ..< seqLen {
                // ─── 1. 打分：scores[j] = Q[i] · K[j] / sqrt(headDim) ───
                //     causal：只算 j <= i，j > i 直接视为 -inf（跳过、不进 softmax）
                var scores = [Float](repeating: 0, count: i + 1)
                for j in 0 ... i {
                    var dot: Float = 0
                    for k in 0 ..< headDim {
                        dot += Q[base + i * headDim + k] * K[base + j * headDim + k]
                    }
                    scores[j] = dot * scale
                }

                // ─── 2. 数值稳定 softmax（只在 j <= i 上归一） ───
                let maxScore = scores.max() ?? 0
                var expScores = [Float](repeating: 0, count: i + 1)
                var sumExp: Float = 0
                for j in 0 ... i {
                    let e = exp(scores[j] - maxScore)
                    expScores[j] = e
                    sumExp += e
                }
                var weights = [Float](repeating: 0, count: i + 1)
                for j in 0 ... i {
                    weights[j] = expScores[j] / sumExp
                }

                // ─── 3. 加权 V ───
                for k in 0 ..< headDim {
                    var sum: Float = 0
                    for j in 0 ... i {
                        sum += weights[j] * V[base + j * headDim + k]
                    }
                    out[base + i * headDim + k] = sum
                }
            }
        }

        return out
    }

    /// ══════════════════════════════════════════════════════════════
    /// Stage C: GQA (Grouped Query Attention)
    /// ══════════════════════════════════════════════════════════════
    static func runGQA() {
        let nHeads = 4
        let nKvHeads = 2
        let seqLen = 3
        let headDim = 4
        let nRep = nHeads / nKvHeads // = 2
        let graph = MPSGraph()

        // Q: [nHeads,    seqLen, headDim] = [4, 3, 4] = 48 elements
        // K: [nKvHeads,  seqLen, headDim] = [2, 3, 4] = 24 elements
        // V: [nKvHeads,  seqLen, headDim] = [2, 3, 4] = 24 elements
        let qData: [Float] = (0 ..< nHeads * seqLen * headDim)
            .map { Float($0) / Float(nHeads * seqLen * headDim) }
        let kData: [Float] = (0 ..< nKvHeads * seqLen * headDim)
            .map { Float(($0 * 3) % 31) / 31.0 }
        let vData: [Float] = (0 ..< nKvHeads * seqLen * headDim)
            .map { Float(($0 * 7) % 29) / 29.0 }

        let qPh = graph.placeholder(
            shape: [NSNumber(value: nHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "qPh"
        )
        let kPh = graph.placeholder(
            shape: [NSNumber(value: nKvHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "kPh"
        )
        let vPh = graph.placeholder(
            shape: [NSNumber(value: nKvHeads), NSNumber(value: seqLen), NSNumber(value: headDim)],
            dataType: .float32,
            name: "vPh"
        )
        let maskPh = graph.placeholder(
            shape: [NSNumber(value: seqLen), NSNumber(value: seqLen)],
            dataType: .float32,
            name: "maskPh"
        )

        let qTensorData = TensorUtils.data(from: qData, shape: [nHeads, seqLen, headDim])
        let kTensorData = TensorUtils.data(from: kData, shape: [nKvHeads, seqLen, headDim])
        let vTensorData = TensorUtils.data(from: vData, shape: [nKvHeads, seqLen, headDim])
        let maskTensorData = createMask(seqLen: seqLen)

        let output = Attention.applyGQA(
            graph: graph,
            Q: qPh,
            K: kPh,
            V: vPh,
            mask: maskPh,
            nHeads: nHeads,
            nKvHeads: nKvHeads,
            seqLen: seqLen,
            headDim: headDim
        )

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [qPh: qTensorData, kPh: kTensorData, vPh: vTensorData, maskPh: maskTensorData],
            targetTensors: [output],
            targetOperations: nil
        )

        let gpuResult = TensorUtils.readFloats(
            from: result[output]!,
            count: nHeads * seqLen * headDim
        )

        // CPU 参考：先手动把 K/V 按「分组连续」规则扩展到 nHeads 份，
        //          再复用已有的 cpuMultiHeadAttention
        let kExpanded = repeatKVContiguous(
            x: kData, nKvHeads: nKvHeads, seqLen: seqLen, headDim: headDim, nRep: nRep
        )
        let vExpanded = repeatKVContiguous(
            x: vData, nKvHeads: nKvHeads, seqLen: seqLen, headDim: headDim, nRep: nRep
        )
        let cpuResult = cpuMultiHeadAttention(
            Q: qData, K: kExpanded, V: vExpanded,
            nHeads: nHeads, seqLen: seqLen, headDim: headDim
        )

        // 打印几个位置
        print("─── GQA 验证 (nHeads=\(nHeads), nKvHeads=\(nKvHeads)) ───")
        print("GPU output (head 0, row 0):", Array(gpuResult[0 ..< headDim]))
        print("CPU output (head 0, row 0):", Array(cpuResult[0 ..< headDim]))

        // head 1 应该和 head 0 用相同的 KV（因为 0/2 == 1/2 == KV head 0）
        let h1Start = 1 * seqLen * headDim
        print("GPU output (head 1, row 0):", Array(gpuResult[h1Start ..< h1Start + headDim]))
        print("CPU output (head 1, row 0):", Array(cpuResult[h1Start ..< h1Start + headDim]))

        // head 2 开始换 KV head 1
        let h2Start = 2 * seqLen * headDim + 2 * headDim // head 2, row 2
        print("GPU output (head 2, row 2):", Array(gpuResult[h2Start ..< h2Start + headDim]))
        print("CPU output (head 2, row 2):", Array(cpuResult[h2Start ..< h2Start + headDim]))

        let maxDiff = zip(gpuResult, cpuResult).map { abs($0 - $1) }.max() ?? 0
        print("Max diff: \(maxDiff)")
        print(maxDiff < 1e-4 ? "✅ PASS" : "❌ FAIL")
    }

    /// 按「分组连续」规则把 KV 从 [nKvHeads, seq, d] 扩展到 [nKvHeads*nRep, seq, d]
    /// 即 [A, B] 变成 [A, A, B, B]（nRep=2 时），而不是 [A, B, A, B]
    static func repeatKVContiguous(
        x: [Float],
        nKvHeads: Int,
        seqLen: Int,
        headDim: Int,
        nRep: Int
    ) -> [Float] {
        let headStride = seqLen * headDim
        let nHeads = nKvHeads * nRep
        var out = [Float](repeating: 0, count: nHeads * headStride)

        for h in 0 ..< nHeads {
            let kvHead = h / nRep // 分组连续：head 0,1 → kv 0；head 2,3 → kv 1
            let srcStart = kvHead * headStride
            let dstStart = h * headStride
            for k in 0 ..< headStride {
                out[dstStart + k] = x[srcStart + k]
            }
        }

        return out
    }

    static func cpuAttention(
        Q: [Float],
        K: [Float],
        V: [Float],
        seqLen: Int,
        dK: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: seqLen * dK)
        let scale = 1.0 / sqrt(Float(dK))

        for i in 0 ..< seqLen {
            // ─── 1. 算打分：scores[j] = Q[i] · K[j] / sqrt(dK) ───
            var scores = [Float](repeating: 0, count: seqLen)
            for j in 0 ..< seqLen {
                var dot: Float = 0
                for k in 0 ..< dK {
                    dot += Q[i * dK + k] * K[j * dK + k]
                }
                scores[j] = dot * scale
            }

            // ─── 2. 数值稳定版 softmax ───
            // 减 max 避免 exp 溢出：
            //   softmax(x)_j = exp(x_j) / Σ exp(x_l)
            //                = exp(x_j - M) / Σ exp(x_l - M)     其中 M = max(x)
            // 这样 exp 的最大参数是 0，不会爆
            let maxScore = scores.max() ?? 0
            var expScores = [Float](repeating: 0, count: seqLen)
            var sumExp: Float = 0
            for j in 0 ..< seqLen {
                let e = exp(scores[j] - maxScore)
                expScores[j] = e
                sumExp += e
            }
            var weights = [Float](repeating: 0, count: seqLen)
            for j in 0 ..< seqLen {
                weights[j] = expScores[j] / sumExp
            }

            // ─── 3. 加权 V：out[i, k] = Σ_j weights[j] * V[j, k] ───
            for k in 0 ..< dK {
                var sum: Float = 0
                for j in 0 ..< seqLen {
                    sum += weights[j] * V[j * dK + k]
                }
                out[i * dK + k] = sum
            }
        }

        return out
    }
}
