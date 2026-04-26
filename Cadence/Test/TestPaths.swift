//
//  TestPaths.swift
//  Cadence
//
//  集中管理测试用到的文件路径。
//  切换模型 / 移动目录时只改这里一处。
//

import Foundation

enum TestPaths {
    /// 当前测试用的模型目录
    static let modelDir = "/Users/sakruhnab1/Documents/Cadence/Models/Qwen3-4B"

    /// fixtures（自造的小测试文件）
    static let fixturesDir = "/Users/sakruhnab1/Documents/Cadence/Models/fixtures"

    /// ─── Tokenizer 文件 ───
    static var vocabPath: String {
        modelDir + "/vocab.json"
    }

    static var mergesPath: String {
        modelDir + "/merges.txt"
    }

    /// ─── 权重文件 ───
    /// safetensors index（多 shard 时的路由表）
    static var weightsIndexPath: String {
        modelDir + "/model.safetensors.index.json"
    }

    /// 第 N 个 shard（n 从 1 开始）
    static func weightsShardPath(_ n: Int, of total: Int) -> String {
        let nStr = String(format: "%05d", n)
        let totalStr = String(format: "%05d", total)
        return modelDir + "/model-\(nStr)-of-\(totalStr).safetensors"
    }

    /// ─── fixtures ───
    static var tinySafetensorsPath: String {
        fixturesDir + "/tiny.safetensors"
    }
}
