//
//  Qwen3DecoderLayerTest.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

// import Foundation
//
// enum Qwen3DecoderLayerTest {
//    static func run() {
//        let router: SafeTensorsRouter
//        do {
//            router = try SafeTensorsRouter(indexPath: TestPaths.weightsIndexPath)
//        } catch {
//            fatalError("route 失败：\(error)")
//        }
//        let layer0 = Qwen3DecoderLayer(layerIdx: 0, router: router)
//
//        // 几个尺寸 sanity check
//        assert(layer0.inputLayernorm.count == 2560)
//        assert(layer0.qProj.count == 4096 * 2560)
//        assert(layer0.qNorm.count == 128)
//        print("layer 0 加载完成")
//
//        // 加载最后一层（跨 shard 测试）
//        let layer35 = Qwen3DecoderLayer(layerIdx: 35, router: router)
//        assert(layer35.qProj.count == 4096 * 2560)
//        print("layer 35 也加载完成")
//    }
// }
