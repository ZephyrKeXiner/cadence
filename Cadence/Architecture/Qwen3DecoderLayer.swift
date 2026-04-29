//
//  Qwen3DecoderLayer.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation
import MetalPerformanceShadersGraph

final class Qwen3DecoderLayer {
    let inputLayernorm: MPSGraphTensorData
    let postAttentionLayernorm: MPSGraphTensorData
    let qProj: MPSGraphTensorData
    let kProj: MPSGraphTensorData
    let vProj: MPSGraphTensorData
    let oProj: MPSGraphTensorData
    let kNorm: MPSGraphTensorData
    let qNorm: MPSGraphTensorData
    let gateProj: MPSGraphTensorData
    let upProj: MPSGraphTensorData
    let downProj: MPSGraphTensorData

    init(layerIdx: Int, router: SafeTensorsRouter) {
        func load(_ subname: String) -> MPSGraphTensorData {
            let fullName = "model.layers.\(layerIdx).\(subname)"
            guard let arr = router.loadAsGPU(fullName) else {
                fatalError("Missing weight: \(fullName)")
            }
            return arr
        }

        inputLayernorm = load("input_layernorm.weight")
        postAttentionLayernorm = load("post_attention_layernorm.weight")
        qProj = load("self_attn.q_proj.weight")
        kProj = load("self_attn.k_proj.weight")
        vProj = load("self_attn.v_proj.weight")
        oProj = load("self_attn.o_proj.weight")
        kNorm = load("self_attn.k_norm.weight")
        qNorm = load("self_attn.q_norm.weight")
        gateProj = load("mlp.gate_proj.weight")
        upProj = load("mlp.up_proj.weight")
        downProj = load("mlp.down_proj.weight")
    }
}
