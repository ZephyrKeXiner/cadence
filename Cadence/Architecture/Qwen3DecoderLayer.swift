//
//  Qwen3DecoderLayer.swift
//  Cadence
//
//  Created by Haotian Gong on 29/4/26.
//

import Foundation

final class Qwen3DecoderLayer {
    let inputLayernorm: [Float]
    let postAttentionLayernorm: [Float]
    let qProj: [Float]
    let kProj: [Float]
    let vProj: [Float]
    let oProj: [Float]
    let kNorm: [Float]
    let qNorm: [Float]
    let gateProj: [Float]
    let upProj: [Float]
    let downProj: [Float]

    init(layerIdx: Int, router: SafeTensorsRouter) {
        func load(_ subname: String) -> [Float] {
            let fullName = "model.layers.\(layerIdx).\(subname)"
            guard let arr = router.loadAsFloat32(fullName) else {
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
