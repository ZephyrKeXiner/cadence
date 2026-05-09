//
//  Qwen3Forward.swift
//  Cadence
//
//  Created by Haotian Gong on 9/5/26.
//

import Foundation
import MetalPerformanceShadersGraph

class Qwen3Forward {
    let graph = MPSGraph()
    let model: Qwen3Model

    init(model: Qwen3Model) {
        self.model = model
    }

    func embedLookup(input: [Int32]) -> MPSGraphTensorData {
        let embed = graph.placeholder(
            shape: model.embedTokens.shape,
            dataType: .bFloat16,
            name: "embed_tensor"
        )
        let indices = graph.placeholder(
            shape: [NSNumber(value: input.count)],
            dataType: .int32,
            name: "indices_tensor"
        )
        let output = graph.gather(
            withUpdatesTensor: embed,
            indicesTensor: indices,
            axis: 0,
            batchDimensions: 0,
            name: "gather_tensor"
        )

        let inputData = Int2TensorData(input: input)

        let result = graph.run(
            with: Device.shared.commandQueue,
            feeds: [embed: model.embedTokens, indices: inputData],
            targetTensors: [output],
            targetOperations: nil
        )

        return result[output]!
    }

    private func Int2TensorData(input: [Int32]) -> MPSGraphTensorData {
        let buffer = input.withUnsafeBytes { rawBuffer in
            Device.shared.mtlDevice.makeBuffer(
                bytes: rawBuffer.baseAddress!,
                length: rawBuffer.count,
                options: .storageModeShared
            )
        }
        return MPSGraphTensorData(buffer!, shape: [NSNumber(value: input.count)], dataType: .int32)
    }
}
