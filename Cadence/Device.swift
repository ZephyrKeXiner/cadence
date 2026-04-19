//
//  Device.swift
//  Cadence
//
//  Created by 龚浩天 on 19/4/26.
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

final class Device {
    static let shared = Device()

    let mtlDevice: MTLDevice
    let commandQueue: MTLCommandQueue
    let graphDevice: MPSGraphDevice

    private init() {
        guard let mtl = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available")
        }

        guard let commandQueue = mtl.makeCommandQueue() else {
            fatalError("Can't create command queue")
        }

        mtlDevice = mtl
        self.commandQueue = commandQueue
        graphDevice = MPSGraphDevice(mtlDevice: mtl)
        print("Device initialized: \(mtl.name)")
    }
}
