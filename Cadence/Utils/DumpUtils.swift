//
//  DumpUtils.swift
//  Cadence
//
//  Created by Haotian Gong on 1/5/26.
//

import Foundation

enum DumpUtils {
    static let dumpDir = "/Users/sakruhnab1/Documents/Cadence/Models/dumps"

    static func loadDump(_ name: String) throws -> [Float] {
        let url = URL(filePath: dumpDir).appending(component: name, directoryHint: .notDirectory)
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { rawbuffer in
            Array(rawbuffer.bindMemory(to: Float.self))
        }
    }

    static func compareDump(_ output: [Float], _ dumpName: String) throws -> Float {
        let dump = try loadDump(dumpName)
        guard output.count == dump.count else {
            fatalError("size mismatch: output.count=\(output.count), dump.count=\(dump.count)")
        }
        let diffs = zip(output, dump).map { x, y in
            abs(x - y)
        }
        guard let maxDiff = diffs.max() else {
            return 0
        }
        return maxDiff
    }
}
