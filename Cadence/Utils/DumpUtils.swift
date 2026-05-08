//
//  DumpUtils.swift
//  Cadence
//
//  Created by Haotian Gong on 1/5/26.
//

import Foundation

enum DumpUtils {
    static let dumpDir = "/Users/sakruhnab1/Documents/Cadence/Models/dumps"
    
    static func loadDump(from name: String) throws -> [Float]{
        let url = URL(filePath: dumpDir).appending(component: name, directoryHint: .notDirectory)
        let data = try Data(contentsOf: url)
        let value = data.withUnsafeBytes { rawbuffer in
            rawbuffer.load(as: [Float].self)
        }
        
        return value
    }
    
    static func compareDump(_ output: [Float], _ dumpName: String) -> Float {
        let dump = loadDump(from: dumpName)
        let diffArray = output.map { x in
            
        }
    }
}
