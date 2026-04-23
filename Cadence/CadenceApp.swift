//
//  CadenceApp.swift
//  Cadence
//
//  Created by Haotian Gong on 17/4/26.
//

import SwiftUI

@main
struct CadenceApp: App {
    init() {
//        MatmulTest.run()
//        print("")
//        RMSNormTest().run()
//        RoPEPropertyTest().run()
//        LayerNormTest.run()
//        SWiGLUTest.run()
//        AttentionPerfTest.run()
        AttentionTest.runGQA()
//        print("")
//        RoPETest().run()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
