//
//  CadenceApp.swift
//  Cadence
//
//  Created by 龚浩天 on 17/4/26.
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
        SWiGLUTest.run()
//        print("")
//        RoPETest().run()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
