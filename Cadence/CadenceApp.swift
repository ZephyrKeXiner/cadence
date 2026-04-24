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
        ByteShadowMapTest.run()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
