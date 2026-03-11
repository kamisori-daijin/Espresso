import SwiftUI

@main
struct EspressoBenchApp: App {
    @State private var model = BenchAppModel()

    var body: some Scene {
        WindowGroup("Espresso Bench") {
            ContentView(model: model)
                .frame(minWidth: 1280, minHeight: 860)
        }
        .defaultSize(width: 1480, height: 920)
        .windowToolbarStyle(.unified)
    }
}
