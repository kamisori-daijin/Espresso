// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SimpleInference",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(path: "../../"),
    ],
    targets: [
        .executableTarget(
            name: "SimpleInference",
            dependencies: [
                .product(name: "RealModelInference", package: "Espresso"),
                .product(name: "ModelSupport",       package: "Espresso"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
