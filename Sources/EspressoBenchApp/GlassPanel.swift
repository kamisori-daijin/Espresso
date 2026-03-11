import SwiftUI

struct GlassPanel<Content: View>: View {
    private let contentPadding: CGFloat
    private let content: Content

    init(contentPadding: CGFloat = 20, @ViewBuilder content: () -> Content) {
        self.contentPadding = contentPadding
        self.content = content()
    }

    var body: some View {
        content
            .padding(contentPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.thinMaterial)
                    .overlay(
                        LinearGradient(
                            colors: [.white.opacity(0.26), .white.opacity(0.06)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .strokeBorder(.white.opacity(0.24), lineWidth: 1)
                    )
            )
            .shadow(color: .black.opacity(0.05), radius: 20, y: 10)
    }
}
