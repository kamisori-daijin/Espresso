import SwiftUI

enum MetricCardEmphasis {
    case primary
    case secondary
}

struct MetricCardView: View {
    let title: String
    let value: String
    let detail: String?
    let tint: Color
    let emphasis: MetricCardEmphasis

    init(
        title: String,
        value: String,
        detail: String? = nil,
        tint: Color,
        emphasis: MetricCardEmphasis = .secondary
    ) {
        self.title = title
        self.value = value
        self.detail = detail
        self.tint = tint
        self.emphasis = emphasis
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(valueFont)
                .foregroundStyle(tint)
                .monospacedDigit()
            if let detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(minHeight: emphasis == .primary ? 106 : 94, alignment: .topLeading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(tint.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(tint.opacity(0.14), lineWidth: 1)
                )
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(title))
        .accessibilityValue(Text([value, detail].compactMap { $0 }.joined(separator: ", ")))
    }

    private var valueFont: Font {
        switch emphasis {
        case .primary:
            return .system(size: 28, weight: .semibold, design: .rounded)
        case .secondary:
            return .system(size: 22, weight: .semibold, design: .rounded)
        }
    }
}
