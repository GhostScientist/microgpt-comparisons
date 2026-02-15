//
//  ContentView.swift
//  microgpt-watchos Watch App
//
//  Created by Dakota Kim on 2/14/26.
//

import SwiftUI
import Foundation
import Combine

struct ContentView: View {
    @StateObject private var model = TrainingViewModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Button {
                    model.startRun()
                } label: {
                    Text(model.isRunning ? "Training..." : "Start Training")
                        .frame(maxWidth: .infinity)
                }
                .disabled(model.isRunning)

                MetricRow(title: "Latest", value: model.latestStatus)
                MetricRow(title: "Inference", value: model.inferenceResult)
                MetricRow(title: "Try", value: "\(model.tryCount)")
                MetricRow(title: "Last Time", value: model.lastDurationText)
                MetricRow(title: "Avg Time", value: model.averageDurationText)

                if let errorMessage = model.errorMessage {
                    Text(errorMessage)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
    }
}

#Preview {
    ContentView()
}

private struct MetricRow: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption2.monospaced())
                .lineLimit(2)
                .minimumScaleFactor(0.8)
        }
    }
}

@MainActor
private final class TrainingViewModel: ObservableObject {
    @Published var latestStatus = "Idle"
    @Published var inferenceResult = "-"
    @Published var isRunning = false
    @Published var errorMessage: String?
    @Published var tryCount = 0
    @Published private var durations: [TimeInterval] = []

    private let runner = MicroGPTRunner()

    var lastDurationText: String {
        guard let value = durations.last else { return "-" }
        return formatDuration(value)
    }

    var averageDurationText: String {
        guard !durations.isEmpty else { return "-" }
        let average = durations.reduce(0, +) / Double(durations.count)
        return formatDuration(average)
    }

    func startRun() {
        guard !isRunning else { return }

        isRunning = true
        errorMessage = nil
        inferenceResult = "-"
        latestStatus = "Preparing run..."
        tryCount += 1

        let start = Date()

        Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                let inference = try await self.runner.run { status in
                    await MainActor.run {
                        self.latestStatus = status
                    }
                }
                let elapsed = Date().timeIntervalSince(start)
                self.durations.append(elapsed)
                self.inferenceResult = inference.isEmpty ? "(empty)" : inference
                self.isRunning = false
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                self.durations.append(elapsed)
                self.errorMessage = error.localizedDescription
                self.latestStatus = "Run failed"
                self.isRunning = false
            }
        }
    }

    private func formatDuration(_ value: TimeInterval) -> String {
        String(format: "%.2fs", value)
    }
}
