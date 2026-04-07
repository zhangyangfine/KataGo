//
//  ModelRunnerView.swift
//  KataGo Anytime
//
//  Created by Chin-Chang Yang on 2025/5/19.
//

import SwiftUI
import KataGoInterface

struct ModelRunnerView: View {
    @State private var selectedModel: NeuralNetworkModel? = nil
    @State private var katagoThread: Thread?
    @AppStorage("ModelRunnerView.selectedModelTitle") private var selectedModelTitle = ""

    var body: some View {
        Group {
            if let selectedModel {
                ContentView(selectedModel: $selectedModel)
            } else {
                ModelPickerView(selectedModel: $selectedModel)
            }
        }
        .onAppear {
            if !selectedModelTitle.isEmpty {
#if !DEBUG
                selectedModel = NeuralNetworkModel.allCases.first(
                    where: { $0.title == selectedModelTitle }
                )
#endif
            }
        }
        .onChange(of: selectedModel) { _, _ in
            if let selectedModel {
                selectedModelTitle = selectedModel.title

                let modelPath: String?
                if selectedModel.builtIn {
                    // Built-in model is bundled as .bin.gz in the app bundle
                    modelPath = Bundle.main.path(forResource: "default_model", ofType: "bin.gz")
                } else {
                    modelPath = selectedModel.downloadedURL?.path()
                }

                if let modelPath {
                    startKataGoThread(modelPath: modelPath)
                } else {
                    // Failed to get model URL, go back to the model picker view
                    self.selectedModel = nil
                }
            }
        }
    }

    private func startKataGoThread(modelPath: String) {
        // Start a thread to run KataGo GTP
        let katagoThread = Thread {
            KataGoHelper.runGtp(modelPath: modelPath)

            Task {
                await MainActor.run {
                    withAnimation {
                        selectedModel = nil
                    }
                }
            }
        }

        // Expand the stack size to resolve a stack overflow problem
        katagoThread.stackSize = 4096 * 256
        katagoThread.start()

        self.katagoThread = katagoThread
    }
}
