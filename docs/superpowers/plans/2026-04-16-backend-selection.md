# Backend and CoreML Board Size Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users choose between MPS/GPU and CoreML/NE backends per model, and configure the compiled CoreML board size, via a gear-icon configuration sheet in the model picker.

**Architecture:** Add a `BackendChoice` enum and per-model `@AppStorage` settings keyed by model filename. A new `BackendConfigSheet` view presents backend and board size pickers. The engine launch path (`KataGoHelper.runGtp`) accepts new parameters for `maxBoardSizeForNNBuffer` and `requireExactNNLen`. The existing board-size blocking in `GobanView` is rewired to use the effective max board length derived from backend choice.

**Tech Stack:** SwiftUI, `@AppStorage`, KataGo C++ GTP override-config

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ios/KataGo iOS/KataGo iOS/BackendChoice.swift` | **Create** | `BackendChoice` enum + `BackendSettings` helper for per-model `@AppStorage` |
| `ios/KataGo iOS/KataGo iOS/BackendConfigSheet.swift` | **Create** | Sheet view with backend picker + CoreML board size picker |
| `ios/KataGo iOS/KataGo iOS/ModelPickerView.swift` | **Modify** | Add gear icon button to each model row |
| `ios/KataGo iOS/KataGoInterface/KataGoHelper.swift` | **Modify** | Accept `maxBoardSizeForNNBuffer` and `requireExactNNLen` params |
| `ios/KataGo iOS/KataGoInterface/KataGoCpp.hpp` | **Modify** | Add new params to `KataGoRunGtp` signature |
| `ios/KataGo iOS/KataGoInterface/KataGoCpp.cpp` | **Modify** | Pass new override-config args |
| `ios/KataGo iOS/KataGo iOS/ModelRunnerView.swift` | **Modify** | Read backend settings, pass to engine launch |
| `ios/KataGo iOS/KataGo iOS/GameSplitView.swift` | **Modify** | Derive `maxBoardLength` from backend settings |
| `ios/KataGo iOS/KataGo iOS/ContentView.swift` | **Modify** | Thread effective `maxBoardLength` through |

---

### Task 1: Create BackendChoice enum and BackendSettings

**Files:**
- Create: `ios/KataGo iOS/KataGo iOS/BackendChoice.swift`

- [ ] **Step 1: Create BackendChoice.swift**

```swift
//
//  BackendChoice.swift
//  KataGo Anytime
//

import Foundation

enum BackendChoice: String, CaseIterable, Identifiable {
    case mpsGPU = "MPS/GPU"
    case coremlNE = "CoreML/NE"

    var id: String { rawValue }

    var metalDeviceToUse: Int {
        switch self {
        case .mpsGPU: return 0
        case .coremlNE: return 100
        }
    }

    static var platformDefault: BackendChoice {
        #if os(macOS)
        return .mpsGPU
        #else
        return .coremlNE
        #endif
    }
}

enum CoreMLBoardSize: Int, CaseIterable, Identifiable {
    case nine = 9
    case thirteen = 13
    case nineteen = 19
    case thirtySevenMax = 37

    var id: Int { rawValue }

    var label: String {
        "\(rawValue)x\(rawValue)"
    }
}

struct BackendSettings {
    private let modelFileName: String

    init(modelFileName: String) {
        self.modelFileName = modelFileName
    }

    var backendKey: String { "backend_\(modelFileName)" }
    var boardSizeKey: String { "coremlBoardSize_\(modelFileName)" }

    var backend: BackendChoice {
        get {
            if let raw = UserDefaults.standard.string(forKey: backendKey),
               let choice = BackendChoice(rawValue: raw) {
                return choice
            }
            return BackendChoice.platformDefault
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: backendKey)
        }
    }

    var coremlBoardSize: CoreMLBoardSize {
        get {
            let raw = UserDefaults.standard.integer(forKey: boardSizeKey)
            if raw != 0, let size = CoreMLBoardSize(rawValue: raw) {
                return size
            }
            return .nineteen
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: boardSizeKey)
        }
    }

    var effectiveMaxBoardLength: Int {
        switch backend {
        case .coremlNE: return coremlBoardSize.rawValue
        case .mpsGPU: return 37
        }
    }

    var requireExactNNLen: Bool {
        switch backend {
        case .coremlNE: return false
        case .mpsGPU: return false
        }
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add "ios/KataGo iOS/KataGo iOS/BackendChoice.swift"
git commit -m "feat: add BackendChoice enum and BackendSettings for per-model backend config"
```

---

### Task 2: Create BackendConfigSheet view

**Files:**
- Create: `ios/KataGo iOS/KataGo iOS/BackendConfigSheet.swift`

- [ ] **Step 1: Create BackendConfigSheet.swift**

```swift
//
//  BackendConfigSheet.swift
//  KataGo Anytime
//

import SwiftUI

struct BackendConfigSheet: View {
    let model: NeuralNetworkModel
    @State private var backend: BackendChoice
    @State private var coremlBoardSize: CoreMLBoardSize
    @Environment(\.dismiss) private var dismiss

    init(model: NeuralNetworkModel) {
        self.model = model
        let settings = BackendSettings(modelFileName: model.fileName)
        self._backend = State(initialValue: settings.backend)
        self._coremlBoardSize = State(initialValue: settings.coremlBoardSize)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Backend", selection: $backend) {
                        ForEach(BackendChoice.allCases) { choice in
                            Text(choice.rawValue).tag(choice)
                        }
                    }
                    .pickerStyle(.segmented)
                } footer: {
                    switch backend {
                    case .mpsGPU:
                        Text("Responsive. No compilation needed.")
                    case .coremlNE:
                        Text("Power-efficient. First launch for a board size takes time to compile.")
                    }
                }

                if backend == .coremlNE {
                    Section("Compiled Board Size") {
                        Picker("Board Size", selection: $coremlBoardSize) {
                            ForEach(CoreMLBoardSize.allCases) { size in
                                Text(size.label).tag(size)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                }
            }
            .navigationTitle(model.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onChange(of: backend) { _, newValue in
                var settings = BackendSettings(modelFileName: model.fileName)
                settings.backend = newValue
            }
            .onChange(of: coremlBoardSize) { _, newValue in
                var settings = BackendSettings(modelFileName: model.fileName)
                settings.coremlBoardSize = newValue
            }
        }
    }
}

#Preview {
    BackendConfigSheet(model: NeuralNetworkModel.allCases[0])
}
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add "ios/KataGo iOS/KataGo iOS/BackendConfigSheet.swift"
git commit -m "feat: add BackendConfigSheet view with backend and board size pickers"
```

---

### Task 3: Add gear icon to ModelPickerView model rows

**Files:**
- Modify: `ios/KataGo iOS/KataGo iOS/ModelPickerView.swift:170-192`

- [ ] **Step 1: Add gear button and sheet state to ModelPickerView**

In `ModelPickerView.swift`, add a `@State` property and modify the model row to include a gear icon that opens the configuration sheet.

Replace the `ForEach` block (lines 171-192) with:

```swift
                    ForEach(NeuralNetworkModel.allCases) { model in
                        if model.visible,
                           let destinationURL = model.downloadedURL {
                            NavigationLink {
                                ModelDetailView(
                                    model: model,
                                    downloader: Downloader(destinationURL: destinationURL),
                                    selectedModel: $selectedModel
                                )
                            } label: {
                                HStack {
                                    Text(model.title)
                                    if model.title == crashedModelTitle {
                                        Spacer()
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .foregroundStyle(.orange)
                                            .accessibilityLabel("Did not finish loading last time")
                                    }
                                }
                            }
                            .swipeActions(edge: .trailing) {
                                Button {
                                    configSheetModel = model
                                } label: {
                                    Label("Settings", systemImage: "gearshape")
                                }
                                .tint(.gray)
                            }
                        }
                    }
```

Also add the state property near line 151 (after `@State private var selectedModelID: UUID?`):

```swift
    @State private var configSheetModel: NeuralNetworkModel?
```

And add the `.sheet` modifier to the `NavigationStack` (after the closing of `.navigationTitle("Select a Model")`, before the closing `}` of the NavigationStack):

```swift
            .sheet(item: $configSheetModel) { model in
                BackendConfigSheet(model: model)
            }
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add "ios/KataGo iOS/KataGo iOS/ModelPickerView.swift"
git commit -m "feat: add gear icon swipe action to model rows for backend configuration"
```

---

### Task 4: Extend C++ interface to accept new config parameters

**Files:**
- Modify: `ios/KataGo iOS/KataGoInterface/KataGoCpp.hpp:15-20`
- Modify: `ios/KataGo iOS/KataGoInterface/KataGoCpp.cpp:70-97`

- [ ] **Step 1: Update KataGoCpp.hpp signature**

Replace the `KataGoRunGtp` declaration (lines 15-20) with:

```cpp
void KataGoRunGtp(string modelPath,
                  string humanModelPath,
                  string configPath,
                  int metalDeviceToUse,
                  int numSearchThreads,
                  int nnMaxBatchSize,
                  int maxBoardSizeForNNBuffer,
                  bool requireExactNNLen);
```

- [ ] **Step 2: Update KataGoCpp.cpp implementation**

Replace the `KataGoRunGtp` function (lines 70-97) with:

```cpp
void KataGoRunGtp(string modelPath,
                  string humanModelPath,
                  string configPath,
                  int metalDeviceToUse,
                  int numSearchThreads,
                  int nnMaxBatchSize,
                  int maxBoardSizeForNNBuffer,
                  bool requireExactNNLen) {
    // Replace the global cout object with the custom one
    cout.rdbuf(&tsbFromKataGo);

    // Replace the global cin object with the custom one
    cin.rdbuf(&tsbToKataGo);

    vector<string> subArgs;

    // Call the main command gtp
    subArgs.push_back(string("gtp"));
    subArgs.push_back(string("-model"));
    subArgs.push_back(modelPath);
    subArgs.push_back(string("-human-model"));
    subArgs.push_back(humanModelPath);
    subArgs.push_back(string("-config"));
    subArgs.push_back(configPath);
    subArgs.push_back(string("-override-config metalDeviceToUseThread0=") + to_string(metalDeviceToUse));
    subArgs.push_back(string("-override-config metalUseFP16=true"));
    subArgs.push_back(string("-override-config numSearchThreads=") + to_string(numSearchThreads));
    subArgs.push_back(string("-override-config nnMaxBatchSize=") + to_string(nnMaxBatchSize));
    subArgs.push_back(string("-override-config maxBoardSizeForNNBuffer=") + to_string(maxBoardSizeForNNBuffer));
    subArgs.push_back(string("-override-config requireMaxBoardSize=") + (requireExactNNLen ? "true" : "false"));
    MainCmds::gtp(subArgs);
}
```

- [ ] **Step 3: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add "ios/KataGo iOS/KataGoInterface/KataGoCpp.hpp" "ios/KataGo iOS/KataGoInterface/KataGoCpp.cpp"
git commit -m "feat: extend KataGoRunGtp with maxBoardSizeForNNBuffer and requireExactNNLen params"
```

---

### Task 5: Update KataGoHelper.swift to pass backend settings

**Files:**
- Modify: `ios/KataGo iOS/KataGoInterface/KataGoHelper.swift:10-48`

- [ ] **Step 1: Update KataGoHelper.runGtp signature and implementation**

Replace the entire `KataGoHelper` class (lines 10-63) with:

```swift
public class KataGoHelper {

#if os(macOS)
    static let metalNumSearchThreads = 16
    static let metalNnMaxBatchSize = 8
#else
    static let metalNumSearchThreads = 2
    static let metalNnMaxBatchSize = 1
#endif

    public class func runGtp(modelPath: String? = nil,
                             metalDeviceToUse: Int = BackendChoice.platformDefault.metalDeviceToUse,
                             maxBoardSizeForNNBuffer: Int = 37,
                             requireExactNNLen: Bool = false) {
        let mainBundle = Bundle.main
        let modelName = "default_model"
        let modelExt = "bin.gz"

        let mainModelPath = modelPath ?? mainBundle.path(forResource: modelName,
                                                         ofType: modelExt)

        let humanModelName = "b18c384nbt-humanv0"
        let humanModelExt = "bin.gz"

        let humanModelPath = mainBundle.path(forResource: humanModelName,
                                             ofType: humanModelExt)

        let configName = "default_gtp"
        let configExt = "cfg"

        let configPath = mainBundle.path(forResource: configName,
                                         ofType: configExt)

        KataGoRunGtp(std.string(mainModelPath ?? "Contents/Resources/default_model.bin.gz"),
                     std.string(humanModelPath ?? "Contents/Resources/b18c384nbt-humanv0.bin.gz"),
                     std.string(configPath ?? "Contents/Resources/default_gtp.cfg"),
                     Int32(metalDeviceToUse),
                     Int32(metalNumSearchThreads),
                     Int32(metalNnMaxBatchSize),
                     Int32(maxBoardSizeForNNBuffer),
                     requireExactNNLen)
    }

    public class func getMessageLine() -> String {
        let cppLine = KataGoGetMessageLine()

        return String(cppLine)
    }

    public class func sendCommand(_ command: String) {
        KataGoSendCommand(std.string(command))
    }

    public class func sendMessage(_ message: String) {
        KataGoSendMessage(std.string(message))
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add "ios/KataGo iOS/KataGoInterface/KataGoHelper.swift"
git commit -m "feat: pass backend settings through KataGoHelper.runGtp to C++ engine"
```

---

### Task 6: Update ModelRunnerView to read backend settings and pass to engine

**Files:**
- Modify: `ios/KataGo iOS/KataGo iOS/ModelRunnerView.swift:69-95`

- [ ] **Step 1: Update ModelRunnerView.onChange and startKataGoThread**

Replace `startKataGoThread` (lines 103-121) with:

```swift
    private func startKataGoThread(modelPath: String,
                                   metalDeviceToUse: Int,
                                   maxBoardSizeForNNBuffer: Int,
                                   requireExactNNLen: Bool) {
        let katagoThread = Thread {
            KataGoHelper.runGtp(modelPath: modelPath,
                                metalDeviceToUse: metalDeviceToUse,
                                maxBoardSizeForNNBuffer: maxBoardSizeForNNBuffer,
                                requireExactNNLen: requireExactNNLen)

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
```

Replace the `startKataGoThread(modelPath: modelPath)` call in `.onChange(of: selectedModel)` (line 94) with:

```swift
            let settings = BackendSettings(modelFileName: newValue.fileName)
            startKataGoThread(
                modelPath: modelPath,
                metalDeviceToUse: settings.backend.metalDeviceToUse,
                maxBoardSizeForNNBuffer: settings.effectiveMaxBoardLength,
                requireExactNNLen: settings.requireExactNNLen
            )
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd "ios/KataGo iOS" && xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add "ios/KataGo iOS/KataGo iOS/ModelRunnerView.swift"
git commit -m "feat: read per-model backend settings in ModelRunnerView for engine launch"
```

---

### Task 7: Wire effective maxBoardLength through GameSplitView and ContentView

**Files:**
- Modify: `ios/KataGo iOS/KataGo iOS/GameSplitView.swift:84,153`
- Modify: `ios/KataGo iOS/KataGo iOS/ContentView.swift:42-46`

- [ ] **Step 1: Update GameSplitView to derive maxBoardLength from backend settings**

In `GameSplitView.swift`, replace line 84:

```swift
                    maxBoardLength: selectedModel?.nnLen ?? 19,
```

with:

```swift
                    maxBoardLength: selectedModel.map { BackendSettings(modelFileName: $0.fileName).effectiveMaxBoardLength } ?? 19,
```

Replace line 153:

```swift
                         maxBoardLength: selectedModel?.nnLen ?? 19,
```

with:

```swift
                         maxBoardLength: selectedModel.map { BackendSettings(modelFileName: $0.fileName).effectiveMaxBoardLength } ?? 19,
```

- [ ] **Step 2: Build all three platforms**

Run:
```bash
cd "ios/KataGo iOS"

xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' -configuration Debug 2>&1 | tail -5

xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=macOS' -configuration Debug 2>&1 | tail -5

xcodebuild build -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=visionOS Simulator,name=Apple Vision Pro' -configuration Debug 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **` for all three.

- [ ] **Step 3: Run tests**

Run:
```bash
xcodebuild test -project "ios/KataGo iOS/KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17' 2>&1 | tail -10
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add "ios/KataGo iOS/KataGo iOS/GameSplitView.swift"
git commit -m "feat: derive maxBoardLength from per-model backend settings"
```
