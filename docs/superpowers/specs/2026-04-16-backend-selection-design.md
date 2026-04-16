# Backend and CoreML Board Size Selection

## Problem

When using CoreML/NE, changing board size causes long response times because CoreML models must be compiled (or loaded from compiled `.mlmodelc`) for each board size. Users have no way to choose between MPS/GPU (instant start, lower power efficiency) and CoreML/NE (compilation delay on first use, better power efficiency and throughput). Users also cannot control the compiled board size for CoreML.

## Solution

Add a per-model backend configuration sheet accessible via a gear icon on each model row in ModelPickerView. Users choose between MPS/GPU and CoreML/NE, and when CoreML/NE is selected, choose the compiled board size.

## Data Model

### New Per-Model Settings

Each `NeuralNetworkModel` gains two persisted properties:

- **`backend`**: Enum `BackendChoice` with cases `.mpsGPU` (device 0) and `.coremlNE` (device 100).
  - Default: `.coremlNE` on iOS/visionOS, `.mpsGPU` on macOS.
- **`coremlBoardSize`**: Int from {9, 13, 19, 37}. Default: 19.
  - Only meaningful when backend is `.coremlNE`.

### Persistence

Settings are stored via `@AppStorage` keyed by model filename (e.g., `"backend_default_model.bin.gz"`, `"coremlBoardSize_default_model.bin.gz"`).

### GTP Config Mapping

**When backend is MPS/GPU:**
- `metalDeviceToUseThread0 = 0`
- `maxBoardSizeForNNBuffer` = model's full `nnLen` (37)
- `requireExactNNLen` follows existing default

**When backend is CoreML/NE:**
- `metalDeviceToUseThread0 = 100`
- `maxBoardSizeForNNBuffer` = chosen `coremlBoardSize`
- `requireExactNNLen = false` (allows smaller boards via masking)

### Effective maxBoardLength

Passed to `GobanView` for the board-size blocking check:
- CoreML/NE: `coremlBoardSize`
- MPS/GPU: model's `nnLen` (37)

## UI Design

### Model Row (ModelPickerView)

Each model row gets a gear icon button (SF Symbol `gearshape`). Tapping it presents a `.sheet`.

### Backend Configuration Sheet

Contents:
1. **Title:** Model name
2. **Backend picker:** `Picker` with `.segmented` style, two options: "MPS/GPU" and "CoreML/NE"
3. **CoreML board size picker:** Only visible when CoreML/NE is selected. `Picker` with options: 9x9, 13x13, 19x19, 37x37. Label: "Compiled Board Size"
4. **Explanatory text:**
   - MPS/GPU: "Responsive. No compilation needed."
   - CoreML/NE: "Power-efficient. First launch for a board size takes time to compile."
5. **Done button** to dismiss

Settings save immediately but only take effect on next engine launch (Play tap).

### Board Size Blocking (Existing)

The existing `ContentUnavailableView` at `GobanView.swift:64-72` blocks games whose board dimensions exceed `maxBoardLength`. No new blocking UI needed — just wire `maxBoardLength` to the effective value based on backend choice.

## Engine Launch Integration

When the user taps Play in `ModelDetailView`, the engine launch path reads the model's persisted backend and board size settings to build GTP config overrides passed to `KataGoHelper.runGtp()`.

No changes to the Quit flow. Switching backend or board size requires quitting and re-launching.

## Platform Defaults

| Platform    | Default Backend | Default CoreML Board Size |
|-------------|----------------|--------------------------|
| iOS         | CoreML/NE      | 19                       |
| visionOS    | CoreML/NE      | 19                       |
| macOS       | MPS/GPU        | 19 (hidden)              |

## Edge Cases

1. **Games exceeding compiled board size:** The existing `ContentUnavailableView` shows "too large board size" message. Game data is preserved — playable after switching to MPS/GPU or increasing compiled board size.

2. **Built-in vs downloaded models:** Both use the same per-model settings mechanism, keyed by filename.

3. **Crash recovery:** Existing `EngineLifecycle` crash recovery is unaffected. CoreML compilation crashes trigger the recovery banner in ModelPickerView.

4. **Threading config:** Not user-configurable. iOS keeps `metalNumSearchThreads=2`, macOS keeps `metalNumSearchThreads=16`, regardless of backend choice.

## Key Files to Modify

| File | Change |
|------|--------|
| `NeuralNetworkModel.swift` | Add `BackendChoice` enum, per-model `@AppStorage` properties |
| `ModelPickerView.swift` | Add gear icon to model rows, new `BackendConfigSheet` view |
| `ModelRunnerView.swift` / `KataGoHelper.swift` | Read backend settings, pass as GTP config overrides |
| `GobanView.swift` | Wire `maxBoardLength` to effective value (already has blocking logic) |
| `GameSplitView.swift` | Pass effective `maxBoardLength` based on backend + board size |
| `ContentView.swift` | Thread through effective `maxBoardLength` |
