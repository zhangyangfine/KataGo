# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of KataGo (a strong open-source Go AI engine) with an iOS/macOS SwiftUI app that wraps the C++ engine. The project uses CoreML and Metal backends optimized for Apple's Neural Engine, providing power-efficient Go analysis on mobile devices.

## Build Commands

### iOS/macOS App (Xcode)
```bash
# Navigate to iOS project
cd ios/KataGo\ iOS

# Open in Xcode
open KataGo\ Anytime.xcodeproj

# Build from command line
xcodebuild -project "KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -configuration Debug build
```

### Running Tests
```bash
# Run unit tests from Xcode or command line
xcodebuild test -project "ios/KataGo iOS/KataGo Anytime.xcodeproj" -scheme "KataGo Anytime" -destination 'platform=iOS Simulator,name=iPhone 17'
```

### Required Resources
Before building, download model files to `ios/KataGo iOS/Resources/`:
- `default_model.bin.gz` - KataGo neural network
- `KataGoModel29x29fp16.mlpackage` - CoreML model for Neural Engine

## Architecture

### Two-Component Design

**C++ Engine (`cpp/`)**: The core KataGo engine with multiple neural network backends:
- `neuralnet/coremlbackend.{cpp,swift}` - CoreML backend for Apple Neural Engine
- `neuralnet/metalbackend.{cpp,swift}` - Metal GPU backend for macOS
- Standard backends: CUDA, OpenCL, Eigen (CPU), TensorRT

**SwiftUI App (`ios/KataGo iOS/`)**: Native iOS/macOS interface:
- `KataGoInterface/` - Framework bridging Swift to C++ via `KataGoHelper.swift`
- `KataGo iOS/` - Main app with SwiftUI views and models

### Key Swift Files

| File | Purpose |
|------|---------|
| `KataGo_iOSApp.swift` | App entry point, SwiftData container setup |
| `ContentView.swift` | Main view, GTP message processing loop |
| `GameSplitView.swift` | Navigation split view, game list sidebar |
| `KataGoModel.swift` | Board state, stones, analysis data models |
| `GobanView.swift` | Go board rendering |
| `KataGoHelper.swift` | C++ interface: `runGtp()`, `sendCommand()`, `getMessageLine()` |
| `GameRecord.swift` | SwiftData model for saved games |
| `GobanState.swift` | Game state management (editing, branching, SGF) |
| `Commentator.swift` | AI commentary using Apple FoundationModels |
| `AudioModel.swift` | Sound effects for stone placement/capture |
| `LinePlotView.swift` | Win rate/score chart with auto-play |
| `BoardLineView.swift` | Board grid lines rendering |

### Communication Pattern

The app communicates with the C++ engine via GTP (Go Text Protocol):
1. Swift sends commands via `KataGoHelper.sendCommand()`
2. C++ engine processes and queues responses
3. Swift polls `KataGoHelper.getMessageLine()` in async loop
4. `ContentView.messaging()` parses responses to update UI state

### Neural Network Backends on Apple Silicon

- **CoreML NE** (Neural Engine): Best power efficiency, ~70 visits/s on iPhone 12
- **CoreML GPU**: Slower than NE
- **Metal GPU**: Used on macOS with `metalNumSearchThreads=16`
- On iOS, limited to 2 search threads for power efficiency

## C++ Source Structure

Key directories (in dependency order):
- `core/` - Low-level utilities, hashing, threading
- `game/` - Board representation (`board.cpp`), rules, history
- `neuralnet/` - NN backends and interface (`nneval.cpp` for batching)
- `search/` - MCTS implementation (`search.cpp`), time controls
- `dataio/` - SGF parsing (`sgf.cpp`), model loading
- `command/` - User commands: `gtp.cpp`, `analysis.cpp`, `benchmark.cpp`

## SwiftData Models

- `GameRecord` - Persisted game with SGF, configuration, timestamps
- `Config` - Game settings (board size, komi, rules, commentary tone, temperature)
- Uses CloudKit for iCloud sync (container: `iCloud.chinchangyang.KataGo-iOS.tw`)

## On-Device AI Commentary

The `Commentator` class uses Apple's FoundationModels framework to generate natural language commentary for moves. Features:
- Configurable tones: technical, educational, encouraging, enthusiastic, poetic
- Analyzes win rate changes, score differences, captured/dead/endangered stones
- Uses `@Generable` struct with `LanguageModelSession` for on-device inference

## Global Settings

Stored via `@AppStorage` in `GameSplitView`:
- `GlobalSettings.soundEffect` - Enable stone placement/capture sounds
- `GlobalSettings.hapticFeedback` - Enable haptic feedback for board interactions

## GTP Commands Used

The app uses KataGo's GTP extensions including:
- `kata-analyze` - Continuous analysis with ownership, winrate
- `showboard` - Get current board state
- `printsgf` - Export game as SGF
- `play <color> <move>` - Make moves
- `kata-set-rule` - Configure rules

## Platform Support

- iOS 26+
- macOS 26+ (native, not Catalyst)
- visionOS 26+
