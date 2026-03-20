//
//  ManualView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2024/8/27.
//

import SwiftUI

// MARK: - Top-level Manual View

struct ManualView: View {
    var body: some View {
        List {
            NavigationLink(destination: ManualGettingStartedView()) {
                Label("Getting Started", systemImage: "star")
            }
            NavigationLink(destination: ManualPlayAgainstAIView()) {
                Label("Playing Against AI (9×9)", systemImage: "person.2")
            }
            NavigationLink(destination: ManualBoardControlsView()) {
                Label("Board & Controls", systemImage: "hand.tap")
            }
            NavigationLink(destination: ManualAnalysisView()) {
                Label("Analysis Tools", systemImage: "chart.line.uptrend.xyaxis")
            }
            NavigationLink(destination: ManualConfigurationView()) {
                Label("Configuration", systemImage: "gearshape")
            }
            NavigationLink(destination: ManualGameManagementView()) {
                Label("Managing Games", systemImage: "folder")
            }
            NavigationLink(destination: ManualCommentaryView()) {
                Label("AI Commentary", systemImage: "bubble.left.and.bubble.right")
            }
        }
        .navigationTitle("Manual")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.large)
        #endif
    }
}

// MARK: - Getting Started

struct ManualGettingStartedView: View {
    var body: some View {
        List {
            Section {
                ManualText(
                    "KataGo Anytime is a Go (Weiqi/Baduk) app powered by the KataGo engine, one of the strongest open-source Go AIs. You can play on any board size up to the model limit, review analysis in real time, and save your games."
                )
            } header: {
                Text("Overview")
            }

            Section {
                ManualStep(number: 1, text: "Open the app. A default game is created automatically.")
                ManualStep(number: 2, text: "Tap the \(Image(systemName: "ellipsis.circle")) menu (top right) to access game options.")
                ManualStep(number: 3, text: "Tap \"New Game\" to start a fresh game, or \"Configurations\" to set board size, komi, and rules before playing.")
                ManualStep(number: 4, text: "In Configurations → Game Settings → AI → White AI, set \"Time per move\" to a value greater than 0 (e.g. 5s) so the AI plays White automatically.")
                ManualStep(number: 5, text: "Tap any intersection on the board to place your stone. The AI analyzes the position immediately and, if Time per move > 0, plays its response.")
            } header: {
                Text("Quick Start")
            }

            Section {
                ManualText("The sidebar (swipe right or tap the grid icon) lists all your saved games. Tap a game to open it. The most recently modified game appears at the top.")
            } header: {
                Text("Game List")
            }

            Section {
                ManualRow(label: "Black stone", value: "You play first by default")
                ManualRow(label: "White stone", value: "Played by AI when \"Time per move\" is set > 0 in Configurations → Game Settings → AI")
                ManualRow(label: "Win rate bar", value: "Vertical bar to the left of the board; updates each move")
                ManualRow(label: "Move number", value: "Shown on each stone on the board when enabled in Configurations")
            } header: {
                Text("Interface at a Glance")
            }
        }
        .navigationTitle("Getting Started")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Playing Against AI (9x9)

struct ManualPlayAgainstAIView: View {
    var body: some View {
        List {
            Section {
                ManualText(
                    "A 9×9 game is the ideal way to learn. Games finish quickly (typically 30–60 moves), the smaller board makes it easy to understand territory and captures, and the AI can respond in seconds once Time per move is configured."
                )
            } header: {
                Text("Why Start with 9×9")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"New Game\".")
                ManualStep(number: 2, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Configurations\".")
                ManualStep(number: 3, text: "Tap \"Game Settings\" → \"Rule\". Set \"Board width\" and \"Board height\" both to 9, set \"Komi\" to 6.5, and choose a scoring rule (Area for Chinese, Territory for Japanese).")
                ManualStep(number: 4, text: "Tap \"AI\", then under \"White AI\" set \"Time per move\" to a value greater than 0 (e.g. 5s). This enables the AI to play White automatically.")
                ManualStep(number: 5, text: "Tap \"Done\" (top right of the sheet) to confirm.")
                ManualStep(number: 6, text: "You are now ready to play on the 9×9 board.")
            } header: {
                Text("Setting Up a 9×9 Game")
            }

            Section {
                ManualStep(number: 1, text: "You play Black. Tap any open intersection to place a stone.")
                ManualStep(number: 2, text: "The AI immediately analyzes the new position, showing candidate moves as colored circles. If \"Time per move\" > 0 for White AI, it also plays a move automatically after the allotted time.")
                ManualStep(number: 3, text: "If White AI \"Time per move\" is 0 (the default), the AI only provides analysis — you must place White stones yourself or increase the time.")
                ManualStep(number: 4, text: "Continue alternating until you want to pass or the game ends.")
                ManualStep(number: 5, text: "To pass your turn, tap the pass intersection below the board grid (visible when \"Show pass\" is enabled in Configurations → Game Settings → View).")
                ManualStep(number: 6, text: "When both players pass consecutively, the game ends. AI auto-play and analysis stop. The score shown is the AI's last estimated score from analysis — the app does not perform formal dead-stone marking or territory counting.")
            } header: {
                Text("Playing the Game")
            }

            Section {
                ManualText("Colored circles on the board show the AI's top candidate moves. The circle color shifts from green (most-visited) toward red (least-visited). The best move is highlighted with a blue ring.")
                ManualText("The win rate bar is a vertical bar to the left of the board. White's share fills from the top; Black's share fills from the bottom. The larger Black's portion, the better Black is doing.")
                ManualText("The score estimate is shown as a number at the center of the win rate bar.")
            } header: {
                Text("Reading AI Analysis")
            }

            Section {
                ManualRow(label: "Group with 2 eyes", value: "Cannot be captured — learn this first")
                ManualRow(label: "Corner territory", value: "Easiest to secure on a 9×9 board")
                ManualRow(label: "Atari", value: "A stone or group with only 1 liberty")
                ManualRow(label: "Capture", value: "Remove all liberties of an opponent's group")
                ManualRow(label: "Ko", value: "A repeated board position — illegal to repeat immediately")
            } header: {
                Text("Key Concepts for Beginners")
            }

            Section {
                ManualText("Tap \(Image(systemName: "backward.frame")) (back 1 step) in the toolbar to take back your last move. You can step back multiple times to explore different lines.")
                ManualText("Tap \(Image(systemName: "forward.frame")) (forward 1 step) to replay a move you stepped back through.")
                ManualText("After stepping back, tap a different intersection to start a new branch. Your original line is preserved and can be revisited by stepping forward.")
            } header: {
                Text("Stepping Back & Branching")
            }

            Section {
                ManualRow(label: "Handicap stones", value: "Set in Configurations → Game Settings → Rule → White handicap bonus")
                ManualRow(label: "AI strength", value: "Set \"Time per move\" in Configurations → Game Settings → AI — shorter time = weaker AI")
                ManualRow(label: "Human SL", value: "Set a Human profile in Configurations → Game Settings → AI for a more natural, human-like playing style")
            } header: {
                Text("Adjusting Difficulty")
            }
        }
        .navigationTitle("Playing Against AI (9×9)")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Board & Controls

struct ManualBoardControlsView: View {
    var body: some View {
        List {
            Section {
                ManualRow(label: "Tap intersection", value: "Place a stone")
                ManualRow(label: "Tap pass intersection (below the board grid)", value: "Pass your turn (requires \"Show pass\" enabled in Configurations → Game Settings → View)")
                ManualRow(label: "Tap existing stone (edit mode)", value: "Remove that stone")
                ManualRow(label: "Pinch / spread", value: "Zoom the board in or out")
                ManualRow(label: "Drag", value: "Pan the board when zoomed in")
            } header: {
                Text("Touch Gestures")
            }

            Section {
                ManualRow(label: "\(Image(systemName: "backward.end")) Go to start", value: "Jump to the beginning of the game")
                ManualRow(label: "\(Image(systemName: "backward")) Back 10", value: "Step back 10 moves")
                ManualRow(label: "\(Image(systemName: "backward.frame")) Back 1", value: "Step back 1 move (undo)")
                ManualRow(label: "Sparkle icon — Toggle analysis", value: "Cycle analysis: running → paused → off")
                ManualRow(label: "\(Image(systemName: "eye")) / \(Image(systemName: "book")) / \(Image(systemName: "eye.slash")) — Visibility", value: "Cycle between showing AI analysis, book moves, or hiding all overlays")
                ManualRow(label: "\(Image(systemName: "forward.frame")) Forward 1", value: "Step forward 1 move (redo)")
                ManualRow(label: "\(Image(systemName: "forward")) Forward 10", value: "Step forward 10 moves")
                ManualRow(label: "\(Image(systemName: "forward.end")) Go to end", value: "Jump to the latest move")
                ManualRow(label: "\(Image(systemName: "lock")) / \(Image(systemName: "lock.open")) — Edit mode", value: "Tap to enter board-editing mode; tap again to return to play mode")
            } header: {
                Text("Toolbar Buttons")
            }

            Section {
                ManualText("Tap the \(Image(systemName: "lock")) button in the toolbar to enter board-editing mode. In this mode you can place or remove Black and White stones freely to set up a specific position. Tap \(Image(systemName: "lock.open")) again to return to play mode.")
            } header: {
                Text("Board Editing Mode")
            }

            Section {
                ManualText("The score lead chart is shown in the info panel above the board. It plots Black's score lead over every move of the game. Drag across the chart to select a move and jump to that board position.")
            } header: {
                Text("Score Lead Chart")
            }

            Section {
                ManualText("After both players pass, AI auto-play and analysis stop. The app does not enter a dedicated scoring mode and does not count territory or mark dead stones. The score estimate visible in the bar is the AI's last calculated estimate from analysis.")
            } header: {
                Text("End of Game")
            }
        }
        .navigationTitle("Board & Controls")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Analysis Tools

struct ManualAnalysisView: View {
    var body: some View {
        List {
            Section {
                ManualText("KataGo continuously analyzes the current position in the background. Analysis results update the board overlay in real time as the engine calculates deeper.")
            } header: {
                Text("Continuous Analysis")
            }

            Section {
                ManualRow(label: "Colored circles", value: "Candidate moves; color shifts from green (most-visited) to red (least-visited)")
                ManualRow(label: "Blue ring", value: "Highlights the best move among all candidates")
                ManualRow(label: "Label on circle", value: "Shows win rate, score lead, or visit count depending on the Analysis information setting")
                ManualRow(label: "Ownership overlay", value: "Grayscale tint on intersections shows which player is estimated to control each point")
            } header: {
                Text("Board Overlay")
            }

            Section {
                ManualText("The win rate bar is a vertical bar to the left of the board. White's portion fills from the top; Black's portion fills from the bottom. A larger black section means Black is leading.")
                ManualText("The score estimate is displayed as a number at the center of the win rate bar.")
            } header: {
                Text("Win Rate & Score Bar")
            }

            Section {
                ManualText("The score lead chart is shown in the info panel above the board. It plots Black's score lead over every move of the game. Use it to identify the turning point — where the score shifted most dramatically.")
                ManualText("Drag across the chart to select a move. The board jumps to that position, letting you review critical moments without manually stepping through moves.")
            } header: {
                Text("Score Lead Chart")
            }

            Section {
                ManualText("KataGo's book lookup mode activates automatically for 9×9 games. Toggle it with the eye button (cycle to the book icon). In book mode, colored circles from the joseki database replace the engine's live analysis.")
            } header: {
                Text("Joseki Book Lookup")
            }

            Section {
                ManualRow(label: "Time per move", value: "Seconds the AI spends searching per move; set in Configurations → Game Settings → AI for each color")
                ManualRow(label: "Analysis threads", value: "Automatically set based on device")
                ManualRow(label: "Neural Engine", value: "Used on iPhone/iPad for power efficiency")
                ManualRow(label: "Metal GPU", value: "Used on Mac for faster analysis")
            } header: {
                Text("Analysis Engine Settings")
            }
        }
        .navigationTitle("Analysis Tools")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Configuration

struct ManualConfigurationView: View {
    var body: some View {
        List {
            Section {
                ManualText("Open Configurations from \(Image(systemName: "ellipsis.circle")) → \"Configurations\". The sheet has two top-level sections: \"Global Settings\" (shared across all games) and \"Game Settings\" (per-game).")
            } header: {
                Text("Opening Configurations")
            }

            Section {
                ManualRow(label: "Board width / Board height", value: "Set each dimension independently; any size from 2 up to the model limit")
                ManualRow(label: "Komi", value: "Compensation points given to White; enter any value")
                ManualRow(label: "Ko rule", value: "SIMPLE, POSITIONAL, or SITUATIONAL")
                ManualRow(label: "Scoring rule", value: "AREA (Chinese) or TERRITORY (Japanese)")
                ManualRow(label: "Tax rule", value: "NONE, SEKI, or ALL")
            } header: {
                Text("Game Settings → Rule")
            }

            Section {
                ManualRow(label: "Time per move (Black AI / White AI)", value: "Seconds the AI thinks before playing. Set to 0 to disable AI auto-play for that color. Higher values produce stronger moves.")
                ManualRow(label: "Human profile (Black AI / White AI)", value: "Makes the AI play in the style of a human at a chosen rank rather than at its full strength")
                ManualRow(label: "Playout doubling advantage", value: "Biases the search toward one side; positive values favor Black, negative favor White")
            } header: {
                Text("Game Settings → AI")
            }

            Section {
                ManualRow(label: "Apple Intelligence", value: "Enable or disable AI commentary generation")
                ManualRow(label: "Tone", value: "Style of commentary: Technical, Educational, Encouraging, Enthusiastic, or Poetic")
                ManualRow(label: "Temperature", value: "Controls how varied the commentary language is")
            } header: {
                Text("Game Settings → Comment")
            }

            Section {
                ManualRow(label: "Sound effect", value: "Stone placement and capture sounds")
                ManualRow(label: "Haptic feedback", value: "Vibration on stone placement")
            } header: {
                Text("Global Settings")
            }

            Section {
                ManualText("Global Settings (sound, haptics) are shared across all games and found in Configurations → Global Settings. Game Settings (rules, AI, view, etc.) apply only to the current game.")
            } header: {
                Text("Global vs. Per-Game Settings")
            }
        }
        .navigationTitle("Configuration")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Managing Games

struct ManualGameManagementView: View {
    var body: some View {
        List {
            Section {
                ManualText("All games are saved automatically in real time. You never need to tap a Save button. Games are also synced to iCloud if you are signed in to your Apple ID.")
            } header: {
                Text("Automatic Saving & iCloud Sync")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"New Game\" to create a blank game.")
                ManualStep(number: 2, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Clone\" to duplicate the current game (preserving moves and settings).")
            } header: {
                Text("Creating Games")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Share\" to open the system share sheet.")
                ManualStep(number: 2, text: "The game is exported as an SGF file, compatible with all standard Go software.")
                ManualStep(number: 3, text: "Share via AirDrop, Mail, Files, or any other available option.")
            } header: {
                Text("Sharing a Game (SGF Export)")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Import\".")
                ManualStep(number: 2, text: "Navigate to an SGF file in the Files app.")
                ManualStep(number: 3, text: "The game is imported and appears at the top of the game list.")
                ManualText("You can also open SGF files from Mail or Files directly — they will be imported automatically.")
            } header: {
                Text("Importing SGF Files")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Delete\".")
                ManualStep(number: 2, text: "Confirm the deletion in the dialog that appears.")
                ManualText("Deletion is permanent and cannot be undone.")
            } header: {
                Text("Deleting a Game")
            }

            Section {
                ManualText("Tap the game name shown in the top toolbar to open the name editor. Enter a new name and tap \"Save\".")
            } header: {
                Text("Renaming a Game")
            }

            Section {
                ManualText("Tap the \(Image(systemName: "photo")) thumbnail button in the \(Image(systemName: "ellipsis.circle")) menu to toggle between large and small game thumbnails in the sidebar.")
            } header: {
                Text("Thumbnail Size")
            }
        }
        .navigationTitle("Managing Games")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - AI Commentary

struct ManualCommentaryView: View {
    var body: some View {
        List {
            Section {
                ManualText("KataGo Anytime uses Apple's on-device language model to generate natural language commentary for each move. Commentary runs entirely on your device — no internet connection required.")
            } header: {
                Text("About AI Commentary")
            }

            Section {
                ManualText("Commentary appears in the info panel above the board. Switch to the comments tab by tapping the \(Image(systemName: "text.rectangle")) button at the bottom of the info panel. It describes the move's impact on win rate, score, and key stones.")
            } header: {
                Text("Where to Find Commentary")
            }

            Section {
                ManualRow(label: "Technical", value: "Precise analysis: win-rate numbers, score evaluation")
                ManualRow(label: "Educational", value: "Explains concepts suitable for learning players")
                ManualRow(label: "Encouraging", value: "Supportive and positive tone regardless of the result")
                ManualRow(label: "Enthusiastic", value: "Energetic and exciting descriptions of the game")
                ManualRow(label: "Poetic", value: "Artistic and metaphorical language")
            } header: {
                Text("Commentary Tones")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Configurations\".")
                ManualStep(number: 2, text: "Tap \"Game Settings\" → \"Comment\".")
                ManualStep(number: 3, text: "Choose your preferred tone from the \"Tone\" picker.")
                ManualStep(number: 4, text: "Tap \"Done\". Commentary will use the new tone from the next move onwards.")
            } header: {
                Text("Changing the Tone")
            }

            Section {
                ManualText("Commentary is generated using Apple FoundationModels, which requires a capable Apple Silicon device (iPhone 15 Pro / M-series Mac or newer). On unsupported devices, commentary is silently disabled.")
            } header: {
                Text("Device Requirements")
            }
        }
        .navigationTitle("AI Commentary")
        #if os(iOS) || os(visionOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
    }
}

// MARK: - Reusable Helper Views

private struct ManualText: View {
    let text: LocalizedStringKey

    init(_ text: String) {
        self.text = LocalizedStringKey(text)
    }

    var body: some View {
        Text(text)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct ManualRow: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.subheadline)
                .bold()
            Text(value)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

private struct ManualStep: View {
    let number: Int
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.subheadline.monospacedDigit())
                .bold()
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(Color.accentColor)
                .clipShape(Circle())
            Text(text)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        ManualView()
    }
}
