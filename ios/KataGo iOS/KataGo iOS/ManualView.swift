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
                    "KataGo Anytime is a Go (Weiqi/Baduk) app powered by the KataGo engine, one of the strongest open-source Go AIs. You can play on a 9×9, 13×13, or 19×19 board, review analysis in real time, and save your games."
                )
            } header: {
                Text("Overview")
            }

            Section {
                ManualStep(number: 1, text: "Open the app. A default game is created automatically.")
                ManualStep(number: 2, text: "Tap the \(Image(systemName: "ellipsis.circle")) menu (top right) to access game options.")
                ManualStep(number: 3, text: "Tap \"New Game\" to start a fresh game, or \"Configurations\" to set board size, komi, and rules before playing.")
                ManualStep(number: 4, text: "In Configurations → AI → White AI, set \"Time per move\" to a value greater than 0 (e.g. 5s) so the AI plays White automatically.")
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
                ManualRow(label: "White stone", value: "Played by AI when \"Time per move\" is set > 0 in Configurations → AI")
                ManualRow(label: "Win rate bar", value: "Shown below the board; updates each move")
                ManualRow(label: "Move number", value: "Displayed in the toolbar")
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
                ManualStep(number: 3, text: "Tap \"Game Settings\" → \"Rule\". Set \"Board Size\" to 9 × 9, \"Komi\" to 6.5, and choose \"Rules\" (Japanese or Chinese are most common).")
                ManualStep(number: 4, text: "Tap \"AI\", then under \"White AI\" set \"Time per move\" to a value greater than 0 (e.g. 5s). This enables the AI to play White automatically.")
                ManualStep(number: 5, text: "Tap \"Done\" (top right of the sheet) to confirm.")
                ManualStep(number: 6, text: "You are now ready to play on the 9×9 board.")
            } header: {
                Text("Setting Up a 9×9 Game")
            }

            Section {
                ManualStep(number: 1, text: "You play Black. Tap any open intersection to place a stone.")
                ManualStep(number: 2, text: "The AI immediately analyzes the new position, showing candidate moves as arrows. If \"Time per move\" > 0 for White AI, it also plays a move automatically after the allotted time.")
                ManualStep(number: 3, text: "If White AI \"Time per move\" is 0 (the default), the AI only provides analysis — you must place White stones yourself or increase the time.")
                ManualStep(number: 4, text: "Continue alternating until you want to pass or the game ends.")
                ManualStep(number: 5, text: "To pass your turn, tap the \(Image(systemName: "hand.raised")) Pass button in the toolbar.")
                ManualStep(number: 6, text: "When both players pass consecutively, the game ends. AI auto-play and analysis stop. The score shown is the AI's last estimated score from analysis — the app does not perform formal dead-stone marking or territory counting.")
            } header: {
                Text("Playing the Game")
            }

            Section {
                ManualText("Blue arrows on the board show the AI's top candidate moves. The percentage label indicates the win rate for that move. The best move has the brightest arrow.")
                ManualText("The win rate bar at the bottom of the board shows Black's estimated win probability. Above 50% is favorable for Black.")
                ManualText("The score estimate below the win rate bar shows the expected point margin.")
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
                ManualText("Tap \(Image(systemName: "arrow.uturn.backward")) (undo) in the toolbar to take back your last move. You can undo multiple times to explore different lines.")
                ManualText("Tap \(Image(systemName: "arrow.uturn.forward")) (redo) to replay a move you undid.")
                ManualText("After undoing, tap a different intersection to start a new branch. Your original line is preserved and can be revisited.")
            } header: {
                Text("Undoing & Branching")
            }

            Section {
                ManualRow(label: "Handicap stones", value: "Set in Configurations → Game Settings → Rule → Handicap")
                ManualRow(label: "AI strength", value: "Set \"Time per move\" in Configurations → AI — shorter time = weaker AI")
                ManualRow(label: "Human SL", value: "Set a Human profile in Configurations → AI for a more natural, human-like playing style")
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
                ManualRow(label: "Tap existing stone (edit mode)", value: "Remove that stone")
                ManualRow(label: "Pinch / spread", value: "Zoom the board in or out")
                ManualRow(label: "Drag", value: "Pan the board when zoomed in")
            } header: {
                Text("Touch Gestures")
            }

            Section {
                ManualRow(label: "\(Image(systemName: "arrow.uturn.backward")) Undo", value: "Take back the last move")
                ManualRow(label: "\(Image(systemName: "arrow.uturn.forward")) Redo", value: "Replay an undone move")
                ManualRow(label: "\(Image(systemName: "hand.raised")) Pass", value: "Pass your turn")
                ManualRow(label: "\(Image(systemName: "flag")) Resign", value: "Concede the current game")
                ManualRow(label: "\(Image(systemName: "play")) Auto-play", value: "Let the AI play both sides automatically")
                ManualRow(label: "\(Image(systemName: "pencil")) Edit", value: "Enter board-editing mode to set up positions")
            } header: {
                Text("Toolbar Buttons")
            }

            Section {
                ManualText("Tap \(Image(systemName: "pencil")) Edit in the toolbar to enter board-editing mode. In this mode you can place or remove Black and White stones freely to set up a specific position. Tap Edit again (or Done) to return to play mode.")
            } header: {
                Text("Board Editing Mode")
            }

            Section {
                ManualText("Tap the analysis chart icon or swipe up on the win rate bar to expand the score/win rate line chart. The chart plots win rate over every move of the game. Tap any point on the chart to jump to that move.")
            } header: {
                Text("Win Rate Chart")
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
                ManualRow(label: "Blue arrows", value: "Candidate moves; brightness indicates rank")
                ManualRow(label: "Percentage label", value: "Win rate if that move is played")
                ManualRow(label: "Ownership overlay", value: "Color tint shows which player controls each intersection")
                ManualRow(label: "Score delta", value: "Point gain or loss shown on candidate moves")
            } header: {
                Text("Board Overlay")
            }

            Section {
                ManualText("The win rate bar runs across the bottom of the board. Black's estimated win probability fills from the left (black side); White's fills from the right. The midpoint line represents 50%.")
                ManualText("The score estimate is displayed as a signed number (positive = Black leads, negative = White leads).")
            } header: {
                Text("Win Rate & Score Bar")
            }

            Section {
                ManualText("The line chart below the board plots Black's win rate over every move. Use it to identify the turning point of the game — where the win rate shifted most dramatically.")
                ManualText("Tap any point on the chart to jump to that board position. This lets you review critical moments without manually stepping through moves.")
            } header: {
                Text("Line Chart")
            }

            Section {
                ManualText("KataGo's book lookup (when enabled) highlights opening moves from a joseki database. A book icon appears next to moves that match known good patterns.")
            } header: {
                Text("Joseki Book Lookup")
            }

            Section {
                ManualRow(label: "Time per move", value: "Seconds the AI spends searching per move; set in Configurations → AI for each color")
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
                ManualText("Open Configurations from \(Image(systemName: "ellipsis.circle")) → \"Configurations\". Settings apply to the current game. Start a new game or use Clone to apply new settings to a fresh game.")
            } header: {
                Text("Opening Configurations")
            }

            Section {
                ManualRow(label: "Board Size", value: "9×9, 13×13, or 19×19 (and custom up to the model limit)")
                ManualRow(label: "Komi", value: "Compensation points given to White (commonly 6.5 or 7.5)")
                ManualRow(label: "Handicap", value: "Number of Black handicap stones placed at start")
                ManualRow(label: "Rules", value: "Japanese, Chinese, Tromp-Taylor, Korean, AGA, or NZ")
            } header: {
                Text("Game Settings")
            }

            Section {
                ManualRow(label: "Time per move (Black AI / White AI)", value: "Seconds the AI thinks before playing. Set to 0 to disable AI auto-play for that color. Higher values produce stronger moves.")
                ManualRow(label: "Human profile (Black AI / White AI)", value: "Makes the AI play in the style of a human at a chosen rank rather than at its full strength")
                ManualRow(label: "Playout doubling advantage", value: "Biases the AI toward aggressive or defensive play")
            } header: {
                Text("AI Strength")
            }

            Section {
                ManualRow(label: "Commentary Tone", value: "Style of AI commentary: technical, educational, encouraging, enthusiastic, or poetic")
                ManualRow(label: "Sound Effects", value: "Stone placement and capture sounds (global setting)")
                ManualRow(label: "Haptic Feedback", value: "Vibration on stone placement (global setting)")
            } header: {
                Text("App Preferences")
            }

            Section {
                ManualText("Global settings (sound, haptics) are shared across all games. They can be toggled in the \(Image(systemName: "ellipsis.circle")) menu or in the Configurations sheet.")
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
                ManualText("Long-press a game in the sidebar, or tap \(Image(systemName: "pencil")) next to the game title, to rename it.")
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
                ManualText("Commentary appears automatically in the comment panel below the board after each move. It describes the move's impact on win rate, score, and key stones.")
            } header: {
                Text("Where to Find Commentary")
            }

            Section {
                ManualRow(label: "Technical", value: "Precise analysis: win-rate numbers, joseki references")
                ManualRow(label: "Educational", value: "Explains concepts suitable for learning players")
                ManualRow(label: "Encouraging", value: "Supportive and positive tone regardless of the result")
                ManualRow(label: "Enthusiastic", value: "Energetic and exciting descriptions of the game")
                ManualRow(label: "Poetic", value: "Artistic and metaphorical language")
            } header: {
                Text("Commentary Tones")
            }

            Section {
                ManualStep(number: 1, text: "Tap \(Image(systemName: "ellipsis.circle")) → \"Configurations\".")
                ManualStep(number: 2, text: "Scroll to the Commentary section.")
                ManualStep(number: 3, text: "Choose your preferred tone.")
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
