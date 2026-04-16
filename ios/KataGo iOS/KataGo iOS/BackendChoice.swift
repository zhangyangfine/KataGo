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
