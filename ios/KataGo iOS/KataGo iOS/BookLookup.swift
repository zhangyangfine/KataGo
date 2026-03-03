//
//  BookLookup.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2026/3/2.
//

import Foundation
import Compression
import SwiftUI

struct BookMoveInfo {
    let winLoss: Double
    let sharpScore: Double
    let adjustedVisits: Int64
    let policyPrior: Double
    let rank: Int
}

@MainActor
@Observable
class BookLookup {
    private(set) var isLoaded = false
    private(set) var isInBook = false
    private(set) var currentPositionId: Int = 0
    private(set) var accumulatedSymmetry: Int = 0
    private(set) var justAdvanced = false

    private var positions: [BookPosition] = []
    private let boardSize = 9

    struct BookPosition {
        let nextPlayer: Int  // 1 = black, 2 = white
        let moves: [BookMove]
        // canonicalPos -> (childId, linkSym)
        let children: [Int: (childId: Int, sym: Int)]
    }

    struct BookMove {
        let positions: [Int]  // canonical positions (y*boardSize+x), boardSize*boardSize for pass
        let winLoss: Double
        let sharpScore: Double
        let adjustedVisits: Int64
        let policyPrior: Double
    }

    init() {
        Task {
            await loadBook()
        }
    }

    /// Test-only initializer that injects hand-crafted positions without file I/O.
    init(positions: [BookPosition]) {
        self.positions = positions
        self.isLoaded = !positions.isEmpty
        self.isInBook = !positions.isEmpty
        self.currentPositionId = 0
        self.accumulatedSymmetry = 0
    }

    // MARK: - Loading

    private func loadBook() async {
        guard let url = Bundle.main.url(forResource: "book9x9jp.json", withExtension: "gz") else {
            return
        }

        // Parse off the main thread to avoid blocking UI
        let parseTask = Task.detached(priority: .userInitiated) {
            Self.parseBookFile(at: url)
        }
        guard let parsed = await parseTask.value else {
            return
        }

        self.positions = parsed
        self.isLoaded = true
        self.isInBook = !parsed.isEmpty
        self.currentPositionId = 0
        self.accumulatedSymmetry = 0
    }

    /// Parse the book file on the current thread. Returns nil on failure.
    private nonisolated static func parseBookFile(at url: URL) -> [BookPosition]? {
        guard let compressedData = try? Data(contentsOf: url),
              let decompressedData = decompressGzip(compressedData),
              let json = try? JSONSerialization.jsonObject(with: decompressedData) as? [String: Any],
              let positionsArray = json["p"] as? [Any] else {
            return nil
        }

        var parsed: [BookPosition] = []
        parsed.reserveCapacity(positionsArray.count)

        for posAny in positionsArray {
            guard let posArr = posAny as? [Any],
                  posArr.count >= 3,
                  let np = posArr[0] as? Int,
                  let movesArr = posArr[1] as? [Any],
                  let childrenArr = posArr[2] as? [Any] else {
                parsed.append(BookPosition(nextPlayer: 1, moves: [], children: [:]))
                continue
            }

            // Parse moves: [[pos_list, wl, ss, av, p], ...]
            var moves: [BookMove] = []
            for moveAny in movesArr {
                guard let moveArr = moveAny as? [Any],
                      moveArr.count >= 5,
                      let posList = moveArr[0] as? [Int] else {
                    continue
                }
                let wl = (moveArr[1] as? NSNumber)?.doubleValue ?? 0
                let ss = (moveArr[2] as? NSNumber)?.doubleValue ?? 0
                let av = Int64((moveArr[3] as? NSNumber)?.doubleValue ?? 0)
                let p = (moveArr[4] as? NSNumber)?.doubleValue ?? 0

                moves.append(BookMove(
                    positions: posList,
                    winLoss: wl,
                    sharpScore: ss,
                    adjustedVisits: av,
                    policyPrior: p
                ))
            }

            // Parse children: [[pos, childId, sym], ...]
            var children: [Int: (childId: Int, sym: Int)] = [:]
            for childAny in childrenArr {
                guard let childArr = childAny as? [Int],
                      childArr.count >= 3 else {
                    continue
                }
                children[childArr[0]] = (childId: childArr[1], sym: childArr[2])
            }

            parsed.append(BookPosition(nextPlayer: np, moves: moves, children: children))
        }

        return parsed
    }

    // MARK: - Gzip decompression

    private nonisolated static func decompressGzip(_ data: Data) -> Data? {
        guard data.count > 18,
              data[0] == 0x1f,
              data[1] == 0x8b,
              data[2] == 0x08 else {
            return nil
        }

        let flags = data[3]
        var offset = 10

        // Skip optional gzip header fields
        if flags & 0x04 != 0 {  // FEXTRA
            guard offset + 2 <= data.count else { return nil }
            let xlen = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            offset += 2 + xlen
        }
        if flags & 0x08 != 0 {  // FNAME
            while offset < data.count && data[offset] != 0 { offset += 1 }
            offset += 1
        }
        if flags & 0x10 != 0 {  // FCOMMENT
            while offset < data.count && data[offset] != 0 { offset += 1 }
            offset += 1
        }
        if flags & 0x02 != 0 {  // FHCRC
            offset += 2
        }

        guard offset < data.count - 8 else { return nil }

        // Read uncompressed size from ISIZE (last 4 bytes, little-endian, mod 2^32)
        let isize = Int(data[data.count - 4]) |
                    (Int(data[data.count - 3]) << 8) |
                    (Int(data[data.count - 2]) << 16) |
                    (Int(data[data.count - 1]) << 24)

        let deflateData = data[offset..<(data.count - 8)]
        // Use isize as hint, but allow larger (isize is mod 2^32)
        let destCapacity = max(isize, deflateData.count * 4)

        guard destCapacity > 0 else { return nil }

        var result = Data(count: destCapacity)
        let decodedSize = result.withUnsafeMutableBytes { destBuffer in
            deflateData.withUnsafeBytes { srcBuffer in
                compression_decode_buffer(
                    destBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    destCapacity,
                    srcBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    deflateData.count,
                    nil,
                    COMPRESSION_ZLIB
                )
            }
        }

        guard decodedSize > 0 else { return nil }
        result.count = decodedSize
        return result
    }

    // MARK: - Symmetry functions (ported from book.js)

    func compose(_ sym1: Int, _ sym2: Int) -> Int {
        var s2 = sym2
        if sym1 & 0x4 != 0 {
            s2 = (s2 & 0x4) | ((s2 & 0x2) >> 1) | ((s2 & 0x1) << 1)
        }
        return sym1 ^ s2
    }

    func applySymmetry(_ pos: Int, sym: Int) -> Int {
        guard pos < boardSize * boardSize else { return pos }
        var y = pos / boardSize
        var x = pos % boardSize

        if sym & 1 != 0 { y = boardSize - 1 - y }
        if sym & 2 != 0 { x = boardSize - 1 - x }
        if sym >= 4 { swap(&x, &y) }

        return x + y * boardSize
    }

    func applyInverseSymmetry(_ pos: Int, sym: Int) -> Int {
        guard pos < boardSize * boardSize else { return pos }
        var y = pos / boardSize
        var x = pos % boardSize

        if sym >= 4 { swap(&x, &y) }
        if sym & 1 != 0 { y = boardSize - 1 - y }
        if sym & 2 != 0 { x = boardSize - 1 - x }

        return x + y * boardSize
    }

    // MARK: - Coordinate mapping

    /// Convert book coordinates (y=0 top) to app BoardPoint (y=0 bottom).
    func bookToAppPoint(bookX: Int, bookY: Int, boardHeight: Int) -> BoardPoint {
        return BoardPoint(x: bookX, y: boardHeight - 1 - bookY)
    }

    /// Convert app BoardPoint to book position index.
    func appPointToBookPos(_ point: BoardPoint, boardWidth: Int, boardHeight: Int) -> Int {
        if point.isPass(width: boardWidth, height: boardHeight) {
            return boardSize * boardSize  // 81 for pass
        }
        let bookX = point.x
        let bookY = boardHeight - 1 - point.y
        return bookY * boardSize + bookX
    }

    // MARK: - Navigation

    func advanceMove(appPoint: BoardPoint, moveIndex: Int, boardWidth: Int, boardHeight: Int) {
        guard isInBook, boardWidth == boardSize, boardHeight == boardSize else { return }

        let displayPos = appPointToBookPos(appPoint, boardWidth: boardWidth, boardHeight: boardHeight)

        // Apply inverse symmetry to get canonical position
        let canonicalPos: Int
        if displayPos >= boardSize * boardSize {
            canonicalPos = displayPos  // pass is symmetry-invariant
        } else {
            canonicalPos = applyInverseSymmetry(displayPos, sym: accumulatedSymmetry)
        }

        guard currentPositionId < positions.count else {
            isInBook = false
            return
        }
        let position = positions[currentPositionId]

        guard let child = position.children[canonicalPos] else {
            isInBook = false
            justAdvanced = true
            return
        }

        let newSym = compose(child.sym, accumulatedSymmetry)
        currentPositionId = child.childId
        accumulatedSymmetry = newSym

        if child.childId >= positions.count {
            isInBook = false
        }

        justAdvanced = true
    }

    func clearJustAdvanced() {
        justAdvanced = false
    }

    func resetToRoot() {
        guard isLoaded else { return }
        currentPositionId = 0
        accumulatedSymmetry = 0
        isInBook = !positions.isEmpty
    }

    /// Replay book state from a list of app BoardPoints (called after undo/forward/game switch).
    func syncFromMoves(_ moves: [BoardPoint], boardWidth: Int, boardHeight: Int) {
        guard isLoaded, boardWidth == boardSize, boardHeight == boardSize else {
            isInBook = false
            return
        }

        resetToRoot()
        for (i, point) in moves.enumerated() {
            advanceMove(appPoint: point, moveIndex: i + 1, boardWidth: boardWidth, boardHeight: boardHeight)
            if !isInBook { break }
        }
        justAdvanced = false
    }

    // MARK: - Display

    var currentPosition: BookPosition? {
        guard isInBook, currentPositionId < positions.count else { return nil }
        return positions[currentPositionId]
    }

    func getBookAnalysis(boardWidth: Int, boardHeight: Int) -> [BoardPoint: BookMoveInfo] {
        guard isInBook,
              boardWidth == boardSize,
              boardHeight == boardSize,
              currentPositionId < positions.count else {
            return [:]
        }

        let position = positions[currentPositionId]
        var result: [BoardPoint: BookMoveInfo] = [:]

        for (rank, move) in position.moves.enumerated() {
            for canonicalPos in move.positions {
                let isPass = canonicalPos >= boardSize * boardSize

                let displayPos: Int
                if isPass {
                    displayPos = canonicalPos
                } else {
                    displayPos = applySymmetry(canonicalPos, sym: accumulatedSymmetry)
                }

                let appPoint: BoardPoint

                if isPass {
                    appPoint = BoardPoint.pass(width: boardWidth, height: boardHeight)
                } else {
                    let bookX = displayPos % boardSize
                    let bookY = displayPos / boardSize
                    appPoint = bookToAppPoint(bookX: bookX, bookY: bookY, boardHeight: boardHeight)
                }

                result[appPoint] = BookMoveInfo(
                    winLoss: move.winLoss,
                    sharpScore: move.sharpScore,
                    adjustedVisits: move.adjustedVisits,
                    policyPrior: move.policyPrior,
                    rank: rank
                )
            }
        }

        return result
    }
}
