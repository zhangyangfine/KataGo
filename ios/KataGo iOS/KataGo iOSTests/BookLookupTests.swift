//
//  BookLookupTests.swift
//  KataGo iOSTests
//
//  Created by Chin-Chang Yang on 2026/3/3.
//

import Testing
@testable import KataGo_Anytime

@MainActor
struct BookLookupTests {

    // MARK: - Helper: build a small book

    /// Root (pos 0, black to play) has one move at canonical pos 0 (top-left)
    /// leading to child pos 1 (white to play, leaf with no children).
    static func twoNodeBook(linkSym: Int = 0) -> [BookLookup.BookPosition] {
        let rootMove = BookLookup.BookMove(
            positions: [0],
            winLoss: 0.6,
            sharpScore: 2.5,
            adjustedVisits: 100,
            policyPrior: 0.8
        )
        let root = BookLookup.BookPosition(
            nextPlayer: 1,
            moves: [rootMove],
            children: [0: (childId: 1, sym: linkSym)]
        )
        let childMove = BookLookup.BookMove(
            positions: [72],  // pos 72 = (8,0) bottom-left on 9x9
            winLoss: -0.3,
            sharpScore: -1.0,
            adjustedVisits: 50,
            policyPrior: 0.5
        )
        let child = BookLookup.BookPosition(
            nextPlayer: 2,
            moves: [childMove],
            children: [:]
        )
        return [root, child]
    }

    /// Root with two children at pos 0 and pos 1.
    static func branchingBook() -> [BookLookup.BookPosition] {
        let move0 = BookLookup.BookMove(
            positions: [0],
            winLoss: 0.6,
            sharpScore: 2.5,
            adjustedVisits: 100,
            policyPrior: 0.8
        )
        let move1 = BookLookup.BookMove(
            positions: [1],
            winLoss: 0.4,
            sharpScore: 1.0,
            adjustedVisits: 80,
            policyPrior: 0.6
        )
        let root = BookLookup.BookPosition(
            nextPlayer: 1,
            moves: [move0, move1],
            children: [
                0: (childId: 1, sym: 0),
                1: (childId: 2, sym: 0)
            ]
        )
        let child0 = BookLookup.BookPosition(nextPlayer: 2, moves: [], children: [:])
        let child1 = BookLookup.BookPosition(nextPlayer: 2, moves: [], children: [:])
        return [root, child0, child1]
    }

    // MARK: - Symmetry: compose

    @Test func composeIdentity() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for s in 0..<8 {
            #expect(book.compose(s, 0) == s)
            #expect(book.compose(0, s) == s)
        }
    }

    @Test func composeInverse() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for s in 0..<8 {
            // Composing with self should yield identity for pure flips (sym < 4)
            // but for all syms, composing s with inverse(s) == 0
            // inverse of sym s is: if s >= 4, swap bits 0,1 then XOR
            // Actually let's just verify compose(s, inv) == 0 by searching
            var found = false
            for inv in 0..<8 {
                if book.compose(s, inv) == 0 {
                    // Also verify the reverse
                    #expect(book.compose(inv, s) == 0)
                    found = true
                    break
                }
            }
            #expect(found, "Every symmetry should have an inverse")
        }
    }

    @Test func composeAssociativity() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for a in 0..<8 {
            for b in 0..<8 {
                for c in 0..<8 {
                    let lhs = book.compose(book.compose(a, b), c)
                    let rhs = book.compose(a, book.compose(b, c))
                    #expect(lhs == rhs, "compose must be associative: (\(a)*\(b))*\(c) != \(a)*(\(b)*\(c))")
                }
            }
        }
    }

    // MARK: - Symmetry: applySymmetry

    @Test func applySymmetryIdentity() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for pos in 0..<81 {
            #expect(book.applySymmetry(pos, sym: 0) == pos)
        }
    }

    @Test func applySymmetryPassInvariant() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let pass = 81
        for sym in 0..<8 {
            #expect(book.applySymmetry(pass, sym: sym) == pass)
        }
    }

    @Test func applySymmetryKnownValues() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // pos 0 = (0,0) top-left on 9x9
        // sym 1 = flip Y: (0,0) -> (0,8) = 8*9+0 = 72
        #expect(book.applySymmetry(0, sym: 1) == 72)
        // sym 2 = flip X: (0,0) -> (8,0) = 0*9+8 = 8
        #expect(book.applySymmetry(0, sym: 2) == 8)
        // sym 4 = transpose: (0,0) -> (0,0) = 0
        #expect(book.applySymmetry(0, sym: 4) == 0)
        // pos 1 = (1,0): sym 4 = transpose -> (0,1) = 1*9+0 = 9
        #expect(book.applySymmetry(1, sym: 4) == 9)
    }

    // MARK: - Symmetry: applyInverseSymmetry round-trip

    @Test func applyInverseSymmetryRoundTrip() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for sym in 0..<8 {
            for pos in 0..<81 {
                let transformed = book.applySymmetry(pos, sym: sym)
                let recovered = book.applyInverseSymmetry(transformed, sym: sym)
                #expect(recovered == pos, "Round-trip failed for pos=\(pos), sym=\(sym)")
            }
        }
    }

    @Test func applyInverseSymmetryPassInvariant() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let pass = 81
        for sym in 0..<8 {
            #expect(book.applyInverseSymmetry(pass, sym: sym) == pass)
        }
    }

    // MARK: - Coordinate mapping

    @Test func bookToAppPointCorners() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // book (0,0) top-left -> app (0, 8) bottom-left on 9x9
        let tl = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        #expect(tl == BoardPoint(x: 0, y: 8))

        // book (8,8) bottom-right -> app (8, 0) top-right
        let br = book.bookToAppPoint(bookX: 8, bookY: 8, boardHeight: 9)
        #expect(br == BoardPoint(x: 8, y: 0))
    }

    @Test func appPointToBookPosCorners() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // app (0,8) = book (0,0) = pos 0
        let pos0 = book.appPointToBookPos(BoardPoint(x: 0, y: 8), boardWidth: 9, boardHeight: 9)
        #expect(pos0 == 0)

        // app (8,0) = book (8,8) = 8*9+8 = 80
        let pos80 = book.appPointToBookPos(BoardPoint(x: 8, y: 0), boardWidth: 9, boardHeight: 9)
        #expect(pos80 == 80)
    }

    @Test func appPointToBookPosPass() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let pass = BoardPoint.pass(width: 9, height: 9)
        let pos = book.appPointToBookPos(pass, boardWidth: 9, boardHeight: 9)
        #expect(pos == 81)
    }

    @Test func coordinateRoundTripAll81() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        for bookPos in 0..<81 {
            let bookX = bookPos % 9
            let bookY = bookPos / 9
            let appPoint = book.bookToAppPoint(bookX: bookX, bookY: bookY, boardHeight: 9)
            let recovered = book.appPointToBookPos(appPoint, boardWidth: 9, boardHeight: 9)
            #expect(recovered == bookPos, "Round-trip failed for bookPos=\(bookPos)")
        }
    }

    // MARK: - Initialization

    @Test func testInitWithPositions() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        #expect(book.isLoaded == true)
        #expect(book.isInBook == true)
        #expect(book.currentPositionId == 0)
        #expect(book.accumulatedSymmetry == 0)
    }

    @Test func testInitEmpty() {
        let book = BookLookup(positions: [])
        #expect(book.isLoaded == false)
        #expect(book.isInBook == false)
    }

    // MARK: - Navigation: advanceMove

    @Test func advanceMoveBasic() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // Root child at canonical pos 0; with sym=0, app point for pos 0 is (0,8)
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        #expect(book.currentPositionId == 1)
        #expect(book.isInBook == true)
        #expect(book.justAdvanced == true)
    }

    @Test func advanceMoveSecondChild() {
        let book = BookLookup(positions: BookLookupTests.branchingBook())
        // Second child is at canonical pos 1 -> book (1,0) -> app (1,8)
        let appPoint = book.bookToAppPoint(bookX: 1, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        #expect(book.currentPositionId == 2)
        #expect(book.isInBook == true)
    }

    @Test func advanceMoveOutOfBook() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // pos (4,4) center has no child
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.advanceMove(appPoint: center, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        #expect(book.isInBook == false)
        #expect(book.justAdvanced == true)
    }

    @Test func advanceMoveAlreadyOutOfBook() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // First go out of book
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.advanceMove(appPoint: center, moveIndex: 1, boardWidth: 9, boardHeight: 9)
        #expect(book.isInBook == false)

        // Second advance should be a no-op (guard returns early)
        book.advanceMove(appPoint: center, moveIndex: 2, boardWidth: 9, boardHeight: 9)
        #expect(book.isInBook == false)
    }

    @Test func advanceMoveLeafNode() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // Advance to child (leaf)
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)
        #expect(book.currentPositionId == 1)

        // Any move from leaf should go out of book
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.advanceMove(appPoint: center, moveIndex: 2, boardWidth: 9, boardHeight: 9)
        #expect(book.isInBook == false)
    }

    @Test func advanceMoveWrongBoardSize() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        // Use 19x19 board size - should be rejected
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 19, boardHeight: 19)

        // State should be unchanged
        #expect(book.currentPositionId == 0)
        #expect(book.isInBook == true)
    }

    // MARK: - Navigation: resetToRoot

    @Test func resetToRootAfterAdvance() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)
        #expect(book.currentPositionId == 1)

        book.resetToRoot()
        #expect(book.currentPositionId == 0)
        #expect(book.accumulatedSymmetry == 0)
        #expect(book.isInBook == true)
    }

    @Test func resetToRootWhenUnloaded() {
        let book = BookLookup(positions: [])
        book.resetToRoot()
        // Should be a no-op
        #expect(book.isLoaded == false)
        #expect(book.isInBook == false)
    }

    // MARK: - Navigation: syncFromMoves

    @Test func syncFromMovesReplay() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.syncFromMoves([appPoint], boardWidth: 9, boardHeight: 9)

        #expect(book.currentPositionId == 1)
        #expect(book.isInBook == true)
        #expect(book.justAdvanced == false)  // syncFromMoves clears justAdvanced
    }

    @Test func syncFromMovesWithOutOfBookMove() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.syncFromMoves([center], boardWidth: 9, boardHeight: 9)

        #expect(book.isInBook == false)
    }

    @Test func syncFromMovesEmpty() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // Advance first, then sync with empty list should reset to root
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        book.syncFromMoves([], boardWidth: 9, boardHeight: 9)
        #expect(book.currentPositionId == 0)
        #expect(book.isInBook == true)
    }

    @Test func syncFromMovesWrongBoardSize() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.syncFromMoves([appPoint], boardWidth: 19, boardHeight: 19)

        #expect(book.isInBook == false)
    }

    // MARK: - Navigation: clearJustAdvanced

    @Test func clearJustAdvancedAfterAdvance() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)
        #expect(book.justAdvanced == true)

        book.clearJustAdvanced()
        #expect(book.justAdvanced == false)
    }

    // MARK: - Display: getBookAnalysis

    @Test func getBookAnalysisRootValues() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let analysis = book.getBookAnalysis(boardWidth: 9, boardHeight: 9)

        // Root has one move at canonical pos 0 -> display (0,8)
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        let info = analysis[appPoint]
        #expect(info != nil)
        #expect(info?.winLoss == 0.6)
        #expect(info?.sharpScore == 2.5)
        #expect(info?.adjustedVisits == 100)
        #expect(info?.policyPrior == 0.8)
        #expect(info?.rank == 0)
    }

    @Test func getBookAnalysisOutOfBookReturnsEmpty() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        // Go out of book
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.advanceMove(appPoint: center, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        let analysis = book.getBookAnalysis(boardWidth: 9, boardHeight: 9)
        #expect(analysis.isEmpty)
    }

    @Test func getBookAnalysisWrongBoardSizeReturnsEmpty() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let analysis = book.getBookAnalysis(boardWidth: 19, boardHeight: 19)
        #expect(analysis.isEmpty)
    }

    @Test func getBookAnalysisLeafWithNoMoves() {
        let positions = [
            BookLookup.BookPosition(nextPlayer: 1, moves: [], children: [:])
        ]
        let book = BookLookup(positions: positions)
        let analysis = book.getBookAnalysis(boardWidth: 9, boardHeight: 9)
        #expect(analysis.isEmpty)
    }

    @Test func getBookAnalysisRanks() {
        let book = BookLookup(positions: BookLookupTests.branchingBook())
        let analysis = book.getBookAnalysis(boardWidth: 9, boardHeight: 9)

        let point0 = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        let point1 = book.bookToAppPoint(bookX: 1, bookY: 0, boardHeight: 9)
        #expect(analysis[point0]?.rank == 0)
        #expect(analysis[point1]?.rank == 1)
    }

    // MARK: - Symmetry navigation

    @Test func advanceMoveWithSymmetryLink() {
        // Link sym=1 means flip-Y is applied at the child
        let book = BookLookup(positions: BookLookupTests.twoNodeBook(linkSym: 1))
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        #expect(book.currentPositionId == 1)
        #expect(book.accumulatedSymmetry == 1)
    }

    @Test func displayCoordinatesTransformWithSymmetry() {
        // After advancing with sym=1 (flip-Y), the child's canonical pos 72 = (0,8)
        // should display at applySymmetry(72, sym=1).
        // pos 72: y=8, x=0. Flip Y: y=0, x=0 -> pos 0. Display (0,8) in app coords.
        let book = BookLookup(positions: BookLookupTests.twoNodeBook(linkSym: 1))
        let appPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        book.advanceMove(appPoint: appPoint, moveIndex: 1, boardWidth: 9, boardHeight: 9)

        let analysis = book.getBookAnalysis(boardWidth: 9, boardHeight: 9)
        // Canonical 72 with sym=1: y=8,x=0 -> flip Y -> y=0,x=0 -> display pos 0 -> app (0,8)
        let expectedPoint = book.bookToAppPoint(bookX: 0, bookY: 0, boardHeight: 9)
        #expect(analysis[expectedPoint] != nil)
        #expect(analysis[expectedPoint]?.winLoss == -0.3)
    }

    // MARK: - currentPosition

    @Test func currentPositionAtRoot() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let pos = book.currentPosition
        #expect(pos != nil)
        #expect(pos?.nextPlayer == 1)
        #expect(pos?.moves.count == 1)
    }

    @Test func currentPositionOutOfBook() {
        let book = BookLookup(positions: BookLookupTests.twoNodeBook())
        let center = book.bookToAppPoint(bookX: 4, bookY: 4, boardHeight: 9)
        book.advanceMove(appPoint: center, moveIndex: 1, boardWidth: 9, boardHeight: 9)
        #expect(book.currentPosition == nil)
    }

    @Test func currentPositionEmptyBook() {
        let book = BookLookup(positions: [])
        #expect(book.currentPosition == nil)
    }
}
