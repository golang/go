// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"cmd/internal/src"
	"math"
)

func isPoorStatementOp(op Op) bool {
	switch op {
	// Note that Nilcheck often vanishes, but when it doesn't, you'd love to start the statement there
	// so that a debugger-user sees the stop before the panic, and can examine the value.
	case OpAddr, OpLocalAddr, OpOffPtr, OpStructSelect, OpConstBool, OpConst8, OpConst16, OpConst32, OpConst64, OpConst32F, OpConst64F:
		return true
	}
	return false
}

// LosesStmtMark returns whether a prog with op as loses its statement mark on the way to DWARF.
// The attributes from some opcodes are lost in translation.
// TODO: this is an artifact of how funcpctab combines information for instructions at a single PC.
// Should try to fix it there.
func LosesStmtMark(as obj.As) bool {
	// is_stmt does not work for these; it DOES for ANOP even though that generates no code.
	return as == obj.APCDATA || as == obj.AFUNCDATA
}

// nextGoodStatementIndex returns an index at i or later that is believed
// to be a good place to start the statement for b.  This decision is
// based on v's Op, the possibility of a better later operation, and
// whether the values following i are the same line as v.
// If a better statement index isn't found, then i is returned.
func nextGoodStatementIndex(v *Value, i int, b *Block) int {
	// If the value is the last one in the block, too bad, it will have to do
	// (this assumes that the value ordering vaguely corresponds to the source
	// program execution order, which tends to be true directly after ssa is
	// first built.
	if i >= len(b.Values)-1 {
		return i
	}
	// Only consider the likely-ephemeral/fragile opcodes expected to vanish in a rewrite.
	if !isPoorStatementOp(v.Op) {
		return i
	}
	// Look ahead to see what the line number is on the next thing that could be a boundary.
	for j := i + 1; j < len(b.Values); j++ {
		if b.Values[j].Pos.IsStmt() == src.PosNotStmt { // ignore non-statements
			continue
		}
		if b.Values[j].Pos.Line() == v.Pos.Line() {
			return j
		}
		return i
	}
	return i
}

// notStmtBoundary indicates which value opcodes can never be a statement
// boundary because they don't correspond to a user's understanding of a
// statement boundary.  Called from *Value.reset(), and *Func.newValue(),
// located here to keep all the statement boundary heuristics in one place.
// Note: *Value.reset() filters out OpCopy because of how that is used in
// rewrite.
func notStmtBoundary(op Op) bool {
	switch op {
	case OpCopy, OpPhi, OpVarKill, OpVarDef, OpUnknown, OpFwdRef, OpArg:
		return true
	}
	return false
}

func numberLines(f *Func) {
	po := f.Postorder()
	endlines := make(map[ID]src.XPos)
	last := uint(0)              // uint follows type of XPos.Line()
	first := uint(math.MaxInt32) // unsigned, but large valid int when cast
	note := func(line uint) {
		if line < first {
			first = line
		}
		if line > last {
			last = line
		}
	}

	// Visit in reverse post order so that all non-loop predecessors come first.
	for j := len(po) - 1; j >= 0; j-- {
		b := po[j]
		// Find the first interesting position and check to see if it differs from any predecessor
		firstPos := src.NoXPos
		firstPosIndex := -1
		if b.Pos.IsStmt() != src.PosNotStmt {
			note(b.Pos.Line())
		}
		for i := 0; i < len(b.Values); i++ {
			v := b.Values[i]
			if v.Pos.IsStmt() != src.PosNotStmt {
				note(v.Pos.Line())
				// skip ahead to better instruction for this line if possible
				i = nextGoodStatementIndex(v, i, b)
				v = b.Values[i]
				firstPosIndex = i
				firstPos = v.Pos
				v.Pos = firstPos.WithDefaultStmt() // default to default
				break
			}
		}

		if firstPosIndex == -1 { // Effectively empty block, check block's own Pos, consider preds.
			if b.Pos.IsStmt() != src.PosNotStmt {
				b.Pos = b.Pos.WithIsStmt()
				endlines[b.ID] = b.Pos
				continue
			}
			line := src.NoXPos
			for _, p := range b.Preds {
				pbi := p.Block().ID
				if endlines[pbi] != line {
					if line == src.NoXPos {
						line = endlines[pbi]
						continue
					} else {
						line = src.NoXPos
						break
					}

				}
			}
			endlines[b.ID] = line
			continue
		}
		// check predecessors for any difference; if firstPos differs, then it is a boundary.
		if len(b.Preds) == 0 { // Don't forget the entry block
			b.Values[firstPosIndex].Pos = firstPos.WithIsStmt()
		} else {
			for _, p := range b.Preds {
				pbi := p.Block().ID
				if endlines[pbi] != firstPos {
					b.Values[firstPosIndex].Pos = firstPos.WithIsStmt()
					break
				}
			}
		}
		// iterate forward setting each new (interesting) position as a statement boundary.
		for i := firstPosIndex + 1; i < len(b.Values); i++ {
			v := b.Values[i]
			if v.Pos.IsStmt() == src.PosNotStmt {
				continue
			}
			note(v.Pos.Line())
			// skip ahead if possible
			i = nextGoodStatementIndex(v, i, b)
			v = b.Values[i]
			if v.Pos.Line() != firstPos.Line() || !v.Pos.SameFile(firstPos) {
				firstPos = v.Pos
				v.Pos = v.Pos.WithIsStmt()
			} else {
				v.Pos = v.Pos.WithDefaultStmt()
			}
		}
		if b.Pos.IsStmt() != src.PosNotStmt && (b.Pos.Line() != firstPos.Line() || !b.Pos.SameFile(firstPos)) {
			b.Pos = b.Pos.WithIsStmt()
			firstPos = b.Pos
		}
		endlines[b.ID] = firstPos
	}
	f.cachedLineStarts = newBiasedSparseMap(int(first), int(last))
}
