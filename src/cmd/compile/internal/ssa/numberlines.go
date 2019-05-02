// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"sort"
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

// LosesStmtMark reports whether a prog with op as loses its statement mark on the way to DWARF.
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
		if b.Values[j].Pos.Line() == v.Pos.Line() && v.Pos.SameFile(b.Values[j].Pos) {
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

func (b *Block) FirstPossibleStmtValue() *Value {
	for _, v := range b.Values {
		if notStmtBoundary(v.Op) {
			continue
		}
		return v
	}
	return nil
}

func flc(p src.XPos) string {
	if p == src.NoXPos {
		return "none"
	}
	return fmt.Sprintf("(%d):%d:%d", p.FileIndex(), p.Line(), p.Col())
}

type fileAndPair struct {
	f  int32
	lp lineRange
}

type fileAndPairs []fileAndPair

func (fap fileAndPairs) Len() int {
	return len(fap)
}
func (fap fileAndPairs) Less(i, j int) bool {
	return fap[i].f < fap[j].f
}
func (fap fileAndPairs) Swap(i, j int) {
	fap[i], fap[j] = fap[j], fap[i]
}

// -d=ssa/number_lines/stats=1 (that bit) for line and file distribution statistics
// -d=ssa/number_lines/debug for information about why particular values are marked as statements.
func numberLines(f *Func) {
	po := f.Postorder()
	endlines := make(map[ID]src.XPos)
	ranges := make(map[int]lineRange)
	note := func(p src.XPos) {
		line := uint32(p.Line())
		i := int(p.FileIndex())
		lp, found := ranges[i]
		change := false
		if line < lp.first || !found {
			lp.first = line
			change = true
		}
		if line > lp.last {
			lp.last = line
			change = true
		}
		if change {
			ranges[i] = lp
		}
	}

	// Visit in reverse post order so that all non-loop predecessors come first.
	for j := len(po) - 1; j >= 0; j-- {
		b := po[j]
		// Find the first interesting position and check to see if it differs from any predecessor
		firstPos := src.NoXPos
		firstPosIndex := -1
		if b.Pos.IsStmt() != src.PosNotStmt {
			note(b.Pos)
		}
		for i := 0; i < len(b.Values); i++ {
			v := b.Values[i]
			if v.Pos.IsStmt() != src.PosNotStmt {
				note(v.Pos)
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
				if f.pass.debug > 0 {
					fmt.Printf("Mark stmt effectively-empty-block %s %s %s\n", f.Name, b, flc(b.Pos))
				}
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
			if f.pass.debug > 0 {
				fmt.Printf("Mark stmt entry-block %s %s %s %s\n", f.Name, b, b.Values[firstPosIndex], flc(firstPos))
			}
		} else { // differing pred
			for _, p := range b.Preds {
				pbi := p.Block().ID
				if endlines[pbi].Line() != firstPos.Line() || !endlines[pbi].SameFile(firstPos) {
					b.Values[firstPosIndex].Pos = firstPos.WithIsStmt()
					if f.pass.debug > 0 {
						fmt.Printf("Mark stmt differing-pred %s %s %s %s, different=%s ending %s\n",
							f.Name, b, b.Values[firstPosIndex], flc(firstPos), p.Block(), flc(endlines[pbi]))
					}
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
			note(v.Pos)
			// skip ahead if possible
			i = nextGoodStatementIndex(v, i, b)
			v = b.Values[i]
			if v.Pos.Line() != firstPos.Line() || !v.Pos.SameFile(firstPos) {
				if f.pass.debug > 0 {
					fmt.Printf("Mark stmt new line %s %s %s %s prev pos = %s\n", f.Name, b, v, flc(v.Pos), flc(firstPos))
				}
				firstPos = v.Pos
				v.Pos = v.Pos.WithIsStmt()
			} else {
				v.Pos = v.Pos.WithDefaultStmt()
			}
		}
		if b.Pos.IsStmt() != src.PosNotStmt && (b.Pos.Line() != firstPos.Line() || !b.Pos.SameFile(firstPos)) {
			if f.pass.debug > 0 {
				fmt.Printf("Mark stmt end of block differs %s %s %s prev pos = %s\n", f.Name, b, flc(b.Pos), flc(firstPos))
			}
			b.Pos = b.Pos.WithIsStmt()
			firstPos = b.Pos
		}
		endlines[b.ID] = firstPos
	}
	if f.pass.stats&1 != 0 {
		// Report summary statistics on the shape of the sparse map about to be constructed
		// TODO use this information to make sparse maps faster.
		var entries fileAndPairs
		for k, v := range ranges {
			entries = append(entries, fileAndPair{int32(k), v})
		}
		sort.Sort(entries)
		total := uint64(0)            // sum over files of maxline(file) - minline(file)
		maxfile := int32(0)           // max(file indices)
		minline := uint32(0xffffffff) // min over files of minline(file)
		maxline := uint32(0)          // max over files of maxline(file)
		for _, v := range entries {
			if f.pass.stats > 1 {
				f.LogStat("file", v.f, "low", v.lp.first, "high", v.lp.last)
			}
			total += uint64(v.lp.last - v.lp.first)
			if maxfile < v.f {
				maxfile = v.f
			}
			if minline > v.lp.first {
				minline = v.lp.first
			}
			if maxline < v.lp.last {
				maxline = v.lp.last
			}
		}
		f.LogStat("SUM_LINE_RANGE", total, "MAXMIN_LINE_RANGE", maxline-minline, "MAXFILE", maxfile, "NFILES", len(entries))
	}
	// cachedLineStarts is an empty sparse map for values that are included within ranges.
	f.cachedLineStarts = newXposmap(ranges)
}
