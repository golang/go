// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"fmt"
)

// fuseEarly runs fuse(f, fuseTypePlain|fuseTypeIntInRange|fuseTypeNanCheck).
func fuseEarly(f *Func) {
	fuse(f, fuseTypePlain|fuseTypeIntInRange|fuseTypeSingleBitDifference|fuseTypeNanCheck)
}

// fuseLate runs fuse(f, fuseTypePlain|fuseTypeIf|fuseTypeBranchRedirect).
func fuseLate(f *Func) { fuse(f, fuseTypePlain|fuseTypeIf|fuseTypeBranchRedirect) }

type fuseType uint8

const (
	fuseTypePlain fuseType = 1 << iota
	fuseTypeIf
	fuseTypeIntInRange
	fuseTypeSingleBitDifference
	fuseTypeNanCheck
	fuseTypeBranchRedirect
	fuseTypeShortCircuit
)

// fuse simplifies control flow by joining basic blocks.
func fuse(f *Func, typ fuseType) {
	for changed := true; changed; {
		changed = false
		// Be sure to avoid quadratic behavior in fuseBlockPlain. See issue 13554.
		// Previously this was dealt with using backwards iteration, now fuseBlockPlain
		// handles large runs of blocks.
		for i := len(f.Blocks) - 1; i >= 0; i-- {
			b := f.Blocks[i]
			if typ&fuseTypeIf != 0 {
				changed = fuseBlockIf(b) || changed
			}
			if typ&fuseTypeIntInRange != 0 {
				changed = fuseIntInRange(b) || changed
			}
			if typ&fuseTypeSingleBitDifference != 0 {
				changed = fuseSingleBitDifference(b) || changed
			}
			if typ&fuseTypeNanCheck != 0 {
				changed = fuseNanCheck(b) || changed
			}
			if typ&fuseTypePlain != 0 {
				changed = fuseBlockPlain(b) || changed
			}
			if typ&fuseTypeShortCircuit != 0 {
				changed = shortcircuitBlock(b) || changed
			}
		}

		if typ&fuseTypeBranchRedirect != 0 {
			changed = fuseBranchRedirect(f) || changed
		}
		if changed {
			f.invalidateCFG()
		}
	}
}

// fuseBlockIf handles the following cases where s0 and s1 are empty blocks.
//
//	   b        b           b       b
//	\ / \ /    | \  /    \ / |     | |
//	 s0  s1    |  s1      s0 |     | |
//	  \ /      | /         \ |     | |
//	   ss      ss           ss      ss
//
// If all Phi ops in ss have identical variables for slots corresponding to
// s0, s1 and b then the branch can be dropped.
// This optimization often comes up in switch statements with multiple
// expressions in a case clause:
//
//	switch n {
//	  case 1,2,3: return 4
//	}
//
// TODO: If ss doesn't contain any OpPhis, are s0 and s1 dead code anyway.
func fuseBlockIf(b *Block) bool {
	if b.Kind != BlockIf {
		return false
	}
	// It doesn't matter how much Preds does s0 or s1 have.
	var ss0, ss1 *Block
	s0 := b.Succs[0].b
	i0 := b.Succs[0].i
	if s0.Kind != BlockPlain || !isEmpty(s0) {
		s0, ss0 = b, s0
	} else {
		ss0 = s0.Succs[0].b
		i0 = s0.Succs[0].i
	}
	s1 := b.Succs[1].b
	i1 := b.Succs[1].i
	if s1.Kind != BlockPlain || !isEmpty(s1) {
		s1, ss1 = b, s1
	} else {
		ss1 = s1.Succs[0].b
		i1 = s1.Succs[0].i
	}
	if ss0 != ss1 {
		if s0.Kind == BlockPlain && isEmpty(s0) && s1.Kind == BlockPlain && isEmpty(s1) {
			// Two special cases where both s0, s1 and ss are empty blocks.
			if s0 == ss1 {
				s0, ss0 = b, ss1
			} else if ss0 == s1 {
				s1, ss1 = b, ss0
			} else {
				return false
			}
		} else {
			return false
		}
	}
	ss := ss0

	// s0 and s1 are equal with b if the corresponding block is missing
	// (2nd, 3rd and 4th case in the figure).

	for _, v := range ss.Values {
		if v.Op == OpPhi && v.Uses > 0 && v.Args[i0] != v.Args[i1] {
			return false
		}
	}

	// We do not need to redirect the Preds of s0 and s1 to ss,
	// the following optimization will do this.
	b.removeEdge(0)
	if s0 != b && len(s0.Preds) == 0 {
		s0.removeEdge(0)
		// Move any (dead) values in s0 to b,
		// where they will be eliminated by the next deadcode pass.
		for _, v := range s0.Values {
			v.Block = b
		}
		b.Values = append(b.Values, s0.Values...)
		// Clear s0.
		s0.Kind = BlockInvalid
		s0.Values = nil
		s0.Succs = nil
		s0.Preds = nil
	}

	b.Kind = BlockPlain
	b.Likely = BranchUnknown
	b.ResetControls()
	// The values in b may be dead codes, and clearing them in time may
	// obtain new optimization opportunities.
	// First put dead values that can be deleted into a slice walkValues.
	// Then put their arguments in walkValues before resetting the dead values
	// in walkValues, because the arguments may also become dead values.
	walkValues := []*Value{}
	for _, v := range b.Values {
		if v.Uses == 0 && v.removeable() {
			walkValues = append(walkValues, v)
		}
	}
	for len(walkValues) != 0 {
		v := walkValues[len(walkValues)-1]
		walkValues = walkValues[:len(walkValues)-1]
		if v.Uses == 0 && v.removeable() {
			walkValues = append(walkValues, v.Args...)
			v.reset(OpInvalid)
		}
	}
	return true
}

// isEmpty reports whether b contains any live values.
// There may be false positives.
func isEmpty(b *Block) bool {
	for _, v := range b.Values {
		if v.Uses > 0 || v.Op.IsCall() || v.Op.HasSideEffects() || v.Type.IsVoid() || opcodeTable[v.Op].nilCheck {
			return false
		}
	}
	return true
}

// fuseBlockPlain handles a run of blocks with length >= 2,
// whose interior has single predecessors and successors,
// b must be BlockPlain, allowing it to be any node except the
// last (multiple successors means not BlockPlain).
// Cycles are handled and merged into b's successor.
func fuseBlockPlain(b *Block) bool {
	if b.Kind != BlockPlain {
		return false
	}

	c := b.Succs[0].b
	if len(c.Preds) != 1 || c == b { // At least 2 distinct blocks.
		return false
	}

	// find earliest block in run.  Avoid simple cycles.
	for len(b.Preds) == 1 && b.Preds[0].b != c && b.Preds[0].b.Kind == BlockPlain {
		b = b.Preds[0].b
	}

	// find latest block in run.  Still beware of simple cycles.
	for {
		if c.Kind != BlockPlain {
			break
		} // Has exactly 1 successor
		cNext := c.Succs[0].b
		if cNext == b {
			break
		} // not a cycle
		if len(cNext.Preds) != 1 {
			break
		} // no other incoming edge
		c = cNext
	}

	// Try to preserve any statement marks on the ends of blocks; move values to C
	var b_next *Block
	for bx := b; bx != c; bx = b_next {
		// For each bx with an end-of-block statement marker,
		// try to move it to a value in the next block,
		// or to the next block's end, if possible.
		b_next = bx.Succs[0].b
		if bx.Pos.IsStmt() == src.PosIsStmt {
			l := bx.Pos.Line() // looking for another place to mark for line l
			outOfOrder := false
			for _, v := range b_next.Values {
				if v.Pos.IsStmt() == src.PosNotStmt {
					continue
				}
				if l == v.Pos.Line() { // Found a Value with same line, therefore done.
					v.Pos = v.Pos.WithIsStmt()
					l = 0
					break
				}
				if l < v.Pos.Line() {
					// The order of values in a block is not specified so OOO in a block is not interesting,
					// but they do all come before the end of the block, so this disqualifies attaching to end of b_next.
					outOfOrder = true
				}
			}
			if l != 0 && !outOfOrder && (b_next.Pos.Line() == l || b_next.Pos.IsStmt() != src.PosIsStmt) {
				b_next.Pos = bx.Pos.WithIsStmt()
			}
		}
		// move all of bx's values to c (note containing loop excludes c)
		for _, v := range bx.Values {
			v.Block = c
		}
	}

	// Compute the total number of values and find the largest value slice in the run, to maximize chance of storage reuse.
	total := 0
	totalBeforeMax := 0 // number of elements preceding the maximum block (i.e. its position in the result).
	max_b := b          // block with maximum capacity

	for bx := b; ; bx = bx.Succs[0].b {
		if cap(bx.Values) > cap(max_b.Values) {
			totalBeforeMax = total
			max_b = bx
		}
		total += len(bx.Values)
		if bx == c {
			break
		}
	}

	// Use c's storage if fused blocks will fit, else use the max if that will fit, else allocate new storage.

	// Take care to avoid c.Values pointing to b.valstorage.
	// See golang.org/issue/18602.

	// It's important to keep the elements in the same order; maintenance of
	// debugging information depends on the order of *Values in Blocks.
	// This can also cause changes in the order (which may affect other
	// optimizations and possibly compiler output) for 32-vs-64 bit compilation
	// platforms (word size affects allocation bucket size affects slice capacity).

	// figure out what slice will hold the values,
	// preposition the destination elements if not allocating new storage
	var t []*Value
	if total <= len(c.valstorage) {
		t = c.valstorage[:total]
		max_b = c
		totalBeforeMax = total - len(c.Values)
		copy(t[totalBeforeMax:], c.Values)
	} else if total <= cap(max_b.Values) { // in place, somewhere
		t = max_b.Values[0:total]
		copy(t[totalBeforeMax:], max_b.Values)
	} else {
		t = make([]*Value, total)
		max_b = nil
	}

	// copy the values
	copyTo := 0
	for bx := b; ; bx = bx.Succs[0].b {
		if bx != max_b {
			copy(t[copyTo:], bx.Values)
		} else if copyTo != totalBeforeMax { // trust but verify.
			panic(fmt.Errorf("totalBeforeMax (%d) != copyTo (%d), max_b=%v, b=%v, c=%v", totalBeforeMax, copyTo, max_b, b, c))
		}
		if bx == c {
			break
		}
		copyTo += len(bx.Values)
	}
	c.Values = t

	// replace b->c edge with preds(b) -> c
	c.predstorage[0] = Edge{}
	if len(b.Preds) > len(b.predstorage) {
		c.Preds = b.Preds
	} else {
		c.Preds = append(c.predstorage[:0], b.Preds...)
	}
	for i, e := range c.Preds {
		p := e.b
		p.Succs[e.i] = Edge{c, i}
	}
	f := b.Func
	if f.Entry == b {
		f.Entry = c
	}

	// trash b's fields, just in case
	for bx := b; bx != c; bx = b_next {
		b_next = bx.Succs[0].b

		bx.Kind = BlockInvalid
		bx.Values = nil
		bx.Preds = nil
		bx.Succs = nil
	}
	return true
}
