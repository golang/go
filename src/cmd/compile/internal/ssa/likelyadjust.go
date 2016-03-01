// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
)

type loop struct {
	header *Block // The header node of this (reducible) loop
	outer  *loop  // loop containing this loop
	// Next two fields not currently used, but cheap to maintain,
	// and aid in computation of inner-ness and list of blocks.
	nBlocks int32 // Number of blocks in this loop but not within inner loops
	isInner bool  // True if never discovered to contain a loop
}

// outerinner records that outer contains inner
func (sdom sparseTree) outerinner(outer, inner *loop) {
	oldouter := inner.outer
	if oldouter == nil || sdom.isAncestorEq(oldouter.header, outer.header) {
		inner.outer = outer
		outer.isInner = false
	}
}

type loopnest struct {
	f     *Func
	b2l   []*loop
	po    []*Block
	sdom  sparseTree
	loops []*loop
}

func min8(a, b int8) int8 {
	if a < b {
		return a
	}
	return b
}

func max8(a, b int8) int8 {
	if a > b {
		return a
	}
	return b
}

const (
	blDEFAULT = 0
	blMin     = blDEFAULT
	blCALL    = 1
	blRET     = 2
	blEXIT    = 3
)

var bllikelies [4]string = [4]string{"default", "call", "ret", "exit"}

func describePredictionAgrees(b *Block, prediction BranchPrediction) string {
	s := ""
	if prediction == b.Likely {
		s = " (agrees with previous)"
	} else if b.Likely != BranchUnknown {
		s = " (disagrees with previous, ignored)"
	}
	return s
}

func describeBranchPrediction(f *Func, b *Block, likely, not int8, prediction BranchPrediction) {
	f.Config.Warnl(int(b.Line), "Branch prediction rule %s < %s%s",
		bllikelies[likely-blMin], bllikelies[not-blMin], describePredictionAgrees(b, prediction))
}

func likelyadjust(f *Func) {
	// The values assigned to certain and local only matter
	// in their rank order.  0 is default, more positive
	// is less likely.  It's possible to assign a negative
	// unlikeliness (though not currently the case).
	certain := make([]int8, f.NumBlocks()) // In the long run, all outcomes are at least this bad. Mainly for Exit
	local := make([]int8, f.NumBlocks())   // for our immediate predecessors.

	nest := loopnestfor(f)
	po := nest.po
	b2l := nest.b2l

	for _, b := range po {
		switch b.Kind {
		case BlockExit:
			// Very unlikely.
			local[b.ID] = blEXIT
			certain[b.ID] = blEXIT

			// Ret, it depends.
		case BlockRet, BlockRetJmp:
			local[b.ID] = blRET
			certain[b.ID] = blRET

			// Calls. TODO not all calls are equal, names give useful clues.
			// Any name-based heuristics are only relative to other calls,
			// and less influential than inferences from loop structure.
		case BlockCall:
			local[b.ID] = blCALL
			certain[b.ID] = max8(blCALL, certain[b.Succs[0].ID])

		default:
			if len(b.Succs) == 1 {
				certain[b.ID] = certain[b.Succs[0].ID]
			} else if len(b.Succs) == 2 {
				// If successor is an unvisited backedge, it's in loop and we don't care.
				// Its default unlikely is also zero which is consistent with favoring loop edges.
				// Notice that this can act like a "reset" on unlikeliness at loops; the
				// default "everything returns" unlikeliness is erased by min with the
				// backedge likeliness; however a loop with calls on every path will be
				// tagged with call cost.  Net effect is that loop entry is favored.
				b0 := b.Succs[0].ID
				b1 := b.Succs[1].ID
				certain[b.ID] = min8(certain[b0], certain[b1])

				l := b2l[b.ID]
				l0 := b2l[b0]
				l1 := b2l[b1]

				prediction := b.Likely
				// Weak loop heuristic -- both source and at least one dest are in loops,
				// and there is a difference in the destinations.
				// TODO what is best arrangement for nested loops?
				if l != nil && l0 != l1 {
					noprediction := false
					switch {
					// prefer not to exit loops
					case l1 == nil:
						prediction = BranchLikely
					case l0 == nil:
						prediction = BranchUnlikely

						// prefer to stay in loop, not exit to outer.
					case l == l0:
						prediction = BranchLikely
					case l == l1:
						prediction = BranchUnlikely
					default:
						noprediction = true
					}
					if f.pass.debug > 0 && !noprediction {
						f.Config.Warnl(int(b.Line), "Branch prediction rule stay in loop%s",
							describePredictionAgrees(b, prediction))
					}

				} else {
					// Lacking loop structure, fall back on heuristics.
					if certain[b1] > certain[b0] {
						prediction = BranchLikely
						if f.pass.debug > 0 {
							describeBranchPrediction(f, b, certain[b0], certain[b1], prediction)
						}
					} else if certain[b0] > certain[b1] {
						prediction = BranchUnlikely
						if f.pass.debug > 0 {
							describeBranchPrediction(f, b, certain[b1], certain[b0], prediction)
						}
					} else if local[b1] > local[b0] {
						prediction = BranchLikely
						if f.pass.debug > 0 {
							describeBranchPrediction(f, b, local[b0], local[b1], prediction)
						}
					} else if local[b0] > local[b1] {
						prediction = BranchUnlikely
						if f.pass.debug > 0 {
							describeBranchPrediction(f, b, local[b1], local[b0], prediction)
						}
					}
				}
				if b.Likely != prediction {
					if b.Likely == BranchUnknown {
						b.Likely = prediction
					}
				}
			}
		}
		if f.pass.debug > 2 {
			f.Config.Warnl(int(b.Line), "BP: Block %s, local=%s, certain=%s", b, bllikelies[local[b.ID]-blMin], bllikelies[certain[b.ID]-blMin])
		}

	}
}

func (l *loop) String() string {
	return fmt.Sprintf("hdr:%s", l.header)
}

func (l *loop) LongString() string {
	i := ""
	o := ""
	if l.isInner {
		i = ", INNER"
	}
	if l.outer != nil {
		o = ", o=" + l.outer.header.String()
	}
	return fmt.Sprintf("hdr:%s%s%s", l.header, i, o)
}

// nearestOuterLoop returns the outer loop of loop most nearly
// containing block b; the header must dominate b.  loop itself
// is assumed to not be that loop.  For acceptable performance,
// we're relying on loop nests to not be terribly deep.
func (l *loop) nearestOuterLoop(sdom sparseTree, b *Block) *loop {
	var o *loop
	for o = l.outer; o != nil && !sdom.isAncestorEq(o.header, b); o = o.outer {
	}
	return o
}

func loopnestfor(f *Func) *loopnest {
	po := postorder(f)
	dom := dominators(f)
	sdom := newSparseTree(f, dom)
	b2l := make([]*loop, f.NumBlocks())
	loops := make([]*loop, 0)

	// Reducible-loop-nest-finding.
	for _, b := range po {
		if f.pass.debug > 3 {
			fmt.Printf("loop finding (0) at %s\n", b)
		}

		var innermost *loop // innermost header reachable from this block

		// IF any successor s of b is in a loop headed by h
		// AND h dominates b
		// THEN b is in the loop headed by h.
		//
		// Choose the first/innermost such h.
		//
		// IF s itself dominates b, the s is a loop header;
		// and there may be more than one such s.
		// Since there's at most 2 successors, the inner/outer ordering
		// between them can be established with simple comparisons.
		for _, bb := range b.Succs {
			l := b2l[bb.ID]

			if sdom.isAncestorEq(bb, b) { // Found a loop header
				if l == nil {
					l = &loop{header: bb, isInner: true}
					loops = append(loops, l)
					b2l[bb.ID] = l
				}
			} else { // Perhaps a loop header is inherited.
				// is there any loop containing our successor whose
				// header dominates b?
				if l != nil && !sdom.isAncestorEq(l.header, b) {
					l = l.nearestOuterLoop(sdom, b)
				}
			}

			if l == nil || innermost == l {
				continue
			}

			if innermost == nil {
				innermost = l
				continue
			}

			if sdom.isAncestor(innermost.header, l.header) {
				sdom.outerinner(innermost, l)
				innermost = l
			} else if sdom.isAncestor(l.header, innermost.header) {
				sdom.outerinner(l, innermost)
			}
		}

		if innermost != nil {
			b2l[b.ID] = innermost
			innermost.nBlocks++
		}
	}
	if f.pass.debug > 1 && len(loops) > 0 {
		fmt.Printf("Loops in %s:\n", f.Name)
		for _, l := range loops {
			fmt.Printf("%s, b=", l.LongString())
			for _, b := range f.Blocks {
				if b2l[b.ID] == l {
					fmt.Printf(" %s", b)
				}
			}
			fmt.Print("\n")
		}
		fmt.Printf("Nonloop blocks in %s:", f.Name)
		for _, b := range f.Blocks {
			if b2l[b.ID] == nil {
				fmt.Printf(" %s", b)
			}
		}
		fmt.Print("\n")
	}
	return &loopnest{f, b2l, po, sdom, loops}
}
