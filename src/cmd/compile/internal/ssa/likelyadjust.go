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

	// Next three fields used by regalloc and/or
	// aid in computation of inner-ness and list of blocks.
	nBlocks int32 // Number of blocks in this loop but not within inner loops
	depth   int16 // Nesting depth of the loop; 1 is outermost.
	isInner bool  // True if never discovered to contain a loop

	// True if all paths through the loop have a call.
	// Computed and used by regalloc; stored here for convenience.
	containsUnavoidableCall bool
}

// outerinner records that outer contains inner
func (sdom SparseTree) outerinner(outer, inner *loop) {
	// There could be other outer loops found in some random order,
	// locate the new outer loop appropriately among them.

	// Outer loop headers dominate inner loop headers.
	// Use this to put the "new" "outer" loop in the right place.
	oldouter := inner.outer
	for oldouter != nil && sdom.isAncestor(outer.header, oldouter.header) {
		inner = oldouter
		oldouter = inner.outer
	}
	if outer == oldouter {
		return
	}
	if oldouter != nil {
		sdom.outerinner(oldouter, outer)
	}

	inner.outer = outer
	outer.isInner = false
}

type loopnest struct {
	f              *Func
	b2l            []*loop
	po             []*Block
	sdom           SparseTree
	loops          []*loop
	hasIrreducible bool // TODO current treatment of irreducible loops is very flaky, if accurate loops are needed, must punt at function level.
}

const (
	blDEFAULT = 0
	blMin     = blDEFAULT
	blCALL    = 1
	blRET     = 2
	blEXIT    = 3
)

var bllikelies = [4]string{"default", "call", "ret", "exit"}

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
	f.Warnl(b.Pos, "Branch prediction rule %s < %s%s",
		bllikelies[likely-blMin], bllikelies[not-blMin], describePredictionAgrees(b, prediction))
}

func likelyadjust(f *Func) {
	// The values assigned to certain and local only matter
	// in their rank order.  0 is default, more positive
	// is less likely. It's possible to assign a negative
	// unlikeliness (though not currently the case).
	certain := f.Cache.allocInt8Slice(f.NumBlocks()) // In the long run, all outcomes are at least this bad. Mainly for Exit
	defer f.Cache.freeInt8Slice(certain)
	local := f.Cache.allocInt8Slice(f.NumBlocks()) // for our immediate predecessors.
	defer f.Cache.freeInt8Slice(local)

	po := f.postorder()
	nest := f.loopnest()
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
		case BlockDefer:
			local[b.ID] = blCALL
			certain[b.ID] = max(blCALL, certain[b.Succs[0].b.ID])

		default:
			if len(b.Succs) == 1 {
				certain[b.ID] = certain[b.Succs[0].b.ID]
			} else if len(b.Succs) == 2 {
				// If successor is an unvisited backedge, it's in loop and we don't care.
				// Its default unlikely is also zero which is consistent with favoring loop edges.
				// Notice that this can act like a "reset" on unlikeliness at loops; the
				// default "everything returns" unlikeliness is erased by min with the
				// backedge likeliness; however a loop with calls on every path will be
				// tagged with call cost. Net effect is that loop entry is favored.
				b0 := b.Succs[0].b.ID
				b1 := b.Succs[1].b.ID
				certain[b.ID] = min(certain[b0], certain[b1])

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
						f.Warnl(b.Pos, "Branch prediction rule stay in loop%s",
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
			// Look for calls in the block.  If there is one, make this block unlikely.
			for _, v := range b.Values {
				if opcodeTable[v.Op].call {
					local[b.ID] = blCALL
					certain[b.ID] = max(blCALL, certain[b.Succs[0].b.ID])
					break
				}
			}
		}
		if f.pass.debug > 2 {
			f.Warnl(b.Pos, "BP: Block %s, local=%s, certain=%s", b, bllikelies[local[b.ID]-blMin], bllikelies[certain[b.ID]-blMin])
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

func (l *loop) isWithinOrEq(ll *loop) bool {
	if ll == nil { // nil means whole program
		return true
	}
	for ; l != nil; l = l.outer {
		if l == ll {
			return true
		}
	}
	return false
}

// nearestOuterLoop returns the outer loop of loop most nearly
// containing block b; the header must dominate b.  loop itself
// is assumed to not be that loop. For acceptable performance,
// we're relying on loop nests to not be terribly deep.
func (l *loop) nearestOuterLoop(sdom SparseTree, b *Block) *loop {
	var o *loop
	for o = l.outer; o != nil && !sdom.IsAncestorEq(o.header, b); o = o.outer {
	}
	return o
}

func loopnestfor(f *Func) *loopnest {
	po := f.postorder()
	sdom := f.Sdom()
	b2l := make([]*loop, f.NumBlocks())
	loops := make([]*loop, 0)
	visited := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(visited)
	sawIrred := false

	if f.pass.debug > 2 {
		fmt.Printf("loop finding in %s\n", f.Name)
	}

	// Reducible-loop-nest-finding.
	for _, b := range po {
		if f.pass != nil && f.pass.debug > 3 {
			fmt.Printf("loop finding at %s\n", b)
		}

		var innermost *loop // innermost header reachable from this block

		// IF any successor s of b is in a loop headed by h
		// AND h dominates b
		// THEN b is in the loop headed by h.
		//
		// Choose the first/innermost such h.
		//
		// IF s itself dominates b, then s is a loop header;
		// and there may be more than one such s.
		// Since there's at most 2 successors, the inner/outer ordering
		// between them can be established with simple comparisons.
		for _, e := range b.Succs {
			bb := e.b
			l := b2l[bb.ID]

			if sdom.IsAncestorEq(bb, b) { // Found a loop header
				if f.pass != nil && f.pass.debug > 4 {
					fmt.Printf("loop finding    succ %s of %s is header\n", bb.String(), b.String())
				}
				if l == nil {
					l = &loop{header: bb, isInner: true}
					loops = append(loops, l)
					b2l[bb.ID] = l
				}
			} else if !visited[bb.ID] { // Found an irreducible loop
				sawIrred = true
				if f.pass != nil && f.pass.debug > 4 {
					fmt.Printf("loop finding    succ %s of %s is IRRED, in %s\n", bb.String(), b.String(), f.Name)
				}
			} else if l != nil {
				// TODO handle case where l is irreducible.
				// Perhaps a loop header is inherited.
				// is there any loop containing our successor whose
				// header dominates b?
				if !sdom.IsAncestorEq(l.header, b) {
					l = l.nearestOuterLoop(sdom, b)
				}
				if f.pass != nil && f.pass.debug > 4 {
					if l == nil {
						fmt.Printf("loop finding    succ %s of %s has no loop\n", bb.String(), b.String())
					} else {
						fmt.Printf("loop finding    succ %s of %s provides loop with header %s\n", bb.String(), b.String(), l.header.String())
					}
				}
			} else { // No loop
				if f.pass != nil && f.pass.debug > 4 {
					fmt.Printf("loop finding    succ %s of %s has no loop\n", bb.String(), b.String())
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
		visited[b.ID] = true
	}

	// Compute depths.
	for _, l := range loops {
		if l.depth != 0 {
			// Already computed because it is an ancestor of
			// a previous loop.
			continue
		}
		// Find depth by walking up the loop tree.
		d := int16(0)
		for x := l; x != nil; x = x.outer {
			if x.depth != 0 {
				d += x.depth
				break
			}
			d++
		}
		// Set depth for every ancestor.
		for x := l; x != nil; x = x.outer {
			if x.depth != 0 {
				break
			}
			x.depth = d
			d--
		}
	}
	// Double-check depths.
	for _, l := range loops {
		want := int16(1)
		if l.outer != nil {
			want = l.outer.depth + 1
		}
		if l.depth != want {
			l.header.Fatalf("bad depth calculation for loop %s: got %d want %d", l.header, l.depth, want)
		}
	}

	ln := &loopnest{f: f, b2l: b2l, po: po, sdom: sdom, loops: loops, hasIrreducible: sawIrred}

	// Curious about the loopiness? "-d=ssa/likelyadjust/stats"
	if f.pass != nil && f.pass.stats > 0 && len(loops) > 0 {

		// Note stats for non-innermost loops are slightly flawed because
		// they don't account for inner loop exits that span multiple levels.

		for _, l := range loops {
			inner := 0
			if l.isInner {
				inner++
			}

			f.LogStat("loopstats in "+f.Name+":",
				l.depth, "depth",
				inner, "is_inner", l.nBlocks, "n_blocks")
		}
	}

	if f.pass != nil && f.pass.debug > 1 && len(loops) > 0 {
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
	return ln
}

// depth returns the loop nesting level of block b.
func (ln *loopnest) depth(b ID) int16 {
	if l := ln.b2l[b]; l != nil {
		return l.depth
	}
	return 0
}
