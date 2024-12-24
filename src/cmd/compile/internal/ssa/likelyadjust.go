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

	// By default, children, exits, and depth are not initialized.
	children []*loop  // loops nested directly within this loop. Initialized by assembleChildren().
	exits    []*Block // exits records blocks reached by exits from this loop. Initialized by findExits().

	// Next three fields used by regalloc and/or
	// aid in computation of inner-ness and list of blocks.
	nBlocks int32 // Number of blocks in this loop but not within inner loops
	depth   int16 // Nesting depth of the loop; 1 is outermost. Initialized by calculateDepths().
	isInner bool  // True if never discovered to contain a loop

	// register allocation uses this.
	containsUnavoidableCall bool // True if all paths through the loop have a call
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

func checkContainsCall(bb *Block) bool {
	if bb.Kind == BlockDefer {
		return true
	}
	for _, v := range bb.Values {
		if opcodeTable[v.Op].call {
			return true
		}
	}
	return false
}

type loopnest struct {
	f              *Func
	b2l            []*loop
	po             []*Block
	sdom           SparseTree
	loops          []*loop
	hasIrreducible bool // TODO current treatment of irreducible loops is very flaky, if accurate loops are needed, must punt at function level.

	// Record which of the lazily initialized fields have actually been initialized.
	initializedChildren, initializedDepth, initializedExits bool
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

	ln := &loopnest{f: f, b2l: b2l, po: po, sdom: sdom, loops: loops, hasIrreducible: sawIrred}

	// Calculate containsUnavoidableCall for regalloc
	dominatedByCall := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(dominatedByCall)
	for _, b := range po {
		if checkContainsCall(b) {
			dominatedByCall[b.ID] = true
		}
	}
	// Run dfs to find path through the loop that avoids all calls.
	// Such path either escapes loop or return back to header.
	// It isn't enough to have exit not dominated by any call, for example:
	// ... some loop
	// call1   call2
	//   \      /
	//     exit
	// ...
	// exit is not dominated by any call, but we don't have call-free path to it.
	for _, l := range loops {
		// Header contains call.
		if dominatedByCall[l.header.ID] {
			l.containsUnavoidableCall = true
			continue
		}
		callfreepath := false
		tovisit := make([]*Block, 0, len(l.header.Succs))
		// Push all non-loop non-exit successors of header onto toVisit.
		for _, s := range l.header.Succs {
			nb := s.Block()
			// This corresponds to loop with zero iterations.
			if !l.iterationEnd(nb, b2l) {
				tovisit = append(tovisit, nb)
			}
		}
		for len(tovisit) > 0 {
			cur := tovisit[len(tovisit)-1]
			tovisit = tovisit[:len(tovisit)-1]
			if dominatedByCall[cur.ID] {
				continue
			}
			// Record visited in dominatedByCall.
			dominatedByCall[cur.ID] = true
			for _, s := range cur.Succs {
				nb := s.Block()
				if l.iterationEnd(nb, b2l) {
					callfreepath = true
				}
				if !dominatedByCall[nb.ID] {
					tovisit = append(tovisit, nb)
				}

			}
			if callfreepath {
				break
			}
		}
		if !callfreepath {
			l.containsUnavoidableCall = true
		}
	}

	// Curious about the loopiness? "-d=ssa/likelyadjust/stats"
	if f.pass != nil && f.pass.stats > 0 && len(loops) > 0 {
		ln.assembleChildren()
		ln.calculateDepths()
		ln.findExits()

		// Note stats for non-innermost loops are slightly flawed because
		// they don't account for inner loop exits that span multiple levels.

		for _, l := range loops {
			x := len(l.exits)
			cf := 0
			if !l.containsUnavoidableCall {
				cf = 1
			}
			inner := 0
			if l.isInner {
				inner++
			}

			f.LogStat("loopstats:",
				l.depth, "depth", x, "exits",
				inner, "is_inner", cf, "always_calls", l.nBlocks, "n_blocks")
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

// assembleChildren initializes the children field of each
// loop in the nest.  Loop A is a child of loop B if A is
// directly nested within B (based on the reducible-loops
// detection above)
func (ln *loopnest) assembleChildren() {
	if ln.initializedChildren {
		return
	}
	for _, l := range ln.loops {
		if l.outer != nil {
			l.outer.children = append(l.outer.children, l)
		}
	}
	ln.initializedChildren = true
}

// calculateDepths uses the children field of loops
// to determine the nesting depth (outer=1) of each
// loop.  This is helpful for finding exit edges.
func (ln *loopnest) calculateDepths() {
	if ln.initializedDepth {
		return
	}
	ln.assembleChildren()
	for _, l := range ln.loops {
		if l.outer == nil {
			l.setDepth(1)
		}
	}
	ln.initializedDepth = true
}

// findExits uses loop depth information to find the
// exits from a loop.
func (ln *loopnest) findExits() {
	if ln.initializedExits {
		return
	}
	ln.calculateDepths()
	b2l := ln.b2l
	for _, b := range ln.po {
		l := b2l[b.ID]
		if l != nil && len(b.Succs) == 2 {
			sl := b2l[b.Succs[0].b.ID]
			if recordIfExit(l, sl, b.Succs[0].b) {
				continue
			}
			sl = b2l[b.Succs[1].b.ID]
			if recordIfExit(l, sl, b.Succs[1].b) {
				continue
			}
		}
	}
	ln.initializedExits = true
}

// depth returns the loop nesting level of block b.
func (ln *loopnest) depth(b ID) int16 {
	if l := ln.b2l[b]; l != nil {
		return l.depth
	}
	return 0
}

// recordIfExit checks sl (the loop containing b) to see if it
// is outside of loop l, and if so, records b as an exit block
// from l and returns true.
func recordIfExit(l, sl *loop, b *Block) bool {
	if sl != l {
		if sl == nil || sl.depth <= l.depth {
			l.exits = append(l.exits, b)
			return true
		}
		// sl is not nil, and is deeper than l
		// it's possible for this to be a goto into an irreducible loop made from gotos.
		for sl.depth > l.depth {
			sl = sl.outer
		}
		if sl != l {
			l.exits = append(l.exits, b)
			return true
		}
	}
	return false
}

func (l *loop) setDepth(d int16) {
	l.depth = d
	for _, c := range l.children {
		c.setDepth(d + 1)
	}
}

// iterationEnd checks if block b ends iteration of loop l.
// Ending iteration means either escaping to outer loop/code or
// going back to header
func (l *loop) iterationEnd(b *Block, b2l []*loop) bool {
	return b == l.header || b2l[b.ID] == nil || (b2l[b.ID] != l && b2l[b.ID].depth <= l.depth)
}
