// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import "fmt"

type Loop struct {
	Header *Block // The header node of this (reducible) loop
	Outer  *Loop  // loop containing this loop

	// Next three fields used by regalloc and/or
	// aid in computation of inner-ness and list of blocks.
	NBlocks int32 // Number of blocks in this loop but not within inner loops
	Depth   int16 // Nesting depth of the loop; 1 is outermost.
	IsInner bool  // True if never discovered to contain a loop

	// True if all paths through the loop have a call.
	// Computed and used by regalloc; stored here for convenience.
	ContainsUnavoidableCall bool
}

type LoopNest struct {
	F              *Func
	B2L            []*Loop
	Po             []*Block
	SDom           SparseTree
	Loops          []*Loop
	HasIrreducible bool // TODO current treatment of irreducible loops is very flaky, if accurate loops are needed, must punt at function level.
}

func Loopnestfor(f *Func) *LoopNest {
	po := f.Postorder()
	sdom := f.Sdom()
	b2l := make([]*Loop, f.NumBlocks())
	loops := make([]*Loop, 0)
	visited := f.Cache.AllocBoolSlice(f.NumBlocks())
	defer f.Cache.FreeBoolSlice(visited)
	sawIrred := false

	if f.Pass.Debug > 2 {
		fmt.Printf("loop finding in %s\n", f.Name)
	}

	// Reducible-loop-nest-finding.
	for _, b := range po {
		if f.Pass != nil && f.Pass.Debug > 3 {
			fmt.Printf("loop finding at %s\n", b)
		}

		var innermost *Loop // innermost header reachable from this block

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
			bb := e.B
			l := b2l[bb.ID]

			if sdom.IsAncestorEq(bb, b) { // Found a loop header
				if f.Pass != nil && f.Pass.Debug > 4 {
					fmt.Printf("loop finding    succ %s of %s is header\n", bb.String(), b.String())
				}
				if l == nil {
					l = &Loop{Header: bb, IsInner: true}
					loops = append(loops, l)
					b2l[bb.ID] = l
				}
			} else if !visited[bb.ID] { // Found an irreducible loop
				sawIrred = true
				if f.Pass != nil && f.Pass.Debug > 4 {
					fmt.Printf("loop finding    succ %s of %s is IRRED, in %s\n", bb.String(), b.String(), f.Name)
				}
			} else if l != nil {
				// TODO handle case where l is irreducible.
				// Perhaps a loop header is inherited.
				// is there any loop containing our successor whose
				// header dominates b?
				if !sdom.IsAncestorEq(l.Header, b) {
					l = l.nearestOuterLoop(sdom, b)
				}
				if f.Pass != nil && f.Pass.Debug > 4 {
					if l == nil {
						fmt.Printf("loop finding    succ %s of %s has no loop\n", bb.String(), b.String())
					} else {
						fmt.Printf("loop finding    succ %s of %s provides loop with header %s\n", bb.String(), b.String(), l.Header.String())
					}
				}
			} else { // No loop
				if f.Pass != nil && f.Pass.Debug > 4 {
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

			if sdom.IsAncestor(innermost.Header, l.Header) {
				sdom.outerinner(innermost, l)
				innermost = l
			} else if sdom.IsAncestor(l.Header, innermost.Header) {
				sdom.outerinner(l, innermost)
			}
		}

		if innermost != nil {
			b2l[b.ID] = innermost
			innermost.NBlocks++
		}
		visited[b.ID] = true
	}

	// Compute depths.
	for _, l := range loops {
		if l.Depth != 0 {
			// Already computed because it is an ancestor of
			// a previous loop.
			continue
		}
		// Find depth by walking up the loop tree.
		d := int16(0)
		for x := l; x != nil; x = x.Outer {
			if x.Depth != 0 {
				d += x.Depth
				break
			}
			d++
		}
		// Set depth for every ancestor.
		for x := l; x != nil; x = x.Outer {
			if x.Depth != 0 {
				break
			}
			x.Depth = d
			d--
		}
	}
	// Double-check depths.
	for _, l := range loops {
		want := int16(1)
		if l.Outer != nil {
			want = l.Outer.Depth + 1
		}
		if l.Depth != want {
			l.Header.Fatalf("bad depth calculation for loop %s: got %d want %d", l.Header, l.Depth, want)
		}
	}

	ln := &LoopNest{F: f, B2L: b2l, Po: po, SDom: sdom, Loops: loops, HasIrreducible: sawIrred}

	// Curious about the loopiness? "-d=ssa/likelyadjust/stats"
	if f.Pass != nil && f.Pass.Stats > 0 && len(loops) > 0 {

		// Note stats for non-innermost loops are slightly flawed because
		// they don't account for inner loop exits that span multiple levels.

		for _, l := range loops {
			inner := 0
			if l.IsInner {
				inner++
			}

			f.LogStat("loopstats in "+f.Name+":",
				l.Depth, "depth",
				inner, "is_inner", l.NBlocks, "n_blocks")
		}
	}

	if f.Pass != nil && f.Pass.Debug > 1 && len(loops) > 0 {
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

// outerinner records that outer contains inner
func (sdom SparseTree) outerinner(outer, inner *Loop) {
	// There could be other outer loops found in some random order,
	// locate the new outer loop appropriately among them.

	// Outer loop headers dominate inner loop headers.
	// Use this to put the "new" "outer" loop in the right place.
	oldouter := inner.Outer
	for oldouter != nil && sdom.IsAncestor(outer.Header, oldouter.Header) {
		inner = oldouter
		oldouter = inner.Outer
	}
	if outer == oldouter {
		return
	}
	if oldouter != nil {
		sdom.outerinner(oldouter, outer)
	}

	inner.Outer = outer
	outer.IsInner = false
}

func (l *Loop) String() string {
	return fmt.Sprintf("hdr:%s", l.Header)
}

func (l *Loop) LongString() string {
	i := ""
	o := ""
	if l.IsInner {
		i = ", INNER"
	}
	if l.Outer != nil {
		o = ", o=" + l.Outer.Header.String()
	}
	return fmt.Sprintf("hdr:%s%s%s", l.Header, i, o)
}

func (l *Loop) IsWithinOrEq(ll *Loop) bool {
	if ll == nil { // nil means whole program
		return true
	}
	for ; l != nil; l = l.Outer {
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
func (l *Loop) nearestOuterLoop(sdom SparseTree, b *Block) *Loop {
	var o *Loop
	for o = l.Outer; o != nil && !sdom.IsAncestorEq(o.Header, b); o = o.Outer {
	}
	return o
}

// Depth returns the loop nesting level of block b.
func (ln *LoopNest) Depth(b ID) int16 {
	if l := ln.B2L[b]; l != nil {
		return l.Depth
	}
	return 0
}
