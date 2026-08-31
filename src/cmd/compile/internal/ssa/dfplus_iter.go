// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"container/heap"
	"iter"
)

// DF(x), the dominance frontier of x, holds every block y such that x
// dominates a predecessor of y but does not strictly dominate y. Its
// transitive closure DF+ (also called the merge set) is where a phi
// may need to be placed for a variable defined in x. DF+ of a set of
// blocks S, denoted IDF(S) (iterated dominance frontier), is the union
// of the DF+ of the blocks in S.

// IterDomFrontierPlus iterates the DF+ of seeds: every block at which
// a phi may need to be placed if a variable were defined in the seed
// blocks. For each frontier block, it also yields the block whose
// outgoing edge discovered the frontier. Frontier blocks are yielded
// at most once, in a deterministic order; an early break stops the walk.
// Seed blocks themselves are not yielded as such, but a seed that is
// also a merge point (e.g. a loop header) is.
// seeds iterator is consumed in full before the walk starts (the current
// algorithm has to walk deeper roots first).
// CFG must not change while iteration is in progress; inserting
// values (like phis) is fine.
func (f *Func) IterDomFrontierPlus(seeds iter.Seq[*Block]) iter.Seq2[*Block, *Block] {
	return func(yield func(*Block, *Block) bool) {
		// Materialize the seeds into a pooled slice reused by walkDFPlus.
		s := f.Cache.AllocBlockSlice(f.NumBlocks())[:0]
		defer f.Cache.FreeBlockSlice(s[:cap(s)])
		for b := range seeds {
			s = append(s, b)
		}
		f.walkDFPlus(s, yield)
	}
}

// Per-block state of a DF+ walk, packed into one flag byte per block.
// None of the bits is cleared during the walk. Each of the following happens at
// most once per block:
// - enters the work queue,
// - is banked as a root,
// - is yielded.
const (
	// The block's subtree walk is done or pending on q.
	flagQueued = 1 << iota
	// The block has been added to the PiggyBank: a seed, or a block
	// yielded earlier in this walk.
	flagPiggyBanked
	// The block has been yielded to the caller.
	flagYielded
)

// walkDFPlus is the engine under IterDomFrontierPlus.
// The walk is the Sreedhar & Gao DJ-graph algorithm, "A Linear Time
// Algorithm for Placing Φ-Nodes". Work is proportional to the dominator
// subtrees walked (skipping subtrees already covered, deeper roots)
// plus the frontier found, and memory is O(f.NumBlocks()). The walk reads
// the CFG's edges and uses the cached dominator tree.
// The seeds slice is reused in place by the PiggyBank.
func (f *Func) walkDFPlus(seeds []*Block, yield func(*Block, *Block) bool) {
	sdom := f.Sdom()

	// Roots to process, deepest first.
	piggyBank := blockHeap{t: sdom, a: seeds[:0]}

	// The worklist is a pooled slice, freed after the walk is done.
	// Each block enters it at most once, so it never outgrows its capacity.
	q := f.Cache.AllocBlockSlice(f.NumBlocks())[:0]
	defer f.Cache.FreeBlockSlice(q[:cap(q)])

	// per-block walk state; see the flag constants above.
	flags := f.Cache.AllocInt8Slice(f.NumBlocks())
	defer f.Cache.FreeInt8Slice(flags)

	// Bank the seeds as roots, compacting in place to drop duplicates.
	for _, b := range seeds {
		if flags[b.ID]&flagPiggyBanked == 0 {
			flags[b.ID] |= flagPiggyBanked
			piggyBank.a = append(piggyBank.a, b)
		}
	}
	heap.Init(&piggyBank)

	// Visit the roots from deepest to shallowest.
	for len(piggyBank.a) > 0 {
		currentRoot := heap.Pop(&piggyBank).(*Block)
		// Walk the subtree below the root, skipping subtrees already
		// covered by previous (deeper) roots, and find the edges
		// exiting it: their targets are the dominance frontier.
		// Roots are popped deepest first, so any block a later root's walk could
		// queue lies strictly below that root and was already queued
		// by its own root-push.
		if flags[currentRoot.ID]&flagQueued != 0 {
			f.Fatalf("root already in queue")
		}
		flags[currentRoot.ID] |= flagQueued
		q = append(q, currentRoot)
		for len(q) > 0 {
			b := q[len(q)-1]
			q = q[:len(q)-1]

			currentRootLevel := sdom.Level(currentRoot)
			for _, e := range b.Succs {
				c := e.Block()
				if sdom.Level(c) > currentRootLevel {
					// a D-edge, or an edge whose target is in currentRoot's subtree.
					continue
				}
				if flags[c.ID]&flagYielded != 0 {
					continue
				}
				flags[c.ID] |= flagYielded
				if flags[c.ID]&flagPiggyBanked == 0 {
					// Bank c as a root; its subtree may find further frontier edges.
					// Invariant: piggyBanked = seeds ∪ yielded
					flags[c.ID] |= flagPiggyBanked
					heap.Push(&piggyBank, c)
				}
				if !yield(c, b) {
					return
				}
			}

			// Visit children if they have not been visited yet.
			for ch := sdom.Child(b); ch != nil; ch = sdom.Sibling(ch) {
				if flags[ch.ID]&flagQueued == 0 {
					flags[ch.ID] |= flagQueued
					q = append(q, ch)
				}
			}
		}
	}
}

// A block heap is used as a priority queue to implement the PiggyBank
// from Sreedhar and Gao.  That paper uses an array which is better
// asymptotically but worse in the common case when the PiggyBank
// holds a sparse set of blocks.
type blockHeap struct {
	a []*Block   // blocks in heap
	t SparseTree // dominator tree; provides block levels for priority
}

func (h *blockHeap) Len() int      { return len(h.a) }
func (h *blockHeap) Swap(i, j int) { a := h.a; a[i], a[j] = a[j], a[i] }

func (h *blockHeap) Push(x any) {
	v := x.(*Block)
	h.a = append(h.a, v)
}
func (h *blockHeap) Pop() any {
	old := h.a
	n := len(old)
	x := old[n-1]
	h.a = old[:n-1]
	return x
}
func (h *blockHeap) Less(i, j int) bool {
	return h.t.Level(h.a[i]) > h.t.Level(h.a[j])
}
