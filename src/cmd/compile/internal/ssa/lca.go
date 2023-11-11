// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math/bits"
)

// Code to compute lowest common ancestors in the dominator tree.
// https://en.wikipedia.org/wiki/Lowest_common_ancestor
// https://en.wikipedia.org/wiki/Range_minimum_query#Solution_using_constant_time_and_linearithmic_space

// lcaRange is a data structure that can compute lowest common ancestor queries
// in O(n lg n) precomputed space and O(1) time per query.
type lcaRange struct {
	// Additional information about each block (indexed by block ID).
	blocks []lcaRangeBlock

	// Data structure for range minimum queries.
	// rangeMin[k][i] contains the ID of the minimum depth block
	// in the Euler tour from positions i to i+1<<k-1, inclusive.
	rangeMin [][]ID
}

type lcaRangeBlock struct {
	b          *Block
	parent     ID    // parent in dominator tree.  0 = no parent (entry or unreachable)
	firstChild ID    // first child in dominator tree
	sibling    ID    // next child of parent
	pos        int32 // an index in the Euler tour where this block appears (any one of its occurrences)
	depth      int32 // depth in dominator tree (root=0, its children=1, etc.)
}

func makeLCArange(f *Func) *lcaRange {
	dom := f.Idom()

	// Build tree
	blocks := make([]lcaRangeBlock, f.NumBlocks())
	for _, b := range f.Blocks {
		blocks[b.ID].b = b
		if dom[b.ID] == nil {
			continue // entry or unreachable
		}
		parent := dom[b.ID].ID
		blocks[b.ID].parent = parent
		blocks[b.ID].sibling = blocks[parent].firstChild
		blocks[parent].firstChild = b.ID
	}

	// Compute euler tour ordering.
	// Each reachable block will appear #children+1 times in the tour.
	tour := make([]ID, 0, f.NumBlocks()*2-1)
	type queueEntry struct {
		bid ID // block to work on
		cid ID // child we're already working on (0 = haven't started yet)
	}
	q := []queueEntry{{f.Entry.ID, 0}}
	for len(q) > 0 {
		n := len(q) - 1
		bid := q[n].bid
		cid := q[n].cid
		q = q[:n]

		// Add block to tour.
		blocks[bid].pos = int32(len(tour))
		tour = append(tour, bid)

		// Proceed down next child edge (if any).
		if cid == 0 {
			// This is our first visit to b. Set its depth.
			blocks[bid].depth = blocks[blocks[bid].parent].depth + 1
			// Then explore its first child.
			cid = blocks[bid].firstChild
		} else {
			// We've seen b before. Explore the next child.
			cid = blocks[cid].sibling
		}
		if cid != 0 {
			q = append(q, queueEntry{bid, cid}, queueEntry{cid, 0})
		}
	}

	// Compute fast range-minimum query data structure
	rangeMin := make([][]ID, 0, bits.Len64(uint64(len(tour))))
	rangeMin = append(rangeMin, tour) // 1-size windows are just the tour itself.
	for logS, s := 1, 2; s < len(tour); logS, s = logS+1, s*2 {
		r := make([]ID, len(tour)-s+1)
		for i := 0; i < len(tour)-s+1; i++ {
			bid := rangeMin[logS-1][i]
			bid2 := rangeMin[logS-1][i+s/2]
			if blocks[bid2].depth < blocks[bid].depth {
				bid = bid2
			}
			r[i] = bid
		}
		rangeMin = append(rangeMin, r)
	}

	return &lcaRange{blocks: blocks, rangeMin: rangeMin}
}

// find returns the lowest common ancestor of a and b.
func (lca *lcaRange) find(a, b *Block) *Block {
	if a == b {
		return a
	}
	// Find the positions of a and b in the Euler tour.
	p1 := lca.blocks[a.ID].pos
	p2 := lca.blocks[b.ID].pos
	if p1 > p2 {
		p1, p2 = p2, p1
	}

	// The lowest common ancestor is the minimum depth block
	// on the tour from p1 to p2.  We've precomputed minimum
	// depth blocks for powers-of-two subsequences of the tour.
	// Combine the right two precomputed values to get the answer.
	logS := uint(log64(int64(p2 - p1)))
	bid1 := lca.rangeMin[logS][p1]
	bid2 := lca.rangeMin[logS][p2-1<<logS+1]
	if lca.blocks[bid1].depth < lca.blocks[bid2].depth {
		return lca.blocks[bid1].b
	}
	return lca.blocks[bid2].b
}
