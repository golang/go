// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file contains code to compute the dominator tree
// of a control-flow graph.

import "log"

// postorder computes a postorder traversal ordering for the
// basic blocks in f.  Unreachable blocks will not appear.
func postorder(f *Func) []*Block {
	mark := make([]byte, f.NumBlocks())
	// mark values
	const (
		notFound    = 0 // block has not been discovered yet
		notExplored = 1 // discovered and in queue, outedges not processed yet
		explored    = 2 // discovered and in queue, outedges processed
		done        = 3 // all done, in output ordering
	)

	// result ordering
	var order []*Block

	// stack of blocks
	var s []*Block
	s = append(s, f.Entry)
	mark[f.Entry.ID] = notExplored
	for len(s) > 0 {
		b := s[len(s)-1]
		switch mark[b.ID] {
		case explored:
			// Children have all been visited.  Pop & output block.
			s = s[:len(s)-1]
			mark[b.ID] = done
			order = append(order, b)
		case notExplored:
			// Children have not been visited yet.  Mark as explored
			// and queue any children we haven't seen yet.
			mark[b.ID] = explored
			for _, c := range b.Succs {
				if mark[c.ID] == notFound {
					mark[c.ID] = notExplored
					s = append(s, c)
				}
			}
		default:
			log.Fatalf("bad stack state %v %d", b, mark[b.ID])
		}
	}
	return order
}

// dominators computes the dominator tree for f.  It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil.  The entry block maps to nil.
func dominators(f *Func) []*Block {
	// A simple algorithm for now
	// Cooper, Harvey, Kennedy
	idom := make([]*Block, f.NumBlocks())

	// Compute postorder walk
	post := postorder(f)

	// Make map from block id to order index (for intersect call)
	postnum := make([]int, f.NumBlocks())
	for i, b := range post {
		postnum[b.ID] = i
	}

	// Make the entry block a self-loop
	idom[f.Entry.ID] = f.Entry
	if postnum[f.Entry.ID] != len(post)-1 {
		log.Fatalf("entry block %v not last in postorder", f.Entry)
	}

	// Compute relaxation of idom entries
	for {
		changed := false

		for i := len(post) - 2; i >= 0; i-- {
			b := post[i]
			var d *Block
			for _, p := range b.Preds {
				if idom[p.ID] == nil {
					continue
				}
				if d == nil {
					d = p
					continue
				}
				d = intersect(d, p, postnum, idom)
			}
			if d != idom[b.ID] {
				idom[b.ID] = d
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	// Set idom of entry block to nil instead of itself.
	idom[f.Entry.ID] = nil
	return idom
}

// intersect finds the closest dominator of both b and c.
// It requires a postorder numbering of all the blocks.
func intersect(b, c *Block, postnum []int, idom []*Block) *Block {
	for b != c {
		if postnum[b.ID] < postnum[c.ID] {
			b = idom[b.ID]
		} else {
			c = idom[c.ID]
		}
	}
	return b
}
