// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// mark values
const (
	notFound    = 0 // block has not been discovered yet
	notExplored = 1 // discovered and in queue, outedges not processed yet
	explored    = 2 // discovered and in queue, outedges processed
	done        = 3 // all done, in output ordering
)

// This file contains code to compute the dominator tree
// of a control-flow graph.

// postorder computes a postorder traversal ordering for the
// basic blocks in f.  Unreachable blocks will not appear.
func postorder(f *Func) []*Block {
	mark := make([]byte, f.NumBlocks())

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
			b.Fatalf("bad stack state %v %d", b, mark[b.ID])
		}
	}
	return order
}

type linkedBlocks func(*Block) []*Block

// dfs performs a depth first search over the blocks starting at the set of
// blocks in the entries list (in arbitrary order). dfnum contains a mapping
// from block id to an int indicating the order the block was reached or
// notFound if the block was not reached.  order contains a mapping from dfnum
// to block.
func dfs(entries []*Block, succFn linkedBlocks) (dfnum []int, order []*Block, parent []*Block) {
	maxBlockID := entries[0].Func.NumBlocks()

	dfnum = make([]int, maxBlockID)
	order = make([]*Block, maxBlockID)
	parent = make([]*Block, maxBlockID)

	n := 0
	s := make([]*Block, 0, 256)
	for _, entry := range entries {
		if dfnum[entry.ID] != notFound {
			continue // already found from a previous entry
		}
		s = append(s, entry)
		parent[entry.ID] = entry
		for len(s) > 0 {
			node := s[len(s)-1]
			s = s[:len(s)-1]

			n++
			for _, w := range succFn(node) {
				// if it has a dfnum, we've already visited it
				if dfnum[w.ID] == notFound {
					s = append(s, w)
					parent[w.ID] = node
					dfnum[w.ID] = notExplored
				}
			}
			dfnum[node.ID] = n
			order[n] = node
		}
	}

	return
}

// dominators computes the dominator tree for f.  It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil.  The entry block maps to nil.
func dominators(f *Func) []*Block {
	preds := func(b *Block) []*Block { return b.Preds }
	succs := func(b *Block) []*Block { return b.Succs }

	//TODO: benchmark and try to find criteria for swapping between
	// dominatorsSimple and dominatorsLT
	return dominatorsLT([]*Block{f.Entry}, preds, succs)
}

// postDominators computes the post-dominator tree for f.
func postDominators(f *Func) []*Block {
	preds := func(b *Block) []*Block { return b.Preds }
	succs := func(b *Block) []*Block { return b.Succs }

	if len(f.Blocks) == 0 {
		return nil
	}

	// find the exit blocks
	var exits []*Block
	for i := len(f.Blocks) - 1; i >= 0; i-- {
		switch f.Blocks[i].Kind {
		case BlockExit, BlockRet, BlockRetJmp, BlockCall, BlockCheck:
			exits = append(exits, f.Blocks[i])
			break
		}
	}

	// infinite loop with no exit
	if exits == nil {
		return make([]*Block, f.NumBlocks())
	}
	return dominatorsLT(exits, succs, preds)
}

// dominatorsLt runs Lengauer-Tarjan to compute a dominator tree starting at
// entry and using predFn/succFn to find predecessors/successors to allow
// computing both dominator and post-dominator trees.
func dominatorsLT(entries []*Block, predFn linkedBlocks, succFn linkedBlocks) []*Block {
	// Based on Lengauer-Tarjan from Modern Compiler Implementation in C -
	// Appel with optimizations from Finding Dominators in Practice -
	// Georgiadis

	// Step 1. Carry out a depth first search of the problem graph. Number
	// the vertices from 1 to n as they are reached during the search.
	dfnum, vertex, parent := dfs(entries, succFn)

	maxBlockID := entries[0].Func.NumBlocks()
	semi := make([]*Block, maxBlockID)
	samedom := make([]*Block, maxBlockID)
	idom := make([]*Block, maxBlockID)
	ancestor := make([]*Block, maxBlockID)
	best := make([]*Block, maxBlockID)
	bucket := make([]*Block, maxBlockID)

	// Step 2. Compute the semidominators of all vertices by applying
	// Theorem 4.  Carry out the computation vertex by vertex in decreasing
	// order by number.
	for i := maxBlockID - 1; i > 0; i-- {
		w := vertex[i]
		if w == nil {
			continue
		}

		if dfnum[w.ID] == notFound {
			// skip unreachable node
			continue
		}

		// Step 3. Implicitly define the immediate dominator of each
		// vertex by applying Corollary 1. (reordered)
		for v := bucket[w.ID]; v != nil; v = bucket[v.ID] {
			u := eval(v, ancestor, semi, dfnum, best)
			if semi[u.ID] == semi[v.ID] {
				idom[v.ID] = w // true dominator
			} else {
				samedom[v.ID] = u // v has same dominator as u
			}
		}

		p := parent[w.ID]
		s := p // semidominator

		var sp *Block
		// calculate the semidominator of w
		for _, v := range w.Preds {
			if dfnum[v.ID] == notFound {
				// skip unreachable predecessor
				continue
			}

			if dfnum[v.ID] <= dfnum[w.ID] {
				sp = v
			} else {
				sp = semi[eval(v, ancestor, semi, dfnum, best).ID]
			}

			if dfnum[sp.ID] < dfnum[s.ID] {
				s = sp
			}
		}

		// link
		ancestor[w.ID] = p
		best[w.ID] = w

		semi[w.ID] = s
		if semi[s.ID] != parent[s.ID] {
			bucket[w.ID] = bucket[s.ID]
			bucket[s.ID] = w
		}
	}

	// Final pass of step 3
	for v := bucket[0]; v != nil; v = bucket[v.ID] {
		idom[v.ID] = bucket[0]
	}

	// Step 4. Explictly define the immediate dominator of each vertex,
	// carrying out the computation vertex by vertex in increasing order by
	// number.
	for i := 1; i < maxBlockID-1; i++ {
		w := vertex[i]
		if w == nil {
			continue
		}
		// w has the same dominator as samedom[w.ID]
		if samedom[w.ID] != nil {
			idom[w.ID] = idom[samedom[w.ID].ID]
		}
	}
	return idom
}

// eval function from LT paper with path compression
func eval(v *Block, ancestor []*Block, semi []*Block, dfnum []int, best []*Block) *Block {
	a := ancestor[v.ID]
	if ancestor[a.ID] != nil {
		b := eval(a, ancestor, semi, dfnum, best)
		ancestor[v.ID] = ancestor[a.ID]
		if dfnum[semi[b.ID].ID] < dfnum[semi[best[v.ID].ID].ID] {
			best[v.ID] = b
		}
	}
	return best[v.ID]
}

// dominators computes the dominator tree for f.  It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil.  The entry block maps to nil.
func dominatorsSimple(f *Func) []*Block {
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
		f.Fatalf("entry block %v not last in postorder", f.Entry)
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
	// TODO: This loop is O(n^2). See BenchmarkNilCheckDeep*.
	for b != c {
		if postnum[b.ID] < postnum[c.ID] {
			b = idom[b.ID]
		} else {
			c = idom[c.ID]
		}
	}
	return b
}
