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
// basic blocks in f. Unreachable blocks will not appear.
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
			// Children have all been visited. Pop & output block.
			s = s[:len(s)-1]
			mark[b.ID] = done
			order = append(order, b)
		case notExplored:
			// Children have not been visited yet. Mark as explored
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

const nscratchslices = 8

// experimentally, functions with 512 or fewer blocks account
// for 75% of memory (size) allocation for dominator computation
// in make.bash.
const minscratchblocks = 512

func (cfg *Config) scratchBlocksForDom(maxBlockID int) (a, b, c, d, e, f, g, h []ID) {
	tot := maxBlockID * nscratchslices
	scratch := cfg.domblockstore
	if len(scratch) < tot {
		// req = min(1.5*tot, nscratchslices*minscratchblocks)
		// 50% padding allows for graph growth in later phases.
		req := (tot * 3) >> 1
		if req < nscratchslices*minscratchblocks {
			req = nscratchslices * minscratchblocks
		}
		scratch = make([]ID, req)
		cfg.domblockstore = scratch
	} else {
		// Clear as much of scratch as we will (re)use
		scratch = scratch[0:tot]
		for i := range scratch {
			scratch[i] = 0
		}
	}

	a = scratch[0*maxBlockID : 1*maxBlockID]
	b = scratch[1*maxBlockID : 2*maxBlockID]
	c = scratch[2*maxBlockID : 3*maxBlockID]
	d = scratch[3*maxBlockID : 4*maxBlockID]
	e = scratch[4*maxBlockID : 5*maxBlockID]
	f = scratch[5*maxBlockID : 6*maxBlockID]
	g = scratch[6*maxBlockID : 7*maxBlockID]
	h = scratch[7*maxBlockID : 8*maxBlockID]

	return
}

// dfs performs a depth first search over the blocks starting at the set of
// blocks in the entries list (in arbitrary order). dfnum contains a mapping
// from block id to an int indicating the order the block was reached or
// notFound if the block was not reached.  order contains a mapping from dfnum
// to block.
func (f *Func) dfs(entries []*Block, succFn linkedBlocks, dfnum, order, parent []ID) (fromID []*Block) {
	maxBlockID := entries[0].Func.NumBlocks()

	fromID = make([]*Block, maxBlockID)

	for _, entry := range entries[0].Func.Blocks {
		eid := entry.ID
		if fromID[eid] != nil {
			panic("Colliding entry IDs")
		}
		fromID[eid] = entry
	}

	n := ID(0)
	s := make([]*Block, 0, 256)
	for _, entry := range entries {
		if dfnum[entry.ID] != notFound {
			continue // already found from a previous entry
		}
		s = append(s, entry)
		parent[entry.ID] = entry.ID
		for len(s) > 0 {
			node := s[len(s)-1]
			s = s[:len(s)-1]

			n++
			for _, w := range succFn(node) {
				// if it has a dfnum, we've already visited it
				if dfnum[w.ID] == notFound {
					s = append(s, w)
					parent[w.ID] = node.ID
					dfnum[w.ID] = notExplored
				}
			}
			dfnum[node.ID] = n
			order[n] = node.ID
		}
	}

	return
}

// dominators computes the dominator tree for f. It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil. The entry block maps to nil.
func dominators(f *Func) []*Block {
	preds := func(b *Block) []*Block { return b.Preds }
	succs := func(b *Block) []*Block { return b.Succs }

	//TODO: benchmark and try to find criteria for swapping between
	// dominatorsSimple and dominatorsLT
	return f.dominatorsLT([]*Block{f.Entry}, preds, succs)
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
	for _, b := range f.Blocks {
		switch b.Kind {
		case BlockExit, BlockRet, BlockRetJmp, BlockCall, BlockCheck:
			exits = append(exits, b)
		}
	}

	// infinite loop with no exit
	if exits == nil {
		return make([]*Block, f.NumBlocks())
	}
	return f.dominatorsLT(exits, succs, preds)
}

// dominatorsLt runs Lengauer-Tarjan to compute a dominator tree starting at
// entry and using predFn/succFn to find predecessors/successors to allow
// computing both dominator and post-dominator trees.
func (f *Func) dominatorsLT(entries []*Block, predFn linkedBlocks, succFn linkedBlocks) []*Block {
	// Based on Lengauer-Tarjan from Modern Compiler Implementation in C -
	// Appel with optimizations from Finding Dominators in Practice -
	// Georgiadis

	maxBlockID := entries[0].Func.NumBlocks()

	dfnum, vertex, parent, semi, samedom, ancestor, best, bucket := f.Config.scratchBlocksForDom(maxBlockID)

	// dfnum := make([]ID, maxBlockID) // conceptually int32, but punning for allocation purposes.
	// vertex := make([]ID, maxBlockID)
	// parent := make([]ID, maxBlockID)

	// semi := make([]ID, maxBlockID)
	// samedom := make([]ID, maxBlockID)
	// ancestor := make([]ID, maxBlockID)
	// best := make([]ID, maxBlockID)
	// bucket := make([]ID, maxBlockID)

	// Step 1. Carry out a depth first search of the problem graph. Number
	// the vertices from 1 to n as they are reached during the search.
	fromID := f.dfs(entries, succFn, dfnum, vertex, parent)

	idom := make([]*Block, maxBlockID)

	// Step 2. Compute the semidominators of all vertices by applying
	// Theorem 4.  Carry out the computation vertex by vertex in decreasing
	// order by number.
	for i := maxBlockID - 1; i > 0; i-- {
		w := vertex[i]
		if w == 0 {
			continue
		}

		if dfnum[w] == notFound {
			// skip unreachable node
			continue
		}

		// Step 3. Implicitly define the immediate dominator of each
		// vertex by applying Corollary 1. (reordered)
		for v := bucket[w]; v != 0; v = bucket[v] {
			u := eval(v, ancestor, semi, dfnum, best)
			if semi[u] == semi[v] {
				idom[v] = fromID[w] // true dominator
			} else {
				samedom[v] = u // v has same dominator as u
			}
		}

		p := parent[w]
		s := p // semidominator

		var sp ID
		// calculate the semidominator of w
		for _, v := range predFn(fromID[w]) {
			if dfnum[v.ID] == notFound {
				// skip unreachable predecessor
				continue
			}

			if dfnum[v.ID] <= dfnum[w] {
				sp = v.ID
			} else {
				sp = semi[eval(v.ID, ancestor, semi, dfnum, best)]
			}

			if dfnum[sp] < dfnum[s] {
				s = sp
			}
		}

		// link
		ancestor[w] = p
		best[w] = w

		semi[w] = s
		if semi[s] != parent[s] {
			bucket[w] = bucket[s]
			bucket[s] = w
		}
	}

	// Final pass of step 3
	for v := bucket[0]; v != 0; v = bucket[v] {
		idom[v] = fromID[bucket[0]]
	}

	// Step 4. Explictly define the immediate dominator of each vertex,
	// carrying out the computation vertex by vertex in increasing order by
	// number.
	for i := 1; i < maxBlockID-1; i++ {
		w := vertex[i]
		if w == 0 {
			continue
		}
		// w has the same dominator as samedom[w]
		if samedom[w] != 0 {
			idom[w] = idom[samedom[w]]
		}
	}
	return idom
}

// eval function from LT paper with path compression
func eval(v ID, ancestor []ID, semi []ID, dfnum []ID, best []ID) ID {
	a := ancestor[v]
	if ancestor[a] != 0 {
		bid := eval(a, ancestor, semi, dfnum, best)
		ancestor[v] = ancestor[a]
		if dfnum[semi[bid]] < dfnum[semi[best[v]]] {
			best[v] = bid
		}
	}
	return best[v]
}

// dominators computes the dominator tree for f. It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil. The entry block maps to nil.
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
