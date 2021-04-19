// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// mark values
type markKind uint8

const (
	notFound    markKind = 0 // block has not been discovered yet
	notExplored markKind = 1 // discovered and in queue, outedges not processed yet
	explored    markKind = 2 // discovered and in queue, outedges processed
	done        markKind = 3 // all done, in output ordering
)

// This file contains code to compute the dominator tree
// of a control-flow graph.

// postorder computes a postorder traversal ordering for the
// basic blocks in f. Unreachable blocks will not appear.
func postorder(f *Func) []*Block {
	return postorderWithNumbering(f, []int32{})
}
func postorderWithNumbering(f *Func, ponums []int32) []*Block {
	mark := make([]markKind, f.NumBlocks())

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
			if len(ponums) > 0 {
				ponums[b.ID] = int32(len(order))
			}
			order = append(order, b)
		case notExplored:
			// Children have not been visited yet. Mark as explored
			// and queue any children we haven't seen yet.
			mark[b.ID] = explored
			for _, e := range b.Succs {
				c := e.b
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

type linkedBlocks func(*Block) []Edge

const nscratchslices = 7

// experimentally, functions with 512 or fewer blocks account
// for 75% of memory (size) allocation for dominator computation
// in make.bash.
const minscratchblocks = 512

func (cfg *Config) scratchBlocksForDom(maxBlockID int) (a, b, c, d, e, f, g []ID) {
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

	return
}

func dominators(f *Func) []*Block {
	preds := func(b *Block) []Edge { return b.Preds }
	succs := func(b *Block) []Edge { return b.Succs }

	//TODO: benchmark and try to find criteria for swapping between
	// dominatorsSimple and dominatorsLT
	return f.dominatorsLTOrig(f.Entry, preds, succs)
}

// dominatorsLTOrig runs Lengauer-Tarjan to compute a dominator tree starting at
// entry and using predFn/succFn to find predecessors/successors to allow
// computing both dominator and post-dominator trees.
func (f *Func) dominatorsLTOrig(entry *Block, predFn linkedBlocks, succFn linkedBlocks) []*Block {
	// Adapted directly from the original TOPLAS article's "simple" algorithm

	maxBlockID := entry.Func.NumBlocks()
	semi, vertex, label, parent, ancestor, bucketHead, bucketLink := f.Config.scratchBlocksForDom(maxBlockID)

	// This version uses integers for most of the computation,
	// to make the work arrays smaller and pointer-free.
	// fromID translates from ID to *Block where that is needed.
	fromID := make([]*Block, maxBlockID)
	for _, v := range f.Blocks {
		fromID[v.ID] = v
	}
	idom := make([]*Block, maxBlockID)

	// Step 1. Carry out a depth first search of the problem graph. Number
	// the vertices from 1 to n as they are reached during the search.
	n := f.dfsOrig(entry, succFn, semi, vertex, label, parent)

	for i := n; i >= 2; i-- {
		w := vertex[i]

		// step2 in TOPLAS paper
		for _, e := range predFn(fromID[w]) {
			v := e.b
			if semi[v.ID] == 0 {
				// skip unreachable predecessor
				// not in original, but we're using existing pred instead of building one.
				continue
			}
			u := evalOrig(v.ID, ancestor, semi, label)
			if semi[u] < semi[w] {
				semi[w] = semi[u]
			}
		}

		// add w to bucket[vertex[semi[w]]]
		// implement bucket as a linked list implemented
		// in a pair of arrays.
		vsw := vertex[semi[w]]
		bucketLink[w] = bucketHead[vsw]
		bucketHead[vsw] = w

		linkOrig(parent[w], w, ancestor)

		// step3 in TOPLAS paper
		for v := bucketHead[parent[w]]; v != 0; v = bucketLink[v] {
			u := evalOrig(v, ancestor, semi, label)
			if semi[u] < semi[v] {
				idom[v] = fromID[u]
			} else {
				idom[v] = fromID[parent[w]]
			}
		}
	}
	// step 4 in toplas paper
	for i := ID(2); i <= n; i++ {
		w := vertex[i]
		if idom[w].ID != vertex[semi[w]] {
			idom[w] = idom[idom[w].ID]
		}
	}

	return idom
}

// dfs performs a depth first search over the blocks starting at entry block
// (in arbitrary order).  This is a de-recursed version of dfs from the
// original Tarjan-Lengauer TOPLAS article.  It's important to return the
// same values for parent as the original algorithm.
func (f *Func) dfsOrig(entry *Block, succFn linkedBlocks, semi, vertex, label, parent []ID) ID {
	n := ID(0)
	s := make([]*Block, 0, 256)
	s = append(s, entry)

	for len(s) > 0 {
		v := s[len(s)-1]
		s = s[:len(s)-1]
		// recursing on v

		if semi[v.ID] != 0 {
			continue // already visited
		}
		n++
		semi[v.ID] = n
		vertex[n] = v.ID
		label[v.ID] = v.ID
		// ancestor[v] already zero
		for _, e := range succFn(v) {
			w := e.b
			// if it has a dfnum, we've already visited it
			if semi[w.ID] == 0 {
				// yes, w can be pushed multiple times.
				s = append(s, w)
				parent[w.ID] = v.ID // keep overwriting this till it is visited.
			}
		}
	}
	return n
}

// compressOrig is the "simple" compress function from LT paper
func compressOrig(v ID, ancestor, semi, label []ID) {
	if ancestor[ancestor[v]] != 0 {
		compressOrig(ancestor[v], ancestor, semi, label)
		if semi[label[ancestor[v]]] < semi[label[v]] {
			label[v] = label[ancestor[v]]
		}
		ancestor[v] = ancestor[ancestor[v]]
	}
}

// evalOrig is the "simple" eval function from LT paper
func evalOrig(v ID, ancestor, semi, label []ID) ID {
	if ancestor[v] == 0 {
		return v
	}
	compressOrig(v, ancestor, semi, label)
	return label[v]
}

func linkOrig(v, w ID, ancestor []ID) {
	ancestor[w] = v
}

// dominators computes the dominator tree for f. It returns a slice
// which maps block ID to the immediate dominator of that block.
// Unreachable blocks map to nil. The entry block maps to nil.
func dominatorsSimple(f *Func) []*Block {
	// A simple algorithm for now
	// Cooper, Harvey, Kennedy
	idom := make([]*Block, f.NumBlocks())

	// Compute postorder walk
	post := f.postorder()

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
			for _, e := range b.Preds {
				p := e.b
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
