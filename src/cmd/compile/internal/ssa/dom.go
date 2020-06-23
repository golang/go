// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file contains code to compute the dominator tree
// of a control-flow graph.

// postorder computes a postorder traversal ordering for the
// basic blocks in f. Unreachable blocks will not appear.
func postorder(f *Func) []*Block {
	return postorderWithNumbering(f, nil)
}

type blockAndIndex struct {
	b     *Block
	index int // index is the number of successor edges of b that have already been explored.
}

// postorderWithNumbering provides a DFS postordering.
// This seems to make loop-finding more robust.
func postorderWithNumbering(f *Func, ponums []int32) []*Block {
	seen := make([]bool, f.NumBlocks())

	// result ordering
	order := make([]*Block, 0, len(f.Blocks))

	// stack of blocks and next child to visit
	// A constant bound allows this to be stack-allocated. 32 is
	// enough to cover almost every postorderWithNumbering call.
	s := make([]blockAndIndex, 0, 32)
	s = append(s, blockAndIndex{b: f.Entry})
	seen[f.Entry.ID] = true
	for len(s) > 0 {
		tos := len(s) - 1
		x := s[tos]
		b := x.b
		if i := x.index; i < len(b.Succs) {
			s[tos].index++
			bb := b.Succs[i].Block()
			if !seen[bb.ID] {
				seen[bb.ID] = true
				s = append(s, blockAndIndex{b: bb})
			}
			continue
		}
		s = s[:tos]
		if ponums != nil {
			ponums[b.ID] = int32(len(order))
		}
		order = append(order, b)
	}
	return order
}

type linkedBlocks func(*Block) []Edge

const nscratchslices = 7

// experimentally, functions with 512 or fewer blocks account
// for 75% of memory (size) allocation for dominator computation
// in make.bash.
const minscratchblocks = 512

func (cache *Cache) scratchBlocksForDom(maxBlockID int) (a, b, c, d, e, f, g []ID) {
	tot := maxBlockID * nscratchslices
	scratch := cache.domblockstore
	if len(scratch) < tot {
		// req = min(1.5*tot, nscratchslices*minscratchblocks)
		// 50% padding allows for graph growth in later phases.
		req := (tot * 3) >> 1
		if req < nscratchslices*minscratchblocks {
			req = nscratchslices * minscratchblocks
		}
		scratch = make([]ID, req)
		cache.domblockstore = scratch
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
	semi, vertex, label, parent, ancestor, bucketHead, bucketLink := f.Cache.scratchBlocksForDom(maxBlockID)

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
	// TODO: This loop is O(n^2). It used to be used in nilcheck,
	// see BenchmarkNilCheckDeep*.
	for b != c {
		if postnum[b.ID] < postnum[c.ID] {
			b = idom[b.ID]
		} else {
			c = idom[c.ID]
		}
	}
	return b
}
