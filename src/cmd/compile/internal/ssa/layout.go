// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

// layout orders basic blocks in f with the goal of minimizing control flow instructions.
// After this phase returns, the order of f.Blocks matters and is the order
// in which those blocks will appear in the assembly output.
func layout(f *Func) {
	f.Blocks = greedyBlockOrder(f)
}

// Register allocation may use a different order which has constraints
// imposed by the linear-scan algorithm.
func layoutRegallocOrder(f *Func) []*Block {
	// remnant of an experiment; perhaps there will be another.
	return layoutOrder(f)
}

func layoutOrder(f *Func) []*Block {
	order := make([]*Block, 0, f.NumBlocks())
	scheduled := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(scheduled)
	idToBlock := f.Cache.allocBlockSlice(f.NumBlocks())
	defer f.Cache.freeBlockSlice(idToBlock)
	indegree := f.Cache.allocIntSlice(f.NumBlocks())
	defer f.Cache.freeIntSlice(indegree)
	posdegree := f.newSparseSet(f.NumBlocks()) // blocks with positive remaining degree
	defer f.retSparseSet(posdegree)
	// blocks with zero remaining degree. Use slice to simulate a LIFO queue to implement
	// the depth-first topology sorting algorithm.
	var zerodegree []ID
	// LIFO queue. Track the successor blocks of the scheduled block so that when we
	// encounter loops, we choose to schedule the successor block of the most recently
	// scheduled block.
	var succs []ID
	exit := f.newSparseSet(f.NumBlocks()) // exit blocks
	defer f.retSparseSet(exit)

	// Populate idToBlock and find exit blocks.
	for _, b := range f.Blocks {
		idToBlock[b.ID] = b
		if b.Kind == BlockExit {
			exit.add(b.ID)
		}
	}

	// Expand exit to include blocks post-dominated by exit blocks.
	for {
		changed := false
		for _, id := range exit.contents() {
			b := idToBlock[id]
		NextPred:
			for _, pe := range b.Preds {
				p := pe.b
				if exit.contains(p.ID) {
					continue
				}
				for _, s := range p.Succs {
					if !exit.contains(s.b.ID) {
						continue NextPred
					}
				}
				// All Succs are in exit; add p.
				exit.add(p.ID)
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// Initialize indegree of each block
	for _, b := range f.Blocks {
		if exit.contains(b.ID) {
			// exit blocks are always scheduled last
			continue
		}
		indegree[b.ID] = len(b.Preds)
		if len(b.Preds) == 0 {
			// Push an element to the tail of the queue.
			zerodegree = append(zerodegree, b.ID)
		} else {
			posdegree.add(b.ID)
		}
	}

	bid := f.Entry.ID
blockloop:
	for {
		// add block to schedule
		b := idToBlock[bid]
		order = append(order, b)
		scheduled[bid] = true
		if len(order) == len(f.Blocks) {
			break
		}

		// Here, the order of traversing the b.Succs affects the direction in which the topological
		// sort advances in depth. Take the following cfg as an example, regardless of other factors.
		//           b1
		//         0/ \1
		//        b2   b3
		// Traverse b.Succs in order, the right child node b3 will be scheduled immediately after
		// b1, traverse b.Succs in reverse order, the left child node b2 will be scheduled
		// immediately after b1. The test results show that reverse traversal performs a little
		// better.
		// Note: You need to consider both layout and register allocation when testing performance.
		for i := len(b.Succs) - 1; i >= 0; i-- {
			c := b.Succs[i].b
			indegree[c.ID]--
			if indegree[c.ID] == 0 {
				posdegree.remove(c.ID)
				zerodegree = append(zerodegree, c.ID)
			} else {
				succs = append(succs, c.ID)
			}
		}

		// Pick the next block to schedule
		// Pick among the successor blocks that have not been scheduled yet.

		// Use likely direction if we have it.
		var likely *Block
		switch b.Likely {
		case BranchLikely:
			likely = b.Succs[0].b
		case BranchUnlikely:
			likely = b.Succs[1].b
		}
		if likely != nil && !scheduled[likely.ID] {
			bid = likely.ID
			continue
		}

		// Use degree for now.
		bid = 0
		// TODO: improve this part
		// No successor of the previously scheduled block works.
		// Pick a zero-degree block if we can.
		for len(zerodegree) > 0 {
			// Pop an element from the tail of the queue.
			cid := zerodegree[len(zerodegree)-1]
			zerodegree = zerodegree[:len(zerodegree)-1]
			if !scheduled[cid] {
				bid = cid
				continue blockloop
			}
		}

		// Still nothing, pick the unscheduled successor block encountered most recently.
		for len(succs) > 0 {
			// Pop an element from the tail of the queue.
			cid := succs[len(succs)-1]
			succs = succs[:len(succs)-1]
			if !scheduled[cid] {
				bid = cid
				continue blockloop
			}
		}

		// Still nothing, pick any non-exit block.
		for posdegree.size() > 0 {
			cid := posdegree.pop()
			if !scheduled[cid] {
				bid = cid
				continue blockloop
			}
		}
		// Pick any exit block.
		// TODO: Order these to minimize jump distances?
		for {
			cid := exit.pop()
			if !scheduled[cid] {
				bid = cid
				continue blockloop
			}
		}
	}
	f.laidout = true
	return order
	//f.Blocks = order
}

// ----------------------------------------------------------------------------
// Greedy Basic Block Layout
//
// This is an adaptation of Pettis & Hansen's greedy algorithm for laying out
// basic blocks. See Profile Guided Code Positioning by Pettis & Hansen. The idea
// is to arrange hot blocks near each other. Initially all blocks are belongs to
// its own chain, then starting from hottest edge and repeatedly merge two proper
// chains iff the edge dest is the first block of dest chain and edge src is the
// last block of src chain. Once all edges are processed, the chains are sorted
// by hottness and merge count and generate final block order.

// chain is a linear sequence of blocks, where the first block is the entry block
// and the last block is the exit block. The chain is used to represent a sequence
// of blocks that are likely to be executed together.
type chain struct {
	id       ID
	blocks   []*Block // ordered blocks in this chain
	priority int      // merge count
}

func (c *chain) first() *Block {
	return c.blocks[0]
}

func (c *chain) last() *Block {
	return c.blocks[len(c.blocks)-1]
}

// edge simply represents a CFG edge
type edge struct {
	src    *Block
	dst    *Block
	weight int // frequency
}

const (
	WeightTaken    = 100
	WeightNotTaken = 0
)

func (e *edge) String() string {
	return fmt.Sprintf("%v->%v(%d)", e.src, e.dst, e.weight)
}

// chainGraph is a directed graph of chains, where each chain is a sequence of
// blocks, and each edge is a CFG edge with a weight. The graph is used to build
// a block layout that minimizes control flow instructions.
type chainGraph struct {
	cid     idAlloc
	chains  []*chain
	edges   []*edge
	b2chain []*chain // indexed by block id
}

func (g *chainGraph) newChain(block *Block) *chain {
	c := &chain{g.cid.get(), []*Block{block}, 0 /*priority*/}
	g.b2chain[block.ID] = c
	g.chains = append(g.chains, c)
	return c
}

func (g *chainGraph) getChain(b *Block) *chain {
	return g.b2chain[b.ID]
}

// mergeChain merges the "from" chain into the "to" chain. The from chain is
// removed then.
func (g *chainGraph) mergeChain(to, from *chain) {
	for _, block := range from.blocks {
		g.b2chain[block.ID] = to
	}
	to.blocks = append(to.blocks, from.blocks...)
	to.priority++ // increment
	g.chains[from.id-1 /*ID always >0*/] = nil
}

func (g *chainGraph) print() {
	fmt.Printf("== Edges:\n")
	for _, edge := range g.edges {
		fmt.Printf("%v\n", edge)
	}
	fmt.Printf("== Chains:\n")
	for _, ch := range g.chains {
		if ch == nil {
			continue
		}
		fmt.Printf("id:%d priority:%d blocks:%v\n", ch.id, ch.priority, ch.blocks)
	}
}

func greedyBlockOrder(fn *Func) []*Block {
	graph := &chainGraph{
		chains:  []*chain{},
		edges:   []*edge{},
		b2chain: make([]*chain, fn.NumBlocks(), fn.NumBlocks()),
	}

	// Initially every block is in its own chain
	for _, block := range fn.Blocks {
		graph.newChain(block)

		if len(block.Succs) == 1 {
			graph.edges = append(graph.edges, &edge{block, block.Succs[0].b, WeightTaken})
		} else if len(block.Succs) == 2 && block.Likely != BranchUnknown {
			// Static branch prediction is available
			taken := 0
			if block.Likely == BranchUnlikely {
				taken = 1
			}
			e1 := &edge{block, block.Succs[taken].b, WeightTaken}
			e2 := &edge{block, block.Succs[1-taken].b, WeightNotTaken}
			graph.edges = append(graph.edges, e1, e2)
		} else {
			// Block predication is unknown or there are more than 2 successors
			for _, succ := range block.Succs {
				e1 := &edge{block, succ.b, WeightTaken}
				graph.edges = append(graph.edges, e1)
			}
		}
	}

	// Sort edges by weight and move slow path to end
	sort.SliceStable(graph.edges, func(i, j int) bool {
		e1, e2 := graph.edges[i], graph.edges[j]
		// Move slow path to end
		if e1.weight == WeightNotTaken && e2.weight == WeightNotTaken {
			return e1.dst.Kind != BlockExit && e2.dst.Kind == BlockExit
		}

		// If the weights are the same, then keep the original order, this
		// ensures that adjacent edges are accessed sequentially, which has
		// a noticeable impact on performance
		return e1.weight >= e2.weight
	})

	// Merge proper chains until no more chains can be merged
	for _, edge := range graph.edges {
		src := graph.getChain(edge.src)
		dst := graph.getChain(edge.dst)
		if src == dst {
			// Loop detected, "rotate" the loop from [..,header,body,latch] to
			// [..,body,latch,header]
			for idx, block := range src.blocks {
				if block == edge.dst && block.Kind != BlockPlain /*already rotated?*/ {
					c := append(src.blocks[0:idx], src.blocks[idx+1:]...)
					c = append(c, block)
					src.blocks = c
					break
				}
			}
			continue
		}
		if edge.dst == dst.first() && edge.src == src.last() {
			graph.mergeChain(src, dst)
		}
	}
	i := 0
	for _, chain := range graph.chains {
		// Remove nil chains because they are merge
		if chain != nil {
			graph.chains[i] = chain
			if chain.first() == fn.Entry {
				// Entry chain must be present at beginning
				graph.chains[0], graph.chains[i] = graph.chains[i], graph.chains[0]
			}
			i++
		}
	}
	graph.chains = graph.chains[:i]

	// Reorder chains based by hottness and priority
	before := make(map[*chain][]*chain)
	for _, edge := range graph.edges {
		// Compute the "before" precedence relation between chain, specifically,
		// the chain that is taken is arranged before the chain that is not taken.
		// This is because hardware prediction thought forward branch is less
		// frequently taken, while backedge is more frequently taken.
		if edge.weight == WeightNotTaken {
			src := graph.getChain(edge.src)
			dst := graph.getChain(edge.dst)
			before[src] = append(before[src], dst)
		}
	}
	sort.SliceStable(graph.chains, func(i, j int) bool {
		c1, c2 := graph.chains[i], graph.chains[j]
		// Entry chain must be present at beginning
		if c1.first() == fn.Entry {
			return true
		}
		if c2.first() == fn.Entry {
			return false
		}
		// Respect precedence relation
		for _, b := range before[c1] {
			if b == c2 {
				return true
			}
		}
		// Higher merge count is considered
		if c1.priority != c2.priority {
			return c1.priority > c2.priority
		}
		// Non-terminated chain is considered
		if s1, s2 := len(c1.last().Succs), len(c2.last().Succs); s1 != s2 {
			return s1 > s2
		}
		// Keep original order if we can't decide
		return false
	})
	assert(graph.chains[0].first() == fn.Entry, "entry chain must be first")

	// Generate final block order
	blockOrder := make([]*Block, 0)
	for _, chain := range graph.chains {
		blockOrder = append(blockOrder, chain.blocks...)
	}
	fn.laidout = true

	if fn.pass.debug > 2 {
		fmt.Printf("Block ordering(%v):\n", fn.Name)
		graph.print()
	}
	if len(blockOrder) != len(fn.Blocks) {
		graph.print()
		fn.Fatalf("miss blocks in final order")
	}
	if entryChain := graph.getChain(fn.Entry); entryChain != graph.chains[0] ||
		entryChain.first() != fn.Entry {
		graph.print()
		fn.Fatalf("entry block is not first block")
	}
	return blockOrder
}
