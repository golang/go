// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// layout orders basic blocks in f with the goal of minimizing control flow instructions.
// After this phase returns, the order of f.Blocks matters and is the order
// in which those blocks will appear in the assembly output.
func layout(f *Func) {
	f.Blocks = layoutOrder(f)
}

// Register allocation may use a different order which has constraints
// imposed by the linear-scan algorithm. Note that f.pass here is
// regalloc, so the switch is conditional on -d=ssa/regalloc/test=N
func layoutRegallocOrder(f *Func) []*Block {

	switch f.pass.test {
	case 0: // layout order
		return layoutOrder(f)
	case 1: // existing block order
		return f.Blocks
	case 2: // reverse of postorder; legal, but usually not good.
		po := f.postorder()
		visitOrder := make([]*Block, len(po))
		for i, b := range po {
			j := len(po) - i - 1
			visitOrder[j] = b
		}
		return visitOrder
	}

	return nil
}

func layoutOrder(f *Func) []*Block {
	order := make([]*Block, 0, f.NumBlocks())
	scheduled := make([]bool, f.NumBlocks())
	idToBlock := make([]*Block, f.NumBlocks())
	indegree := make([]int, f.NumBlocks())
	posdegree := f.newSparseSet(f.NumBlocks()) // blocks with positive remaining degree
	defer f.retSparseSet(posdegree)
	zerodegree := f.newSparseSet(f.NumBlocks()) // blocks with zero remaining degree
	defer f.retSparseSet(zerodegree)
	exit := f.newSparseSet(f.NumBlocks()) // exit blocks
	defer f.retSparseSet(exit)

	// Initialize indegree of each block
	for _, b := range f.Blocks {
		idToBlock[b.ID] = b
		if b.Kind == BlockExit {
			// exit blocks are always scheduled last
			// TODO: also add blocks post-dominated by exit blocks
			exit.add(b.ID)
			continue
		}
		indegree[b.ID] = len(b.Preds)
		if len(b.Preds) == 0 {
			zerodegree.add(b.ID)
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

		for _, e := range b.Succs {
			c := e.b
			indegree[c.ID]--
			if indegree[c.ID] == 0 {
				posdegree.remove(c.ID)
				zerodegree.add(c.ID)
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
		mindegree := f.NumBlocks()
		for _, e := range order[len(order)-1].Succs {
			c := e.b
			if scheduled[c.ID] || c.Kind == BlockExit {
				continue
			}
			if indegree[c.ID] < mindegree {
				mindegree = indegree[c.ID]
				bid = c.ID
			}
		}
		if bid != 0 {
			continue
		}
		// TODO: improve this part
		// No successor of the previously scheduled block works.
		// Pick a zero-degree block if we can.
		for zerodegree.size() > 0 {
			cid := zerodegree.pop()
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
