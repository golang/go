// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"slices"
)

// loopRotate converts loops with a check-loop-condition-at-beginning
// to loops with a check-loop-condition-at-end.
// This helps loops avoid extra unnecessary jumps.
//
//	 loop:
//	   CMPQ ...
//	   JGE exit
//	   ...
//	   JMP loop
//	 exit:
//
//	  JMP entry
//	loop:
//	  ...
//	entry:
//	  CMPQ ...
//	  JLT loop
func loopRotate(f *Func) {
	loopnest := f.loopnest()
	if loopnest.hasIrreducible {
		return
	}
	if len(loopnest.loops) == 0 {
		return
	}

	idToIdx := f.Cache.allocIntSlice(f.NumBlocks())
	defer f.Cache.freeIntSlice(idToIdx)
	for i, b := range f.Blocks {
		idToIdx[b.ID] = i
	}

	// Set of blocks we're moving, by ID.
	move := map[ID]struct{}{}

	// Map from block ID to the moving blocks that should
	// come right after it.
	// If a block, which has its ID present in keys of the 'after' map,
	// occurs in some other block's 'after' list, that represents whole
	// nested loop, e.g. consider an inner loop I nested into an outer
	// loop O. It and Ot are corresponding top block for these loops
	// chosen by our algorithm, and It is in the Ot's 'after' list.
	//
	//    Before:                     After:
	//
	//       e                       e
	//       │                       │
	//       │                       │Ot ◄───┐
	//       ▼                       ▼▼      │
	//   ┌───Oh ◄────┐           ┌─┬─Oh      │
	//   │   │       │           │ │         │
	//   │   │       │           │ │ It◄───┐ │
	//   │   ▼       │           │ │ ▼     │ │
	//   │ ┌─Ih◄───┐ │           │ └►Ih    │ │
	//   │ │ │     │ │           │ ┌─┤     │ │
	//   │ │ ▼     │ │           │ │ ▼     │ │
	//   │ │ Ib    │ │           │ │ Ib    │ │
	//   │ │ └─►It─┘ │           │ │ └─────┘ │
	//   │ │         │           │ │         │
	//   │ └►Ie      │           │ └►Ie      │
	//   │   └─►Ot───┘           │   └───────┘
	//   │                       │
	//   └──►Oe                  └──►Oe
	//
	// We build the 'after' lists for each of the top blocks Ot and It:
	//   after[Ot]: Oh, It, Ie
	//   after[It]: Ih, Ib
	after := map[ID][]*Block{}

	// Map from loop header ID to the new top block for the loop.
	tops := map[ID]*Block{}

	// Order loops to rotate any child loop before adding its top block
	// to the parent loop's 'after' list.
	loopOrder := f.Cache.allocIntSlice(len(loopnest.loops))
	for i := range loopOrder {
		loopOrder[i] = i
	}
	defer f.Cache.freeIntSlice(loopOrder)
	slices.SortFunc(loopOrder, func(i, j int) int {
		di := loopnest.loops[i].depth
		dj := loopnest.loops[j].depth
		switch {
		case di > dj:
			return -1
		case di < dj:
			return 1
		default:
			return 0
		}
	})

	// Check each loop header and decide if we want to move it.
	for _, loopIdx := range loopOrder {
		loop := loopnest.loops[loopIdx]
		b := loop.header
		var p *Block // b's in-loop predecessor
		for _, e := range b.Preds {
			if e.b.Kind != BlockPlain {
				continue
			}
			if loopnest.b2l[e.b.ID] != loop {
				continue
			}
			p = e.b
		}
		if p == nil {
			continue
		}
		tops[loop.header.ID] = p
		p.Hotness |= HotInitial
		if f.IsPgoHot {
			p.Hotness |= HotPgo
		}
		// blocks will be arranged so that p is ordered first, if it isn't already.
		if p == b { // p is header, already first (and also, only block in the loop)
			continue
		}
		p.Hotness |= HotNotFlowIn

		// the loop header b follows p
		after[p.ID] = []*Block{b}
		for {
			nextIdx := idToIdx[b.ID] + 1
			if nextIdx >= len(f.Blocks) { // reached end of function (maybe impossible?)
				break
			}
			nextb := f.Blocks[nextIdx]
			if nextb == p { // original loop predecessor is next
				break
			}
			if bloop := loopnest.b2l[nextb.ID]; bloop != nil {
				if bloop == loop || bloop.outer == loop && tops[bloop.header.ID] == nextb {
					after[p.ID] = append(after[p.ID], nextb)
				}
			}
			b = nextb
		}
		// Swap b and p so that we'll handle p before b when moving blocks.
		f.Blocks[idToIdx[loop.header.ID]] = p
		f.Blocks[idToIdx[p.ID]] = loop.header
		idToIdx[loop.header.ID], idToIdx[p.ID] = idToIdx[p.ID], idToIdx[loop.header.ID]

		// Place loop blocks after p.
		for _, b := range after[p.ID] {
			move[b.ID] = struct{}{}
		}
	}

	// Move blocks to their destinations in a single pass.
	// We rely here on the fact that loop headers must come
	// before the rest of the loop.  And that relies on the
	// fact that we only identify reducible loops.
	j := 0
	// Some blocks that are not part of a loop may be placed
	// between loop blocks. In order to avoid these blocks from
	// being overwritten, use a temporary slice.
	oldOrder := f.Cache.allocBlockSlice(len(f.Blocks))
	defer f.Cache.freeBlockSlice(oldOrder)
	copy(oldOrder, f.Blocks)
	var moveBlocks func(bs []*Block)
	moveBlocks = func(blocks []*Block) {
		for _, a := range blocks {
			f.Blocks[j] = a
			j++
			if nextBlocks, ok := after[a.ID]; ok {
				moveBlocks(nextBlocks)
			}
		}
	}
	for _, b := range oldOrder {
		if _, ok := move[b.ID]; ok {
			continue
		}
		f.Blocks[j] = b
		j++
		moveBlocks(after[b.ID])
	}
	if j != len(oldOrder) {
		f.Fatalf("bad reordering in looprotate")
	}
}
