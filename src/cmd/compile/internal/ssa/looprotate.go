// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// loopRotate converts loops with a check-loop-condition-at-beginning
// to loops with a check-loop-condition-at-end.
// This helps loops avoid extra unnecessary jumps.
//
//   loop:
//     CMPQ ...
//     JGE exit
//     ...
//     JMP loop
//   exit:
//
//    JMP entry
//  loop:
//    ...
//  entry:
//    CMPQ ...
//    JLT loop
func loopRotate(f *Func) {
	loopnest := f.loopnest()
	if len(loopnest.loops) == 0 {
		return
	}

	// Set of blocks we're moving, by ID.
	move := map[ID]struct{}{}

	// Map from block ID to the moving block that should
	// come right after it.
	after := map[ID]*Block{}

	// Check each loop header and decide if we want to move it.
	for _, loop := range loopnest.loops {
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
		if p == nil || p == b {
			continue
		}

		// Place b after p.
		move[b.ID] = struct{}{}
		after[p.ID] = b
	}

	// Move blocks to their destinations in a single pass.
	// We rely here on the fact that loop headers must come
	// before the rest of the loop.  And that relies on the
	// fact that we only identify reducible loops.
	j := 0
	for i, b := range f.Blocks {
		if _, ok := move[b.ID]; ok {
			continue
		}
		f.Blocks[j] = b
		j++
		if a := after[b.ID]; a != nil {
			if j > i {
				f.Fatalf("head before tail in loop %s", b)
			}
			f.Blocks[j] = a
			j++
		}
	}
	if j != len(f.Blocks) {
		f.Fatalf("bad reordering in looprotate")
	}
}
