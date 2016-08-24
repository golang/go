// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// We are looking for loops with following structure
// (loop bodies may have control flow inside):
//
//              +--------------+
//              |              |
//              |  preheader   |
//              |              |
//              +-------+------+
//                      |
//                      |
//              +-------v------+
//              |              |
//       +------>    header    |
//       |      |              |
//       |      +-------+------+
//       |              |
//       |              |
//       |      +-------v------+
//       |      |              |
//       +------+  loop body   |
//              |              |
//              +--------------+
//
//
// We consider all phis and memory operations as initial loop dependent set.
// So loop independent values are all loop values,
// minus transitive closure of initial loop dependent values.
// We remove those values from their BBs and move them to preheader.

func licm(f *Func) {
	// See likelyadjust.go for details about loop info.
	nest := loopnestfor(f)
	if len(nest.loops) == 0 || nest.hasIrreducible {
		return
	}

	uses := uses(f)
	defer uses.free(f)

	loopDependent := f.Cache.allocBoolSlice(f.NumValues())
	defer f.Cache.freeBoolSlice(loopDependent)
	queue := f.Cache.allocValueSlice(f.NumValues())
	defer f.Cache.freeValueSlice(queue)
	queue = queue[:0]

	// Start with all values we can't move out of loops.
	for _, b := range f.Blocks {
		if loop := nest.b2l[b.ID]; loop == nil || !loop.isInner {
			// Values outside any loop we don't care about.
			// Values not in a leaf loop we can't handle.
			continue
		}
		for _, v := range b.Values {
			loopDep := false
			if v.Op == OpPhi {
				loopDep = true
			} else if v.Type.IsMemory() {
				// We can't move state-modifying code.
				// (TODO: but maybe this is handled by memory Phis anyway?)
				loopDep = true
			} else if v.Type.IsFlags() || v.Type.IsTuple() && (v.Type.FieldType(0).IsFlags() || v.Type.FieldType(1).IsFlags()) {
				// This is not required for correctness. It is just to
				// keep the live range of flag values low.
				loopDep = true
			} else if opcodeTable[v.Op].nilCheck {
				// NilCheck in case loop executes 0 times. (It has a memory arg anyway?)
				loopDep = true
			} else if v.MemoryArg() != nil {
				// Because the state of memory might be different at the loop start. (Also handled by Phi?)
				loopDep = true
			} else if v.Type.IsPtr() {
				// Can't move pointer arithmetic, as it may be guarded by conditionals
				// and this could materialize a bad pointer across a safepoint.
				loopDep = true
			}
			if loopDep {
				loopDependent[v.ID] = true
				queue = append(queue, v)
			}
		}
	}

	// If a value can't be moved out of a loop, neither can its users.
	// The queue contains values which are loop dependent, but their users
	// have not been marked as loop dependent yet.
	for len(queue) > 0 {
		v := queue[len(queue)-1]
		queue = queue[:len(queue)-1]

		for _, u := range uses.get(v) {
			if loop := nest.b2l[u.Block.ID]; loop == nil || !loop.isInner {
				continue // see above
			}
			if loopDependent[u.ID] {
				continue
			}
			loopDependent[u.ID] = true
			queue = append(queue, u)
		}
	}

	// Anything not marked as loop-dependent can be moved out of its loop.
	for _, b := range f.Blocks {
		loop := nest.b2l[b.ID]
		if loop == nil || !loop.isInner {
			// loopDependent check is wrong for loops containing other loops,
			// because then a value might have an argument computed inside
			// a nested loop.
			continue
		}
		if len(loop.header.Preds) != 2 {
			continue // is never true?
		}
		anyMoved := false
		for i, v := range b.Values {
			if loopDependent[v.ID] {
				continue
			}
			// Figure out where to move loop-independent values.
			h := loop.header
			var inIdx int
			if int(h.Preds[0].b.ID) >= len(nest.b2l) || nest.b2l[h.Preds[0].b.ID] != loop {
				inIdx = 0
			} else {
				inIdx = 1
			}
			dest := h.Preds[inIdx].b
			if dest.Kind != BlockPlain {
				outIdx := h.Preds[inIdx].i
				// Introduce a new block between the loop
				// header predecessor and the loop header itself.
				mid := f.NewBlock(BlockPlain)
				mid.Pos = dest.Pos
				// Splice into graph.
				mid.Preds = append(mid.Preds, Edge{dest, outIdx})
				mid.Succs = append(mid.Succs, Edge{h, inIdx})
				h.Preds[inIdx] = Edge{mid, 0}
				dest.Succs[outIdx] = Edge{mid, 0}

				dest = mid
			}

			b.Values[i] = nil
			v.Block = dest
			dest.Values = append(dest.Values, v)
			anyMoved = true
		}
		if anyMoved {
			// We just nil'd entries in b.Values above. Compact out the nils.
			i := 0
			for _, v := range b.Values {
				if v == nil {
					continue
				}
				b.Values[i] = v
				i++
			}
			b.Values = b.Values[:i]
		}
	}
}
