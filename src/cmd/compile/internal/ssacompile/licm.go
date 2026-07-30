// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

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

func licm(f *ssacore.Func) {
	// See likelyadjust.go for details about loop info.
	nest := ssacore.Loopnestfor(f)
	if len(nest.Loops) == 0 || nest.HasIrreducible {
		return
	}

	uses := uses(f)
	defer uses.free(f)

	loopDependent := f.Cache.AllocBoolSlice(f.NumValues())
	defer f.Cache.FreeBoolSlice(loopDependent)
	queue := f.Cache.AllocValueSlice(f.NumValues())
	defer f.Cache.FreeValueSlice(queue)
	queue = queue[:0]

	// Start with all values we can't move out of loops.
	for _, b := range f.Blocks {
		if loop := nest.B2L[b.ID]; loop == nil || !loop.IsInner {
			// Values outside any loop we don't care about.
			// Values not in a leaf loop we can't handle.
			continue
		}
		for _, v := range b.Values {
			if ssaop.OpcodeTable[v.Op].EarlyOk {
				// Double check we didn't mark the wrong ops as earlyOk
				if v.Type.IsMemory() || ssaop.OpcodeTable[v.Op].NilCheck || ssaop.OpcodeTable[v.Op].HasSideEffects || v.MemoryArg() != nil {
					v.Fatalf("op %s has bad earlyOk mark", v.Op)
				}
				if !v.Type.IsPtr() {
					// Note: can't move pointer arithmetic, as it may be guarded by conditionals
					// and thus could materialize a bad pointer across a safepoint.

					continue // Ok to lift out of loop.
				}
			}
			if v.Op == ssaop.OpSelect0 || v.Op == ssaop.OpSelect1 {
				// These ops can (and must) move with the op they are selecting from.
				continue
			}
			loopDependent[v.ID] = true
			queue = append(queue, v)
		}
	}

	// If a value can't be moved out of a loop, neither can its users.
	// The queue contains values which are loop dependent, but their users
	// have not been marked as loop dependent yet.
	for len(queue) > 0 {
		v := queue[len(queue)-1]
		queue = queue[:len(queue)-1]

		for _, u := range uses.get(v) {
			if loop := nest.B2L[u.Block.ID]; loop == nil || !loop.IsInner {
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
		loop := nest.B2L[b.ID]
		if loop == nil || !loop.IsInner {
			// loopDependent check is wrong for loops containing other loops,
			// because then a value might have an argument computed inside
			// a nested loop.
			continue
		}
		if len(loop.Header.Preds) != 2 {
			continue // is never true?
		}
		anyMoved := false
		for i, v := range b.Values {
			if loopDependent[v.ID] {
				continue
			}
			// Figure out where to move loop-independent values.
			h := loop.Header
			var inIdx int
			if int(h.Preds[0].B.ID) >= len(nest.B2L) || nest.B2L[h.Preds[0].B.ID] != loop {
				inIdx = 0
			} else {
				inIdx = 1
			}
			dest := h.Preds[inIdx].B
			if dest.Kind != block.BlockPlain {
				outIdx := h.Preds[inIdx].I
				// Introduce a new block between the loop
				// header predecessor and the loop header itself.
				mid := f.NewBlock(block.BlockPlain)
				mid.Pos = dest.Pos
				// Splice into graph.
				mid.Preds = append(mid.Preds, ssacore.Edge{B: dest, I: outIdx})
				mid.Succs = append(mid.Succs, ssacore.Edge{B: h, I: inIdx})
				h.Preds[inIdx] = ssacore.Edge{B: mid, I: 0}
				dest.Succs[outIdx] = ssacore.Edge{B: mid, I: 0}

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
