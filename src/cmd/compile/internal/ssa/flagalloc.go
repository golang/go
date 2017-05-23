// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

const flagRegMask = regMask(1) << 33 // TODO: arch-specific

// flagalloc allocates the flag register among all the flag-generating
// instructions. Flag values are recomputed if they need to be
// spilled/restored.
func flagalloc(f *Func) {
	// Compute the in-register flag value we want at the end of
	// each block. This is basically a best-effort live variable
	// analysis, so it can be much simpler than a full analysis.
	// TODO: do we really need to keep flag values live across blocks?
	// Could we force the flags register to be unused at basic block
	// boundaries?  Then we wouldn't need this computation.
	end := make([]*Value, f.NumBlocks())
	for n := 0; n < 2; n++ {
		// Walk blocks backwards. Poor-man's postorder traversal.
		for i := len(f.Blocks) - 1; i >= 0; i-- {
			b := f.Blocks[i]
			// Walk values backwards to figure out what flag
			// value we want in the flag register at the start
			// of the block.
			flag := end[b.ID]
			if b.Control != nil && b.Control.Type.IsFlags() {
				flag = b.Control
			}
			for j := len(b.Values) - 1; j >= 0; j-- {
				v := b.Values[j]
				if v == flag {
					flag = nil
				}
				if opcodeTable[v.Op].reg.clobbers&flagRegMask != 0 {
					flag = nil
				}
				for _, a := range v.Args {
					if a.Type.IsFlags() {
						flag = a
					}
				}
			}
			if flag != nil {
				for _, e := range b.Preds {
					p := e.b
					end[p.ID] = flag
				}
			}
		}
	}

	// For blocks which have a flags control value, that's the only value
	// we can leave in the flags register at the end of the block. (There
	// is no place to put a flag regeneration instruction.)
	for _, b := range f.Blocks {
		v := b.Control
		if v != nil && v.Type.IsFlags() && end[b.ID] != v {
			end[b.ID] = nil
		}
		if b.Kind == BlockDefer {
			// Defer blocks internally use/clobber the flags value.
			end[b.ID] = nil
		}
	}

	// Add flag recomputations where they are needed.
	// TODO: Remove original instructions if they are never used.
	var oldSched []*Value
	for _, b := range f.Blocks {
		oldSched = append(oldSched[:0], b.Values...)
		b.Values = b.Values[:0]
		// The current live flag value the pre-flagalloc copy).
		var flag *Value
		if len(b.Preds) > 0 {
			flag = end[b.Preds[0].b.ID]
			// Note: the following condition depends on the lack of critical edges.
			for _, e := range b.Preds[1:] {
				p := e.b
				if end[p.ID] != flag {
					f.Fatalf("live flag in %s's predecessors not consistent", b)
				}
			}
		}
		for _, v := range oldSched {
			if v.Op == OpPhi && v.Type.IsFlags() {
				f.Fatalf("phi of flags not supported: %s", v.LongString())
			}
			// Make sure any flag arg of v is in the flags register.
			// If not, recompute it.
			for i, a := range v.Args {
				if !a.Type.IsFlags() {
					continue
				}
				if a == flag {
					continue
				}
				// Recalculate a
				c := a.copyInto(b)
				// Update v.
				v.SetArg(i, c)
				// Remember the most-recently computed flag value.
				flag = a
			}
			// Issue v.
			b.Values = append(b.Values, v)
			if opcodeTable[v.Op].reg.clobbers&flagRegMask != 0 {
				flag = nil
			}
			if v.Type.IsFlags() {
				flag = v
			}
		}
		if v := b.Control; v != nil && v != flag && v.Type.IsFlags() {
			// Recalculate control value.
			c := v.copyInto(b)
			b.SetControl(c)
			flag = v
		}
		if v := end[b.ID]; v != nil && v != flag {
			// Need to reissue flag generator for use by
			// subsequent blocks.
			_ = v.copyInto(b)
			// Note: this flag generator is not properly linked up
			// with the flag users. This breaks the SSA representation.
			// We could fix up the users with another pass, but for now
			// we'll just leave it.  (Regalloc has the same issue for
			// standard regs, and it runs next.)
		}
	}

	// Save live flag state for later.
	for _, b := range f.Blocks {
		b.FlagsLiveAtEnd = end[b.ID] != nil
	}
}
