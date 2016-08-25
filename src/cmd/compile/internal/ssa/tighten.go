// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// tighten moves Values closer to the Blocks in which they are used.
// This can reduce the amount of register spilling required,
// if it doesn't also create more live values.
// For now, it handles only the trivial case in which a
// Value with one or fewer args is only used in a single Block,
// and not in a phi value.
// TODO: Do something smarter.
// A Value can be moved to any block that
// dominates all blocks in which it is used.
// Figure out when that will be an improvement.
func tighten(f *Func) {
	// For each value, the number of blocks in which it is used.
	uses := make([]int32, f.NumValues())

	// For each value, whether that value is ever an arg to a phi value.
	phi := make([]bool, f.NumValues())

	// For each value, one block in which that value is used.
	home := make([]*Block, f.NumValues())

	changed := true
	for changed {
		changed = false

		// Reset uses
		for i := range uses {
			uses[i] = 0
		}
		// No need to reset home; any relevant values will be written anew anyway.
		// No need to reset phi; once used in a phi, always used in a phi.

		for _, b := range f.Blocks {
			for _, v := range b.Values {
				for _, w := range v.Args {
					if v.Op == OpPhi {
						phi[w.ID] = true
					}
					uses[w.ID]++
					home[w.ID] = b
				}
			}
			if b.Control != nil {
				uses[b.Control.ID]++
				home[b.Control.ID] = b
			}
		}

		for _, b := range f.Blocks {
			for i := 0; i < len(b.Values); i++ {
				v := b.Values[i]
				if v.Op == OpPhi || v.Op == OpGetClosurePtr || v.Op == OpConvert || v.Op == OpArg {
					// GetClosurePtr & Arg must stay in entry block.
					// OpConvert must not float over call sites.
					// TODO do we instead need a dependence edge of some sort for OpConvert?
					// Would memory do the trick, or do we need something else that relates
					// to safe point operations?
					continue
				}
				if len(v.Args) > 0 && v.Args[len(v.Args)-1].Type.IsMemory() {
					// We can't move values which have a memory arg - it might
					// make two memory values live across a block boundary.
					continue
				}
				if uses[v.ID] == 1 && !phi[v.ID] && home[v.ID] != b && len(v.Args) < 2 {
					// v is used in exactly one block, and it is not b.
					// Furthermore, it takes at most one input,
					// so moving it will not increase the
					// number of live values anywhere.
					// Move v to that block.
					c := home[v.ID]
					c.Values = append(c.Values, v)
					v.Block = c
					last := len(b.Values) - 1
					b.Values[i] = b.Values[last]
					b.Values[last] = nil
					b.Values = b.Values[:last]
					changed = true
				}
			}
		}
	}
}

// phiTighten moves constants closer to phi users.
// This pass avoids having lots of constants live for lots of the program.
// See issue 16407.
func phiTighten(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			for i, a := range v.Args {
				if !a.rematerializeable() {
					continue // not a constant we can move around
				}
				if a.Block == b.Preds[i].b {
					continue // already in the right place
				}
				// Make a copy of a, put in predecessor block.
				v.SetArg(i, a.copyInto(b.Preds[i].b))
			}
		}
	}
}
