// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// checkFunc checks invariants of f.
func checkFunc(f *Func) {
	blockMark := make([]bool, f.NumBlocks())
	valueMark := make([]bool, f.NumValues())

	for _, b := range f.Blocks {
		if blockMark[b.ID] {
			f.Fatal("block %s appears twice in %s!", b, f.Name)
		}
		blockMark[b.ID] = true
		if b.Func != f {
			f.Fatal("%s.Func=%s, want %s", b, b.Func.Name, f.Name)
		}

		for i, c := range b.Succs {
			for j, d := range b.Succs {
				if i != j && c == d {
					f.Fatal("%s.Succs has duplicate block %s", b, c)
				}
			}
		}
		// Note: duplicate successors are hard in the following case:
		//      if(...) goto x else goto x
		//   x: v = phi(a, b)
		// If the conditional is true, does v get the value of a or b?
		// We could solve this other ways, but the easiest is just to
		// require (by possibly adding empty control-flow blocks) that
		// all successors are distinct.  They will need to be distinct
		// anyway for register allocation (duplicate successors implies
		// the existence of critical edges).

		for _, p := range b.Preds {
			var found bool
			for _, c := range p.Succs {
				if c == b {
					found = true
					break
				}
			}
			if !found {
				f.Fatal("block %s is not a succ of its pred block %s", b, p)
			}
		}

		switch b.Kind {
		case BlockExit:
			if len(b.Succs) != 0 {
				f.Fatal("exit block %s has successors", b)
			}
			if b.Control == nil {
				f.Fatal("exit block %s has no control value", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatal("exit block %s has non-memory control value %s", b, b.Control.LongString())
			}
		case BlockPlain:
			if len(b.Succs) != 1 {
				f.Fatal("plain block %s len(Succs)==%d, want 1", b, len(b.Succs))
			}
			if b.Control != nil {
				f.Fatal("plain block %s has non-nil control %s", b, b.Control.LongString())
			}
		case BlockIf:
			if len(b.Succs) != 2 {
				f.Fatal("if block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatal("if block %s has no control value", b)
			}
			if !b.Control.Type.IsBoolean() {
				f.Fatal("if block %s has non-bool control value %s", b, b.Control.LongString())
			}
		case BlockCall:
			if len(b.Succs) != 2 {
				f.Fatal("call block %s len(Succs)==%d, want 2", b, len(b.Succs))
			}
			if b.Control == nil {
				f.Fatal("call block %s has no control value", b)
			}
			if !b.Control.Type.IsMemory() {
				f.Fatal("call block %s has non-memory control value %s", b, b.Control.LongString())
			}
			if b.Succs[1].Kind != BlockExit {
				f.Fatal("exception edge from call block %s does not go to exit but %s", b, b.Succs[1])
			}
		}

		for _, v := range b.Values {
			if valueMark[v.ID] {
				f.Fatal("value %s appears twice!", v.LongString())
			}
			valueMark[v.ID] = true

			if v.Block != b {
				f.Fatal("%s.block != %s", v, b)
			}
			if v.Op == OpPhi && len(v.Args) != len(b.Preds) {
				f.Fatal("phi length %s does not match pred length %d for block %s", v.LongString(), len(b.Preds), b)
			}

			// TODO: check for cycles in values
			// TODO: check type
		}
	}

	for _, id := range f.bid.free {
		if blockMark[id] {
			f.Fatal("used block b%d in free list", id)
		}
	}
	for _, id := range f.vid.free {
		if valueMark[id] {
			f.Fatal("used value v%d in free list", id)
		}
	}
}
