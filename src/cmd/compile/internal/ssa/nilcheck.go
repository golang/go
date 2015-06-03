// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// nilcheckelim eliminates unnecessary nil checks.
func nilcheckelim(f *Func) {
	// Exit early if there are no nil checks to eliminate.
	var found bool
	for _, b := range f.Blocks {
		if checkedptr(b) != nil {
			found = true
			break
		}
	}
	if !found {
		return
	}

	// Eliminate redundant nil checks.
	// A nil check is redundant if the same
	// nil check has been performed by a
	// dominating block.
	// The efficacy of this pass depends
	// heavily on the efficacy of the cse pass.
	idom := dominators(f) // TODO: cache the dominator tree in the function, clearing when the CFG changes?
	for _, b := range f.Blocks {
		ptr := checkedptr(b)
		if ptr == nil {
			continue
		}
		var elim bool
		// Walk up the dominator tree,
		// looking for identical nil checks.
		for c := idom[b.ID]; c != nil; c = idom[c.ID] {
			if checkedptr(c) == ptr {
				elim = true
				break
			}
		}
		if elim {
			// Eliminate the nil check.
			// The deadcode pass will remove vestigial values,
			// and the fuse pass will join this block with its successor.
			b.Kind = BlockPlain
			b.Control = nil
			removePredecessor(b, b.Succs[1])
			b.Succs = b.Succs[:1]
		}
	}

	// TODO: Eliminate more nil checks.
	// For example, pointers to function arguments
	// and pointers to static values cannot be nil.
	// We could also track pointers constructed by
	// taking the address of another value.
	// We can also recursively remove any chain of
	// fixed offset calculations,
	// i.e. struct fields and array elements,
	// even with non-constant indices:
	// x is non-nil iff x.a.b[i].c is.
}

// checkedptr returns the Value, if any,
// that is used in a nil check in b's Control op.
func checkedptr(b *Block) *Value {
	if b.Kind == BlockIf && b.Control.Op == OpIsNonNil {
		return b.Control.Args[0]
	}
	return nil
}
