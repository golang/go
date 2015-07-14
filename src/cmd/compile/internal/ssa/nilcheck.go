// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// nilcheckelim eliminates unnecessary nil checks.
func nilcheckelim(f *Func) {
	// A nil check is redundant if the same nil check was successful in a
	// dominating block. The efficacy of this pass depends heavily on the
	// efficacy of the cse pass.
	idom := dominators(f)
	domTree := make([][]*Block, f.NumBlocks())

	// Create a block ID -> [dominees] mapping
	for _, b := range f.Blocks {
		if dom := idom[b.ID]; dom != nil {
			domTree[dom.ID] = append(domTree[dom.ID], b)
		}
	}

	// TODO: Eliminate more nil checks.
	// We can recursively remove any chain of fixed offset calculations,
	// i.e. struct fields and array elements, even with non-constant
	// indices: x is non-nil iff x.a.b[i].c is.

	type walkState int
	const (
		Work   walkState = iota // clear nil check if we should and traverse to dominees regardless
		RecPtr                  // record the pointer as being nil checked
		ClearPtr
	)

	type bp struct {
		block *Block // block, or nil in RecPtr/ClearPtr state
		ptr   *Value // if non-nil, ptr that is to be set/cleared in RecPtr/ClearPtr state
		op    walkState
	}

	work := make([]bp, 0, 256)
	work = append(work, bp{block: f.Entry, ptr: checkedptr(f.Entry)})

	// map from value ID to bool indicating if value is known to be non-nil
	// in the current dominator path being walked.  This slice is updated by
	// walkStates to maintain the known non-nil values.
	nonNilValues := make([]bool, f.NumValues())

	// perform a depth first walk of the dominee tree
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]

		var pushRecPtr bool
		switch node.op {
		case Work:
			if node.ptr != nil {
				// already have a nilcheck in the dominator path
				if nonNilValues[node.ptr.ID] {
					// Eliminate the nil check.
					// The deadcode pass will remove vestigial values,
					// and the fuse pass will join this block with its successor.
					node.block.Kind = BlockPlain
					node.block.Control = nil
					f.removePredecessor(node.block, node.block.Succs[1])
					node.block.Succs = node.block.Succs[:1]
				} else {
					// new nilcheck so add a ClearPtr node to clear the
					// ptr from the map of nil checks once we traverse
					// back up the tree
					work = append(work, bp{op: ClearPtr, ptr: node.ptr})
					// and cause a new setPtr to be appended after the
					// block's dominees
					pushRecPtr = true
				}
			}
		case RecPtr:
			nonNilValues[node.ptr.ID] = true
			continue
		case ClearPtr:
			nonNilValues[node.ptr.ID] = false
			continue
		}

		var nilBranch *Block
		for _, w := range domTree[node.block.ID] {
			// TODO: Since we handle the false side of OpIsNonNil
			// correctly, look into rewriting user nil checks into
			// OpIsNonNil so they can be eliminated also

			// we are about to traverse down the 'ptr is nil' side
			// of a nilcheck block, so save it for later
			if node.block.Kind == BlockIf && node.block.Control.Op == OpIsNonNil &&
				w == node.block.Succs[1] {
				nilBranch = w
				continue
			}
			work = append(work, bp{block: w, ptr: checkedptr(w)})
		}

		if nilBranch != nil {
			// we pop from the back of the work slice, so this sets
			// up the false branch to be operated on before the
			// node.ptr is recorded
			work = append(work, bp{op: RecPtr, ptr: node.ptr})
			work = append(work, bp{block: nilBranch, ptr: checkedptr(nilBranch)})
		} else if pushRecPtr {
			work = append(work, bp{op: RecPtr, ptr: node.ptr})
		}
	}
}

// nilcheckelim0 is the original redundant nilcheck elimination algorithm.
func nilcheckelim0(f *Func) {
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
		// TODO: This loop is O(n^2). See BenchmarkNilCheckDeep*.
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
			f.removePredecessor(b, b.Succs[1])
			b.Succs = b.Succs[:1]
		}
	}
}

// checkedptr returns the Value, if any,
// that is used in a nil check in b's Control op.
func checkedptr(b *Block) *Value {
	if b.Kind == BlockIf && b.Control.Op == OpIsNonNil {
		return b.Control.Args[0]
	}
	return nil
}
