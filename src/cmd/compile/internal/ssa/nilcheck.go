// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// nilcheckelim eliminates unnecessary nil checks.
func nilcheckelim(f *Func) {
	// A nil check is redundant if the same nil check was successful in a
	// dominating block. The efficacy of this pass depends heavily on the
	// efficacy of the cse pass.
	idom := f.idom
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
	work = append(work, bp{block: f.Entry})

	// map from value ID to bool indicating if value is known to be non-nil
	// in the current dominator path being walked. This slice is updated by
	// walkStates to maintain the known non-nil values.
	nonNilValues := make([]bool, f.NumValues())

	// make an initial pass identifying any non-nil values
	for _, b := range f.Blocks {
		// a value resulting from taking the address of a
		// value, or a value constructed from an offset of a
		// non-nil ptr (OpAddPtr) implies it is non-nil
		for _, v := range b.Values {
			if v.Op == OpAddr || v.Op == OpAddPtr {
				nonNilValues[v.ID] = true
			} else if v.Op == OpPhi {
				// phis whose arguments are all non-nil
				// are non-nil
				argsNonNil := true
				for _, a := range v.Args {
					if !nonNilValues[a.ID] {
						argsNonNil = false
					}
				}
				if argsNonNil {
					nonNilValues[v.ID] = true
				}
			}
		}
	}

	// perform a depth first walk of the dominee tree
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]

		switch node.op {
		case Work:
			checked := checkedptr(node.block) // ptr being checked for nil/non-nil
			nonnil := nonnilptr(node.block)   // ptr that is non-nil due to this blocks pred

			if checked != nil {
				// already have a nilcheck in the dominator path, or this block is a success
				// block for the same value it is checking
				if nonNilValues[checked.ID] || checked == nonnil {
					// Eliminate the nil check.
					// The deadcode pass will remove vestigial values,
					// and the fuse pass will join this block with its successor.

					// Logging in the style of the former compiler -- and omit line 1,
					// which is usually in generated code.
					if f.Config.Debug_checknil() && node.block.Control.Line > 1 {
						f.Config.Warnl(node.block.Control.Line, "removed nil check")
					}

					switch node.block.Kind {
					case BlockIf:
						node.block.Kind = BlockFirst
						node.block.SetControl(nil)
					case BlockCheck:
						node.block.Kind = BlockPlain
						node.block.SetControl(nil)
					default:
						f.Fatalf("bad block kind in nilcheck %s", node.block.Kind)
					}
				}
			}

			if nonnil != nil && !nonNilValues[nonnil.ID] {
				// this is a new nilcheck so add a ClearPtr node to clear the
				// ptr from the map of nil checks once we traverse
				// back up the tree
				work = append(work, bp{op: ClearPtr, ptr: nonnil})
			}

			// add all dominated blocks to the work list
			for _, w := range domTree[node.block.ID] {
				work = append(work, bp{block: w})
			}

			if nonnil != nil && !nonNilValues[nonnil.ID] {
				work = append(work, bp{op: RecPtr, ptr: nonnil})
			}
		case RecPtr:
			nonNilValues[node.ptr.ID] = true
			continue
		case ClearPtr:
			nonNilValues[node.ptr.ID] = false
			continue
		}
	}
}

// checkedptr returns the Value, if any,
// that is used in a nil check in b's Control op.
func checkedptr(b *Block) *Value {
	if b.Kind == BlockCheck {
		return b.Control.Args[0]
	}
	if b.Kind == BlockIf && b.Control.Op == OpIsNonNil {
		return b.Control.Args[0]
	}
	return nil
}

// nonnilptr returns the Value, if any,
// that is non-nil due to b being the successor block
// of an OpIsNonNil or OpNilCheck block for the value and having a single
// predecessor.
func nonnilptr(b *Block) *Value {
	if len(b.Preds) == 1 {
		bp := b.Preds[0].b
		if bp.Kind == BlockCheck {
			return bp.Control.Args[0]
		}
		if bp.Kind == BlockIf && bp.Control.Op == OpIsNonNil && bp.Succs[0].b == b {
			return bp.Control.Args[0]
		}
	}
	return nil
}
