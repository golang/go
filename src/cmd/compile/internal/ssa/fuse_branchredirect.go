// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// fuseBranchRedirect checks for a CFG in which the outbound branch
// of an If block can be derived from its predecessor If block, in
// some such cases, we can redirect the predecessor If block to the
// corresponding successor block directly. For example:
//
//	p:
//	  v11 = Less64 <bool> v10 v8
//	  If v11 goto b else u
//	b: <- p ...
//	  v17 = Leq64 <bool> v10 v8
//	  If v17 goto s else o
//
// We can redirect p to s directly.
//
// The implementation here borrows the framework of the prove pass.
//
//	1, Traverse all blocks of function f to find If blocks.
//	2,   For any If block b, traverse all its predecessors to find If blocks.
//	3,     For any If block predecessor p, update relationship p->b.
//	4,     Traverse all successors of b.
//	5,       For any successor s of b, try to update relationship b->s, if a
//	         contradiction is found then redirect p to another successor of b.
func fuseBranchRedirect(f *Func) bool {
	ft := newFactsTable(f)
	ft.checkpoint()

	changed := false
	for i := len(f.Blocks) - 1; i >= 0; i-- {
		b := f.Blocks[i]
		if b.Kind != BlockIf {
			continue
		}
		// b is either empty or only contains the control value.
		// TODO: if b contains only OpCopy or OpNot related to b.Controls,
		// such as Copy(Not(Copy(Less64(v1, v2)))), perhaps it can be optimized.
		bCtl := b.Controls[0]
		if bCtl.Block != b && len(b.Values) != 0 || (len(b.Values) != 1 || bCtl.Uses != 1) && bCtl.Block == b {
			continue
		}

		for k := 0; k < len(b.Preds); k++ {
			pk := b.Preds[k]
			p := pk.b
			if p.Kind != BlockIf || p == b {
				continue
			}
			pbranch := positive
			if pk.i == 1 {
				pbranch = negative
			}
			ft.checkpoint()
			// Assume branch p->b is taken.
			addBranchRestrictions(ft, p, pbranch)
			// Check if any outgoing branch is unreachable based on the above condition.
			parent := b
			for j, bbranch := range [...]branch{positive, negative} {
				ft.checkpoint()
				// Try to update relationship b->child, and check if the contradiction occurs.
				addBranchRestrictions(ft, parent, bbranch)
				unsat := ft.unsat
				ft.restore()
				if !unsat {
					continue
				}
				// This branch is impossible,so redirect p directly to another branch.
				out := 1 ^ j
				child := parent.Succs[out].b
				if child == b {
					continue
				}
				b.removePred(k)
				p.Succs[pk.i] = Edge{child, len(child.Preds)}
				// Fix up Phi value in b to have one less argument.
				for _, v := range b.Values {
					if v.Op != OpPhi {
						continue
					}
					b.removePhiArg(v, k)
				}
				// Fix up child to have one more predecessor.
				child.Preds = append(child.Preds, Edge{p, pk.i})
				ai := b.Succs[out].i
				for _, v := range child.Values {
					if v.Op != OpPhi {
						continue
					}
					v.AddArg(v.Args[ai])
				}
				if b.Func.pass.debug > 0 {
					b.Func.Warnl(b.Controls[0].Pos, "Redirect %s based on %s", b.Controls[0].Op, p.Controls[0].Op)
				}
				changed = true
				k--
				break
			}
			ft.restore()
		}
		if len(b.Preds) == 0 && b != f.Entry {
			// Block is now dead.
			b.Kind = BlockInvalid
		}
	}
	ft.restore()
	ft.cleanup(f)
	return changed
}
