// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// logosConstBranches lets checkBranches see the labels and gotos inside
// the blocks of a `const if` (M7-S1) or `const switch` (M7-S4) statement,
// so a label used only there counts as used. The blocks keep the break
// and continue targets of the enclosing statement: whether an unlabeled
// break or continue may appear in them is a types2 rule, so the parser
// reports nothing for it here. innerBlock is the closure of blockBranches
// that walks one nested block.
func logosConstBranches(stmt Stmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	inner := targets{ctxt.breaks, ctxt.continues, -1}
	switch s := stmt.(type) {
	case *ConstIfStmt:
		innerBlock(inner, s.Then.Pos(), s.Then.List)
		if s.Else != nil {
			innerBlock(inner, s.Else.Pos(), []Stmt{s.Else})
		}
	case *ConstSwitchStmt:
		for _, c := range s.Cases {
			innerBlock(inner, c.Case, c.Body)
		}
	}
}
