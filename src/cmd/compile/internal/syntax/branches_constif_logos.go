// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

const logosConstSwitchJump = "[comptime::const_switch_jump]"

func logosConstBranches(ls *labelScope, stmt Stmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	inner := targets{ctxt.breaks, ctxt.continues, -1}
	switch s := stmt.(type) {
	case *ConstIfStmt:
		innerBlock(inner, s.Then.Pos(), s.Then.List)
		if s.Else != nil {
			innerBlock(inner, s.Else.Pos(), []Stmt{s.Else})
		}
	case *ConstSwitchStmt:
		uncased := targets{nil, nil, -1}
		outer := ls.constContinues
		ls.constContinues = ls.logosContinues(ctxt)
		ls.constSwitchDepth++
		for _, c := range s.Cases {
			innerBlock(uncased, c.Case, c.Body)
		}
		ls.constSwitchDepth--
		ls.constContinues = outer
	}
}

func (ls *labelScope) logosContinues(ctxt targets) *ForStmt {
	if ctxt.continues != nil {
		return ctxt.continues
	}
	return ls.constContinues
}

func (ls *labelScope) logosJumpError(s *BranchStmt, plain string) {
	if ls.constSwitchDepth > 0 {
		ls.errf(s.Pos(), "%s %s directly in a const switch case: the cases lower into generated switches", logosConstSwitchJump, s.Tok)
		return
	}
	ls.errf(s.Pos(), "%s", plain)
}
