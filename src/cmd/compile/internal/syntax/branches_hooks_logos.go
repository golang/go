// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func logosBranchesGroup(ls *labelScope, stmt Stmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	switch s := stmt.(type) {
	case *GroupStmt:
		inner := targets{ctxt.breaks, ctxt.continues, -1}
		innerBlock(inner, s.Body.Pos(), s.Body.List)
	case *GroupChild:
		checkBranches(s.Body, ls.errh)
	}
}

func logosBranchesConstFor(s *ConstForStmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	inner := targets{ctxt.breaks, ctxt.continues, -1}
	innerBlock(inner, s.Body.Pos(), s.Body.List)
}

func logosBranchesSwitch(ls *labelScope, s *SwitchStmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	inner := targets{s, ls.logosContinues(ctxt), -1}
	depth := ls.constSwitchDepth
	ls.constSwitchDepth = 0
	for i, cc := range s.Body {
		inner.caseIndex = i
		innerBlock(inner, cc.Pos(), cc.Body)
	}
	ls.constSwitchDepth = depth
}

func logosBranchesSelect(ls *labelScope, s *SelectStmt, ctxt targets, innerBlock func(targets, Pos, []Stmt)) {
	inner := targets{s, ls.logosContinues(ctxt), -1}
	depth := ls.constSwitchDepth
	ls.constSwitchDepth = 0
	for _, cc := range s.Body {
		innerBlock(inner, cc.Pos(), cc.Body)
	}
	ls.constSwitchDepth = depth
}
