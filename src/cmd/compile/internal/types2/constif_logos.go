// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

const logosNotConstant = "[comptime::not_constant]"

func (check *Checker) logosConstIf(ctxt stmtContext, s *syntax.ConstIfStmt) {
	live := check.logosConstIfLive(s)
	if live >= 0 {
		check.logosMarkConstIfDead(s, live)
	}
	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)
	check.stmt(inner, s.Then)
	if s.Else != nil {
		check.stmt(inner, s.Else)
	}
}

func (check *Checker) logosConstIfLive(s *syntax.ConstIfStmt) int {
	var x operand
	outer := check.logosPred
	check.logosPred = logosPredState{depth: 1}
	check.expr(nil, &x, s.Cond)
	deferred := check.logosPred.calls
	check.logosPred = outer
	switch {
	case !x.isValid():
	case x.val == nil && check.logosPredOnly(s.Cond, deferred):
	case x.val == nil:
		check.errorf(s.Cond, InvalidCond, "%s const if condition is not constant", logosNotConstant)
	case !allBoolean(x.typ()):
		check.error(s.Cond, InvalidCond, "non-boolean condition in const if statement")
	case constant.BoolVal(x.val):
		return 0
	default:
		return 1
	}
	return -1
}

func (check *Checker) logosMarkConstIfDead(s *syntax.ConstIfStmt, live int) {
	if check.Info.LogosConstIfDead == nil {
		check.Info.LogosConstIfDead = make(map[syntax.Stmt]bool)
	}
	var dead syntax.Stmt
	if live == 0 {
		dead = s.Else
	} else {
		dead = s.Then
	}
	markDeadStmts(check.Info.LogosConstIfDead, dead)
}

func (check *Checker) logosTerminatingConstIf(s *syntax.ConstIfStmt) bool {
	if v, ok := check.constIfValue(s.Cond); ok {
		if v {
			return check.isTerminating(s.Then, "")
		}
		return s.Else != nil && check.isTerminating(s.Else, "")
	}
	return s.Else != nil && check.isTerminating(s.Then, "") && check.isTerminating(s.Else, "")
}

func (check *Checker) constIfValue(e syntax.Expr) (bool, bool) {
	_, val, ok := check.logosChecked(e)
	if !ok || val == nil || val.Kind() != constant.Bool {
		return false, false
	}
	return constant.BoolVal(val), true
}

func logosHasBreakConstIf(s *syntax.ConstIfStmt, label string, implicit bool) bool {
	if hasBreak(s.Then, label, implicit) {
		return true
	}
	return s.Else != nil && hasBreak(s.Else, label, implicit)
}

func markDeadStmts(dead map[syntax.Stmt]bool, s syntax.Stmt) {
	if s == nil {
		return
	}
	dead[s] = true
	syntax.Inspect(s, func(n syntax.Node) bool {
		if st, ok := n.(syntax.Stmt); ok && st != s {
			dead[st] = true
		}
		return true
	})
}
