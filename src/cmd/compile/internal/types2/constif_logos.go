// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

// LoGo/s M7-S1 `const if` checking (SPEC Sec 2.5, T-205). The condition
// must be a constant boolean; both branches type-check, so an undefined
// name in the dead branch is an error. Lowering keeps the live branch.
// Dead-branch statements are recorded in info.LogosConstIfDead for the
// sibling M7-S2/S3 comptime checks.

const logosNotConstant = "[comptime::not_constant]"

// logosConstIf checks a `const if` statement: the condition must be a
// constant boolean, and both branches check in the enclosing context.
// The live branch index feeds the dead-branch map; an invalid condition
// still checks both branches without marking either dead.
func (check *Checker) logosConstIf(ctxt stmtContext, s *syntax.ConstIfStmt) {
	var x operand
	check.expr(nil, &x, s.Cond)
	live := -1
	if x.isValid() {
		switch {
		case x.val == nil:
			check.errorf(s.Cond, InvalidCond, "%s const if condition is not constant", logosNotConstant)
		case !allBoolean(x.typ()):
			check.error(s.Cond, InvalidCond, "non-boolean condition in const if statement")
		case constant.BoolVal(x.val):
			live = 0
		default:
			live = 1
		}
	}
	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)
	check.stmt(inner, s.Then)
	if s.Else != nil {
		check.stmt(inner, s.Else)
	}
	if live >= 0 {
		check.logosMarkConstIfDead(s, live)
	}
}

// logosMarkConstIfDead records every statement of the dead branch of s
// in info.LogosConstIfDead: live 0 keeps Then and kills Else, live 1
// keeps Else (or nothing when Else is absent) and kills Then.
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

// logosTerminatingConstIf reports whether a `const if` terminates: the
// live branch decides when the condition is constant, and both branches
// must terminate otherwise.
func (check *Checker) logosTerminatingConstIf(s *syntax.ConstIfStmt) bool {
	if v, ok := check.constIfValue(s.Cond); ok {
		if v {
			return check.isTerminating(s.Then, "")
		}
		return s.Else != nil && check.isTerminating(s.Else, "")
	}
	return s.Else != nil && check.isTerminating(s.Then, "") && check.isTerminating(s.Else, "")
}

// constIfValue reads the constant boolean value of a checked condition.
func (check *Checker) constIfValue(e syntax.Expr) (bool, bool) {
	tv, ok := check.Info.Types[e]
	if !ok || tv.Value == nil || tv.Value.Kind() != constant.Bool {
		return false, false
	}
	return constant.BoolVal(tv.Value), true
}

// logosHasBreakConstIf reports whether a `const if` contains a break
// matching the label search.
func logosHasBreakConstIf(s *syntax.ConstIfStmt, label string, implicit bool) bool {
	if hasBreak(s.Then, label, implicit) {
		return true
	}
	return s.Else != nil && hasBreak(s.Else, label, implicit)
}

// markDeadStmts records dead and every statement nested in it, so a
// comptime check on any nested node can test membership directly.
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
