// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements isTerminating.

package types2

import (
	"cmd/compile/internal/syntax"
)

// isTerminating reports if s is a terminating statement.
// If s is labeled, label is the label name; otherwise s
// is "".
func (check *Checker) isTerminating(s syntax.Stmt, label string) bool {
	switch s := s.(type) {
	default:
		unreachable()

	case *syntax.DeclStmt, *syntax.EmptyStmt, *syntax.SendStmt,
		*syntax.AssignStmt, *syntax.CallStmt:
		// no chance

	case *syntax.LabeledStmt:
		return check.isTerminating(s.Stmt, s.Label.Value)

	case *syntax.ExprStmt:
		// calling the predeclared (possibly parenthesized) panic() function is terminating
		if call, ok := unparen(s.X).(*syntax.CallExpr); ok && check.isPanic[call] {
			return true
		}

	case *syntax.ReturnStmt:
		return true

	case *syntax.BranchStmt:
		if s.Tok == syntax.Goto || s.Tok == syntax.Fallthrough {
			return true
		}

	case *syntax.BlockStmt:
		return check.isTerminatingList(s.List, "")

	case *syntax.IfStmt:
		if s.Else != nil &&
			check.isTerminating(s.Then, "") &&
			check.isTerminating(s.Else, "") {
			return true
		}

	case *syntax.SwitchStmt:
		return check.isTerminatingSwitch(s.Body, label)

	case *syntax.SelectStmt:
		for _, cc := range s.Body {
			if !check.isTerminatingList(cc.Body, "") || hasBreakList(cc.Body, label, true) {
				return false
			}

		}
		return true

	case *syntax.ForStmt:
		if _, ok := s.Init.(*syntax.RangeClause); ok {
			// Range clauses guarantee that the loop terminates,
			// so the loop is not a terminating statement. See issue 49003.
			break
		}
		if s.Cond == nil && !hasBreak(s.Body, label, true) {
			return true
		}
	}

	return false
}

func (check *Checker) isTerminatingList(list []syntax.Stmt, label string) bool {
	// trailing empty statements are permitted - skip them
	for i := len(list) - 1; i >= 0; i-- {
		if _, ok := list[i].(*syntax.EmptyStmt); !ok {
			return check.isTerminating(list[i], label)
		}
	}
	return false // all statements are empty
}

func (check *Checker) isTerminatingSwitch(body []*syntax.CaseClause, label string) bool {
	hasDefault := false
	for _, cc := range body {
		if cc.Cases == nil {
			hasDefault = true
		}
		if !check.isTerminatingList(cc.Body, "") || hasBreakList(cc.Body, label, true) {
			return false
		}
	}
	return hasDefault
}

// TODO(gri) For nested breakable statements, the current implementation of hasBreak
//	     will traverse the same subtree repeatedly, once for each label. Replace
//           with a single-pass label/break matching phase.

// hasBreak reports if s is or contains a break statement
// referring to the label-ed statement or implicit-ly the
// closest outer breakable statement.
func hasBreak(s syntax.Stmt, label string, implicit bool) bool {
	switch s := s.(type) {
	default:
		unreachable()

	case *syntax.DeclStmt, *syntax.EmptyStmt, *syntax.ExprStmt,
		*syntax.SendStmt, *syntax.AssignStmt, *syntax.CallStmt,
		*syntax.ReturnStmt:
		// no chance

	case *syntax.LabeledStmt:
		return hasBreak(s.Stmt, label, implicit)

	case *syntax.BranchStmt:
		if s.Tok == syntax.Break {
			if s.Label == nil {
				return implicit
			}
			if s.Label.Value == label {
				return true
			}
		}

	case *syntax.BlockStmt:
		return hasBreakList(s.List, label, implicit)

	case *syntax.IfStmt:
		if hasBreak(s.Then, label, implicit) ||
			s.Else != nil && hasBreak(s.Else, label, implicit) {
			return true
		}

	case *syntax.SwitchStmt:
		if label != "" && hasBreakCaseList(s.Body, label, false) {
			return true
		}

	case *syntax.SelectStmt:
		if label != "" && hasBreakCommList(s.Body, label, false) {
			return true
		}

	case *syntax.ForStmt:
		if label != "" && hasBreak(s.Body, label, false) {
			return true
		}
	}

	return false
}

func hasBreakList(list []syntax.Stmt, label string, implicit bool) bool {
	for _, s := range list {
		if hasBreak(s, label, implicit) {
			return true
		}
	}
	return false
}

func hasBreakCaseList(list []*syntax.CaseClause, label string, implicit bool) bool {
	for _, s := range list {
		if hasBreakList(s.Body, label, implicit) {
			return true
		}
	}
	return false
}

func hasBreakCommList(list []*syntax.CommClause, label string, implicit bool) bool {
	for _, s := range list {
		if hasBreakList(s.Body, label, implicit) {
			return true
		}
	}
	return false
}
