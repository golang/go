// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

// This file lowers M7-S1 `const if` statements (SPEC Sec 2.5, T-205): a
// `const if` becomes its live branch block, so only live code reaches
// the backend, or vanishes when the condition is false with no else.
// The rewrite uses small deletion splices around the kept block rather
// than one replacement of the whole statement, so splices of nested
// constructs stay valid beside them.

// lowerConstIf prunes the dead `const if` branches of one function body.
func (fc *funcLower) lowerConstIf() {
	fc.body.List = fc.lowerConstIfList(fc.body.List)
}

// lowerConstIfList prunes every `const if` of one statement list.
func (fc *funcLower) lowerConstIfList(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		if kept, ok := fc.pruneConstIfStmt(s); ok {
			out = append(out, kept)
		}
	}
	return out
}

// pruneConstIfStmt prunes the `const if` statements under s, descending
// into every child list except nested function literals, which lower
// under their own funcLower. It reports false when s itself vanishes.
func (fc *funcLower) pruneConstIfStmt(s syntax.Stmt) (syntax.Stmt, bool) {
	if c, ok := s.(*syntax.ConstIfStmt); ok {
		return fc.pruneConstIf(c)
	}
	fc.pruneConstIfChildren(s)
	return s, true
}

// pruneConstIfChildren prunes the statement lists that s holds.
func (fc *funcLower) pruneConstIfChildren(s syntax.Stmt) {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = fc.lowerConstIfList(s.List)
	case *syntax.IfStmt:
		fc.pruneConstIfElse(s)
	case *syntax.ForStmt:
		s.Body.List = fc.lowerConstIfList(s.Body.List)
	case *syntax.SwitchStmt:
		fc.pruneCaseClauses(s.Body)
	case *syntax.SelectStmt:
		fc.pruneCommClauses(s.Body)
	case *syntax.LabeledStmt:
		fc.pruneLabeled(s)
	case *syntax.ConstSwitchStmt:
		fc.pruneConstCases(s.Cases)
	case *syntax.GroupStmt:
		s.Body.List = fc.lowerConstIfList(s.Body.List)
	case *syntax.GroupChild:
		s.Body.List = fc.lowerConstIfList(s.Body.List)
	}
}

// pruneConstIfElse prunes the branches of an if statement, following an
// else-if chain link by link.
func (fc *funcLower) pruneConstIfElse(s *syntax.IfStmt) {
	s.Then.List = fc.lowerConstIfList(s.Then.List)
	switch els := s.Else.(type) {
	case *syntax.BlockStmt:
		els.List = fc.lowerConstIfList(els.List)
	case *syntax.IfStmt:
		fc.pruneConstIfElse(els)
	}
}

// pruneCaseClauses prunes the body of every clause of a switch statement.
func (fc *funcLower) pruneCaseClauses(clauses []*syntax.CaseClause) {
	for _, clause := range clauses {
		clause.Body = fc.lowerConstIfList(clause.Body)
	}
}

// pruneCommClauses prunes the body of every clause of a select statement.
func (fc *funcLower) pruneCommClauses(clauses []*syntax.CommClause) {
	for _, clause := range clauses {
		clause.Body = fc.lowerConstIfList(clause.Body)
	}
}

// pruneConstCases prunes the body of every case of a `const switch`.
func (fc *funcLower) pruneConstCases(cases []*syntax.ConstSwitchCase) {
	for _, c := range cases {
		c.Body = fc.lowerConstIfList(c.Body)
	}
}

// pruneLabeled prunes the statement a label holds. When that statement
// is a `const if` that vanishes, the label keeps an empty statement, so a
// jump to the label stays valid in the source and in the lowered tree.
func (fc *funcLower) pruneLabeled(s *syntax.LabeledStmt) {
	if kept, ok := fc.pruneConstIfStmt(s.Stmt); ok {
		s.Stmt = kept
		return
	}
	inner := span{fc.fl.off(s.Stmt.Pos()), fc.fl.off(syntax.EndPos(s.Stmt))}
	empty := &syntax.EmptyStmt{}
	empty.SetPos(s.Stmt.Pos())
	s.Stmt = empty
	fc.deleteConstSpan(inner.start, inner.end)
	fc.fl.add(inner.start, inner.start, ";")
}

// pruneConstIf replaces one `const if` statement, including its whole
// else chain, with the live branch. Two deletions cut the source before
// and after the live block, so the header and every dead branch go away
// with their splices while the splices of later lowering inside the live
// block keep their own offsets. A false condition with no live branch
// deletes the statement and reports false.
func (fc *funcLower) pruneConstIf(s *syntax.ConstIfStmt) (syntax.Stmt, bool) {
	start := fc.fl.off(s.Pos())
	end := fc.fl.off(syntax.EndPos(s))
	live := fc.liveBranch(s)
	if live == nil {
		fc.deleteConstSpan(start, end)
		return nil, false
	}
	live.List = fc.lowerConstIfList(live.List)
	fc.cutConstSpan(start, fc.fl.off(live.Pos()))
	fc.cutConstSpan(fc.fl.off(syntax.EndPos(live))+1, end)
	return live, true
}

// liveBranch returns the block that the const-if chain rooted at s selects,
// or nil when no branch is selected. It follows the chain without rewriting
// it, so the caller cuts the statement around the block it returns.
func (fc *funcLower) liveBranch(s *syntax.ConstIfStmt) *syntax.BlockStmt {
	val, ok := typeConst(fc.fl.pkg.info, s.Cond)
	if !ok || val.Kind() != constant.Bool {
		base.Fatalf("logos: const if condition at %s has no constant boolean value", s.Cond.Pos().String())
	}
	if constant.BoolVal(val) {
		return s.Then
	}
	switch els := s.Else.(type) {
	case nil:
		return nil
	case *syntax.BlockStmt:
		return els
	case *syntax.ConstIfStmt:
		return fc.liveBranch(els)
	default:
		base.Fatalf("logos: const if else branch at %s is a %T", s.Pos().String(), s.Else)
	}
	return nil
}

// cutConstSpan is deleteConstSpan for the regions around a live branch:
// the comments of the region survive, so the eject text keeps the
// comments of the header and of the dead branches.
func (fc *funcLower) cutConstSpan(start, end int) {
	if start >= end {
		return
	}
	_ = fc.fl.dropInner(span{start, end})
	fc.fl.add(start, end, leftoverComments(fc.fl, span{start, end}, nil))
}

// deleteConstSpan discards the splices strictly inside [start, end) and
// deletes the source text there.
func (fc *funcLower) deleteConstSpan(start, end int) {
	if start >= end {
		return
	}
	_ = fc.fl.dropInner(span{start, end})
	fc.fl.add(start, end, "")
}
