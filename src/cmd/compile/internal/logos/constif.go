// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

func (fc *funcLower) lowerConstIf() {
	fc.body.List = fc.lowerConstIfList(fc.body.List)
}

func (fc *funcLower) lowerConstIfList(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		if kept, ok := fc.pruneConstIfStmt(s); ok {
			out = append(out, kept)
		}
	}
	return out
}

func (fc *funcLower) pruneConstIfStmt(s syntax.Stmt) (syntax.Stmt, bool) {
	if c, ok := s.(*syntax.ConstIfStmt); ok {
		return fc.pruneConstIf(c)
	}
	fc.pruneConstIfChildren(s)
	return s, true
}

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

func (fc *funcLower) pruneConstIfElse(s *syntax.IfStmt) {
	s.Then.List = fc.lowerConstIfList(s.Then.List)
	switch els := s.Else.(type) {
	case *syntax.BlockStmt:
		els.List = fc.lowerConstIfList(els.List)
	case *syntax.IfStmt:
		fc.pruneConstIfElse(els)
	}
}

func (fc *funcLower) pruneCaseClauses(clauses []*syntax.CaseClause) {
	for _, clause := range clauses {
		clause.Body = fc.lowerConstIfList(clause.Body)
	}
}

func (fc *funcLower) pruneCommClauses(clauses []*syntax.CommClause) {
	for _, clause := range clauses {
		clause.Body = fc.lowerConstIfList(clause.Body)
	}
}

func (fc *funcLower) pruneConstCases(cases []*syntax.ConstSwitchCase) {
	for _, c := range cases {
		c.Body = fc.lowerConstIfList(c.Body)
	}
}

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

func (fc *funcLower) cutConstSpan(start, end int) {
	if start >= end {
		return
	}
	_ = fc.fl.dropInner(span{start, end})
	fc.fl.addT(start, end, leftoverComments(fc.fl, span{start, end}, nil))
}

func (fc *funcLower) deleteConstSpan(start, end int) {
	if start >= end {
		return
	}
	_ = fc.fl.dropInner(span{start, end})
	fc.fl.add(start, end, "")
}
