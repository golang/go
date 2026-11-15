// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"go/constant"
	"go/token"
	"strconv"
)

func (fl *fileLower) cfCondValue(e syntax.Expr, active cfChain, sites *cfSites) (constant.Value, bool) {
	if v, ok := fl.cfDirectValue(e, active, sites); ok {
		return v, true
	}
	return fl.cfRecordedValue(e, active)
}

func (fl *fileLower) cfDirectValue(e syntax.Expr, active cfChain, sites *cfSites) (constant.Value, bool) {
	switch e := e.(type) {
	case *syntax.ParenExpr:
		return fl.cfCondValue(e.X, active, sites)
	case *syntax.Name:
		return cfNameValue(e)
	case *syntax.Operation:
		return fl.cfOperationValue(e, active, sites)
	case *syntax.CallExpr:
		return fl.cfPredicateValue(e, active, sites)
	}
	return nil, false
}

func cfNameValue(n *syntax.Name) (constant.Value, bool) {
	switch n.Value {
	case "true":
		return constant.MakeBool(true), true
	case "false":
		return constant.MakeBool(false), true
	}
	return nil, false
}

func (fl *fileLower) cfOperationValue(e *syntax.Operation, active cfChain, sites *cfSites) (constant.Value, bool) {
	x, ok := fl.cfCondValue(e.X, active, sites)
	if !ok {
		return nil, false
	}
	if e.Y == nil {
		return cfUnaryValue(e.Op, x)
	}
	y, ok := fl.cfCondValue(e.Y, active, sites)
	if !ok {
		return nil, false
	}
	return cfBinaryValue(e.Op, x, y)
}

func cfUnaryValue(op syntax.Operator, x constant.Value) (constant.Value, bool) {
	if op != syntax.Not || x.Kind() != constant.Bool {
		return nil, false
	}
	return constant.UnaryOp(token.NOT, x, 0), true
}

func cfBinaryValue(op syntax.Operator, x, y constant.Value) (constant.Value, bool) {
	if x.Kind() != y.Kind() {
		return nil, false
	}
	switch op {
	case syntax.AndAnd:
		return cfBoolOp(x, token.LAND, y)
	case syntax.OrOr:
		return cfBoolOp(x, token.LOR, y)
	case syntax.Eql:
		return constant.MakeBool(constant.Compare(x, token.EQL, y)), true
	case syntax.Neq:
		return constant.MakeBool(constant.Compare(x, token.NEQ, y)), true
	}
	return nil, false
}

func cfBoolOp(x constant.Value, op token.Token, y constant.Value) (constant.Value, bool) {
	if x.Kind() != constant.Bool {
		return nil, false
	}
	return constant.BinaryOp(x, op, y), true
}

func (fl *fileLower) cfPredicateValue(call *syntax.CallExpr, active cfChain, sites *cfSites) (constant.Value, bool) {
	if v, ok := fl.logosLayoutCallValue(call, active, sites); ok {
		return v, true
	}
	targ, ok := cfPredicateTarget(call, sites)
	if !ok {
		return nil, false
	}
	t, ok := fl.cfResolveTypeExpr(targ, active, sites)
	if !ok {
		return nil, false
	}
	_, isStruct := types2.Unalias(t).Underlying().(*types2.Struct)
	return constant.MakeBool(isStruct), true
}

func cfPredicateTarget(call *syntax.CallExpr, sites *cfSites) (syntax.Expr, bool) {
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 || sites == nil {
		return nil, false
	}
	sel, ok := idx.X.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "IsStruct" {
		return nil, false
	}
	pkg, ok := sel.X.(*syntax.Name)
	targs := syntax.UnpackListExpr(idx.Index)
	if !ok || pkg.Value != sites.comptimeName || len(targs) != 1 {
		return nil, false
	}
	return targs[0], true
}

func (fl *fileLower) cfRecordedValue(e syntax.Expr, active cfChain) (constant.Value, bool) {
	mentions := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		if name, ok := n.(*syntax.Name); ok && cfActiveFor(active, name.Value) != nil {
			mentions = true
		}
		return !mentions
	})
	if mentions {
		return nil, false
	}
	return typeConst(fl.pkg.info, e)
}

func (fl *fileLower) cfLiveBranch(s *syntax.ConstIfStmt, active cfChain, sites *cfSites) (live *syntax.BlockStmt, ok bool) {
	val, ok := fl.cfCondValue(s.Cond, active, sites)
	if !ok || val.Kind() != constant.Bool {
		return nil, false
	}
	if constant.BoolVal(val) {
		return s.Then, true
	}
	switch els := s.Else.(type) {
	case *syntax.BlockStmt:
		return els, true
	case *syntax.ConstIfStmt:
		return fl.cfLiveBranch(els, active, sites)
	}
	return nil, true
}

func cfInConstFor(n syntax.Node, parents map[syntax.Node]syntax.Node) bool {
	for p := parents[n]; p != nil; p = parents[p] {
		if _, ok := p.(*syntax.ConstForStmt); ok {
			return true
		}
	}
	return false
}

func (fl *fileLower) cfPruneConstIf(s *syntax.ConstIfStmt, active cfChain, run *cfRun, add func(int, int, string)) *syntax.BlockStmt {
	live, ok := fl.cfLiveBranch(s, active, run.sites)
	if !ok {
		fl.cfError(run.sites.at(s.Cond.Pos()), "[comptime::not_constant] const if condition cannot be evaluated for this instantiation")
	}
	starts := run.ctx.starts
	start := offAt(starts, s.Pos())
	end := offAt(starts, syntax.EndPos(s))
	liveStart, liveEnd := end, end
	if live != nil {
		liveStart = offAt(starts, live.Pos())
		liveEnd = offAt(starts, syntax.EndPos(live)) + 1
	}
	add(start, liveStart, "")
	add(liveEnd, end, "")
	return live
}

func (fl *fileLower) cfInnerConstFors(body *syntax.BlockStmt, active cfChain, sites *cfSites) []*syntax.ConstForStmt {
	var out []*syntax.ConstForStmt
	var visit func(syntax.Node) bool
	visit = func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ConstForStmt:
			out = append(out, n)
			return false
		case *syntax.ConstIfStmt:
			if live, _ := fl.cfLiveBranch(n, active, sites); live != nil {
				syntax.Inspect(live, visit)
			}
			return false
		}
		return true
	}
	syntax.Inspect(body, visit)
	return out
}

func (b *cfRecordBuilder) foldCond(e syntax.Expr) {
	if val, ok := typeConst(b.fl.pkg.info, e); ok && val.Kind() == constant.Bool {
		ctx := cfCtx{buf: b.fl.srcMap, starts: b.fl.start}
		s := offAt(ctx.starts, syntax.StartPos(e))
		end := cfFoldEnd(ctx, e)
		b.subs = append(b.subs, splice{start: s, end: end, text: keepComments(ctx.buf.b, s, end, strconv.FormatBool(constant.BoolVal(val)))})
		return
	}
	switch e := e.(type) {
	case *syntax.ParenExpr:
		b.foldCond(e.X)
	case *syntax.Operation:
		b.foldCond(e.X)
		if e.Y != nil {
			b.foldCond(e.Y)
		}
	default:
		syntax.Inspect(e, b.visit)
	}
}
