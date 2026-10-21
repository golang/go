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

// LoGo/s M7-S2/M7-S3 `const if` inside expansions (SPEC Sec 2.5, T-205).
// A comptime generic copy and an unrolled const for body are built from
// source text that the checker never saw per instantiation, so their
// `const if` statements cannot read a recorded constant. They evaluate
// here instead: the condition is a boolean expression over `true`,
// `false`, comptime.IsStruct[X]() with X resolved in the expansion
// context, and the connectives `!`, `&&`, `||`, `==`, `!=`. A condition
// that does not depend on the expansion reads the constant the checker
// recorded. The live branch replaces the statement in the text; the dead
// branches go away with their text, so a const for over a non-struct
// inside one never resolves.

// cfCondValue evaluates the constant expression e of a `const if`
// condition under the active bindings. It reports false when e is not
// evaluable.
func (fl *fileLower) cfCondValue(e syntax.Expr, active cfChain, sites *cfSites) (constant.Value, bool) {
	if v, ok := fl.cfDirectValue(e, active, sites); ok {
		return v, true
	}
	return fl.cfRecordedValue(e, active)
}

// cfDirectValue evaluates e from its own syntax, without the checker's
// record.
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

// cfNameValue reads `true` and `false`.
func cfNameValue(n *syntax.Name) (constant.Value, bool) {
	switch n.Value {
	case "true":
		return constant.MakeBool(true), true
	case "false":
		return constant.MakeBool(false), true
	}
	return nil, false
}

// cfOperationValue evaluates `!x` and the binary connectives.
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

// cfUnaryValue applies `!` to a boolean value.
func cfUnaryValue(op syntax.Operator, x constant.Value) (constant.Value, bool) {
	if op != syntax.Not || x.Kind() != constant.Bool {
		return nil, false
	}
	return constant.UnaryOp(token.NOT, x, 0), true
}

// cfBinaryValue applies a connective to two values of one kind.
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

// cfBoolOp applies a boolean connective to two boolean values.
func cfBoolOp(x constant.Value, op token.Token, y constant.Value) (constant.Value, bool) {
	if x.Kind() != constant.Bool {
		return nil, false
	}
	return constant.BinaryOp(x, op, y), true
}

// cfPredicateValue evaluates comptime.IsStruct[X](): true when the type
// X resolves to, under its underlying type, a struct.
func (fl *fileLower) cfPredicateValue(call *syntax.CallExpr, active cfChain, sites *cfSites) (constant.Value, bool) {
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

// cfPredicateTarget returns the type argument of a call written
// `<comptime>.IsStruct[X]()`.
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

// cfRecordedValue reads the constant the checker recorded for e. An
// expression that names an active const for variable has one value per
// iteration, so the recorded value of the last iteration never stands
// in for it.
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

// cfLiveBranch returns the block that the const-if chain rooted at s
// selects, or nil when no branch is selected. ok is false when a
// condition of the chain is not evaluable.
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

// cfInConstFor reports whether n lies under a const for of its tree.
func cfInConstFor(n syntax.Node, parents map[syntax.Node]syntax.Node) bool {
	for p := parents[n]; p != nil; p = parents[p] {
		if _, ok := p.(*syntax.ConstForStmt); ok {
			return true
		}
	}
	return false
}

// cfPruneConstIf adds the two deletions that turn the `const if` chain s
// into its live block: the text before the block and the text after it.
// With no live block both cover the whole statement. It returns the live
// block, whose own substitutions the caller still collects, and reports
// a condition that is not evaluable.
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

// cfInnerConstFors lists the outermost const fors under body in order
// that the expansion keeps: a const for in a dead branch of a `const if`
// is not listed, and no dead branch is searched.
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

// foldCond splices the constant sub-conditions of a `const if`
// condition into the record text as `true` or `false`: a record is
// parsed again without type information, so a condition that does not
// depend on a type parameter must already be a literal there. The
// predicate leaves, which wait for an instantiation, keep their text,
// and the names inside them take the record's markers through visit.
func (b *cfRecordBuilder) foldCond(e syntax.Expr) {
	if val, ok := typeConst(b.fl.pkg.info, e); ok && val.Kind() == constant.Bool {
		ctx := cfCtx{src: b.fl.src, starts: b.fl.start}
		s := offAt(ctx.starts, syntax.StartPos(e))
		b.subs = append(b.subs, splice{start: s, end: cfExactEnd(ctx, e), text: strconv.FormatBool(constant.BoolVal(val))})
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
