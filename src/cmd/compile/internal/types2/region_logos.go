// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// logosRegionEscape prefixes every region escape diagnostic.
const logosRegionEscape = "[correctness::region_escape]"

func logosRegionIsRegion(t Type) bool {
	n, _ := Unalias(t).(*Named)
	if n == nil || n.Obj() == nil || n.Obj().Pkg() == nil {
		return false
	}
	return n.Obj().Pkg().Path() == "logos/arena" && n.Obj().Name() == "Region"
}

func logosRegionCarries(t Type) bool {
	if t == nil || !isValid(t) {
		return false
	}
	if logosRegionIsRegion(t) {
		return true
	}
	switch u := t.Underlying().(type) {
	case *Struct:
		for i, n := 0, u.NumFields(); i < n; i++ {
			if logosRegionCarries(u.Field(i).Type()) {
				return true
			}
		}
	case *Array:
		return logosRegionCarries(u.Elem())
	}
	return false
}

func (check *Checker) logosRegionLocal(v Object) *Var {
	w, _ := v.(*Var)
	if w == nil || w.pkg != check.pkg || w.Parent() == nil || w.Parent() == check.pkg.scope {
		return nil
	}
	return w
}

func logosRegionBlank(lhs syntax.Expr) bool {
	n, _ := syntax.Unparen(lhs).(*syntax.Name)
	return n != nil && n.Value == "_"
}

func (check *Checker) logosRegionRoot(lhs syntax.Expr) *Var {
	switch e := syntax.Unparen(lhs).(type) {
	case *syntax.Name:
		if e.Value == "_" {
			return nil
		}
		return check.logosRegionLocal(check.lookup(e.Value))
	case *syntax.SelectorExpr:
		switch check.logosRegionUnderlying(e.X).(type) {
		case *Pointer, *Interface, *Map:
			return nil
		}
		return check.logosRegionRoot(e.X)
	case *syntax.IndexExpr:
		if _, ok := check.logosRegionUnderlying(e.X).(*Array); !ok {
			return nil
		}
		return check.logosRegionRoot(e.X)
	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			return nil
		}
	}
	return nil
}

func (check *Checker) logosRegionUnderlying(e syntax.Expr) Type {
	t := check.Info.TypeOf(e)
	if t == nil {
		return nil
	}
	return Unalias(t).Underlying()
}

func (check *Checker) logosRegionHolds(e syntax.Expr) bool {
	e = syntax.Unparen(e)
	if t := check.Info.TypeOf(e); t != nil && logosRegionCarries(t) {
		return true
	}
	switch x := e.(type) {
	case *syntax.FuncLit:
		return check.logosRegionLit[x]
	case *syntax.Name:
		if v := check.logosRegionLocal(check.Info.Uses[x]); v != nil {
			return check.logosRegionVar[v]
		}
	case *syntax.CompositeLit:
		return check.logosRegionHoldsElems(x)
	case *syntax.Operation:
		return x.Op == syntax.And && x.Y == nil && check.logosRegionHolds(x.X)
	case *syntax.CallExpr:
		if check.logosRegionPassesThrough(x.Fun) {
			return check.logosRegionHoldsAny(x.ArgList)
		}
	}
	return false
}

func (check *Checker) logosRegionHoldsAny(list []syntax.Expr) bool {
	for _, e := range list {
		if check.logosRegionHolds(e) {
			return true
		}
	}
	return false
}

func (check *Checker) logosRegionHoldsElems(lit *syntax.CompositeLit) bool {
	_, keyed := check.logosRegionUnderlying(lit).(*Map)
	for _, el := range lit.ElemList {
		if kv, ok := el.(*syntax.KeyValueExpr); ok {
			if keyed && check.logosRegionHolds(kv.Key) {
				return true
			}
			el = kv.Value
		}
		if check.logosRegionHolds(el) {
			return true
		}
	}
	return false
}

func (check *Checker) logosRegionPassesThrough(fun syntax.Expr) bool {
	if tv, ok := check.Info.Types[syntax.Unparen(fun)]; ok && tv.IsType() {
		return true
	}
	name, _ := syntax.Unparen(fun).(*syntax.Name)
	if name == nil || name.Value != "append" {
		return false
	}
	_, builtin := check.Info.Uses[name].(*Builtin)
	return builtin
}

func (check *Checker) logosRegionInitMap() {
	if check.logosRegionLit == nil {
		check.logosRegionLit = make(map[*syntax.FuncLit]bool)
	}
	if check.logosRegionVar == nil {
		check.logosRegionVar = make(map[*Var]bool)
	}
}

func logosRegionScopeInside(s, of *Scope) bool {
	for p := s; p != nil; p = p.Parent() {
		if p == of {
			return true
		}
	}
	return false
}

func (check *Checker) logosRegionLitMark(e *syntax.FuncLit, sig *Signature) {
	if !check.conf.Logos || e.Body == nil || sig == nil {
		return
	}
	check.logosRegionInitMap()
	check.later(func() {
		syntax.Inspect(e.Body, func(n syntax.Node) bool {
			name, ok := n.(*syntax.Name)
			if !ok || name.Value == "_" {
				return true
			}
			v := check.logosRegionLocal(check.Info.Uses[name])
			if v == nil || logosRegionScopeInside(v.Parent(), sig.scope) || !logosRegionCarries(v.typ) && !check.logosRegionVar[v] {
				return true
			}
			check.logosRegionLit[e] = true
			return false
		})
	})
}

func (check *Checker) logosRegionEscapedLit(at poser, what string, e syntax.Expr) {
	check.later(func() {
		if check.logosRegionHolds(e) {
			check.errorf(at, IncompatibleAssign, "%s cannot %s %s: it holds a region value", logosRegionEscape, what, ExprString(e))
		}
	})
}

func (check *Checker) logosRegionAssign(lhs syntax.Expr, x *operand) {
	if !check.conf.Logos || x == nil || !x.isValid() {
		return
	}
	root := check.logosRegionRoot(lhs)
	blank := logosRegionBlank(lhs)
	if !blank && root == nil && logosRegionCarries(x.typ()) {
		check.errorf(x, IncompatibleAssign, "%s cannot store %s in %s: a region value must stay inside its block", logosRegionEscape, ExprString(x.expr), ExprString(lhs))
		return
	}
	if name, _ := syntax.Unparen(lhs).(*syntax.Name); name != nil {
		if lv := check.logosRegionLocal(check.lookup(name.Value)); lv != nil {
			check.logosRegionInitMap()
			check.later(func() {
				if check.logosRegionHolds(x.expr) {
					check.logosRegionVar[lv] = true
				}
			})
		}
	}
	if !blank && root == nil {
		check.logosRegionEscapedLit(lhs, "store", x.expr)
	}
}

func (check *Checker) logosRegionDecl(lhs *Var, x *operand) {
	if !check.conf.Logos || x == nil || !x.isValid() {
		return
	}
	local := check.logosRegionLocal(lhs)
	if local == nil {
		if lhs == nil || lhs.pkg != check.pkg || lhs.Name() == "_" || check.scope == check.pkg.scope {
			return
		}
		local = lhs
	}
	check.logosRegionInitMap()
	e := x.expr
	check.later(func() {
		if check.logosRegionHolds(e) {
			check.logosRegionVar[local] = true
		}
	})
}

func (check *Checker) logosRegionResults(results []syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	for _, r := range results {
		if t := check.Info.TypeOf(r); t != nil && logosRegionCarries(t) {
			check.errorf(r, IncompatibleAssign, "%s cannot return %s", logosRegionEscape, ExprString(r))
			continue
		}
		check.logosRegionEscapedLit(r, "return", r)
	}
}

func (check *Checker) logosRegionNamedResults(at poser, results *Tuple) {
	if !check.conf.Logos {
		return
	}
	for _, result := range results.vars {
		check.later(func() {
			if logosRegionCarries(result.typ) || check.logosRegionVar[result] {
				check.errorf(at, IncompatibleAssign, "%s cannot return %s: it holds a region value", logosRegionEscape, result.name)
			}
		})
	}
}

func (check *Checker) logosRegionSend(val *operand) {
	if !check.conf.Logos || !val.isValid() {
		return
	}
	if logosRegionCarries(val.typ()) {
		check.errorf(val, IncompatibleAssign, "%s cannot send %s on a channel", logosRegionEscape, ExprString(val.expr))
		return
	}
	check.logosRegionEscapedLit(val.expr, "send", val.expr)
}

func (check *Checker) logosRegionSuspended(keyword string, call syntax.Expr) {
	if !check.conf.Logos || keyword != "go" {
		return
	}
	c, ok := syntax.Unparen(call).(*syntax.CallExpr)
	if !ok {
		return
	}
	for _, arg := range c.ArgList {
		if t := check.Info.TypeOf(arg); t != nil && logosRegionCarries(t) {
			check.errorf(arg, InvalidGo, "%s cannot pass %s to a goroutine", logosRegionEscape, ExprString(arg))
			continue
		}
		check.logosRegionEscapedLit(arg, "pass to a goroutine", arg)
	}
	if lit, ok := syntax.Unparen(c.Fun).(*syntax.FuncLit); ok {
		check.logosRegionEscapedLit(lit, "run with go", lit)
		return
	}
	check.logosRegionEscapedLit(c.Fun, "run with go", c.Fun)
}
