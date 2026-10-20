// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmp"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// bareExpr returns e without its enclosing parentheses.
func bareExpr(e syntax.Expr) syntax.Expr {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// elemTypes returns the types of the values of an expression of type t: the
// members of a tuple, or t itself. A nil t has none.
func elemTypes(t types2.Type) []types2.Type {
	if t == nil {
		return nil
	}
	tuple, ok := t.(*types2.Tuple)
	if !ok {
		return []types2.Type{t}
	}
	out := make([]types2.Type, tuple.Len())
	for i := range out {
		out[i] = tuple.At(i).Type()
	}
	return out
}

// assignPairs returns the targets and the values of an assignment or short
// declaration. It reports false for a statement with no value, such as x++,
// and for a compound assignment.
func assignPairs(a *syntax.AssignStmt) (lhs, rhs []syntax.Expr, ok bool) {
	if a.Rhs == nil || (a.Op != 0 && a.Op != syntax.Def) {
		return nil, nil, false
	}
	return exprList(a.Lhs), exprList(a.Rhs), true
}

// declPairs returns the names and the values of a variable declaration. It
// reports false for a declaration with no value.
func declPairs(d *syntax.VarDecl) (lhs, rhs []syntax.Expr, ok bool) {
	if d.Values == nil {
		return nil, nil, false
	}
	names := make([]syntax.Expr, len(d.NameList))
	for i, n := range d.NameList {
		names[i] = n
	}
	return names, exprList(d.Values), true
}

// callsBuiltin reports whether call calls the predeclared function name.
func callsBuiltin(pass *Pass, call *syntax.CallExpr, name string) bool {
	id, ok := bareExpr(call.Fun).(*syntax.Name)
	if !ok || id.Value != name {
		return false
	}
	_, ok = pass.Info.Uses[id].(*types2.Builtin)
	return ok
}

// convertsType reports whether call converts its argument to a type.
func convertsType(call *syntax.CallExpr) bool {
	return len(call.ArgList) == 1 && call.Fun.GetTypeInfo().IsType()
}

// localVar returns the variable id names or declares, when it is a
// variable of a function: not a field and not a package-level variable.
func localVar(pass *Pass, id *syntax.Name) *types2.Var {
	v, ok := cmp.Or(pass.Info.Defs[id], pass.Info.Uses[id]).(*types2.Var)
	if !ok || v.IsField() || v.Pkg() == nil || v.Parent() == nil || v.Parent() == v.Pkg().Scope() {
		return nil
	}
	return v
}
