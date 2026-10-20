// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"maps"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
)

// nilFacts is the set of paths a nil test has proved to hold a non-nil value
// at one point of a function. A path is an identifier, or a chain of field
// selections from one, in source form: p, h.P, os.Stdout. The facts follow
// the structured control flow of a function body: an if or for condition
// proves a path inside the guarded block, an early exit on a nil test proves
// it for the statements after the if, and an assignment to a path, or to a
// path it extends, ends the proof. A nilFacts is never mutated after it is
// built, and never nil.
type nilFacts map[string]bool

// union returns the facts of f and g together.
func (f nilFacts) union(g nilFacts) nilFacts {
	out := maps.Clone(f)
	maps.Copy(out, g)
	return out
}

// intersect returns the facts that both f and g hold.
func (f nilFacts) intersect(g nilFacts) nilFacts {
	out := nilFacts{}
	for k := range f {
		if g[k] {
			out[k] = true
		}
	}
	return out
}

// without returns the facts that survive the assignments ks.
func (f nilFacts) without(ks []kill) nilFacts {
	if len(ks) == 0 {
		return f
	}
	out := nilFacts{}
	for k := range f {
		if !slices.ContainsFunc(ks, func(x kill) bool { return x.ends(k) }) {
			out[k] = true
		}
	}
	return out
}

// kill is the path an assignment writes. self tells whether the path itself
// changes, or only the paths below it: an element or pointer store leaves a
// slice or pointer variable as it was.
type kill struct {
	key  string
	self bool
}

// ends reports whether the write ends the proof for path.
func (k kill) ends(path string) bool {
	return (k.self && path == k.key) || strings.HasPrefix(path, k.key+".")
}

// pathKey returns the source form of the path e, or "" when e is not a path.
func pathKey(e syntax.Expr) string {
	switch e := e.(type) {
	case *syntax.ParenExpr:
		return pathKey(e.X)
	case *syntax.Name:
		if e.Value == "_" || e.Value == "nil" {
			return ""
		}
		return e.Value
	case *syntax.SelectorExpr:
		if base := pathKey(e.X); base != "" {
			return base + "." + e.Sel.Value
		}
	}
	return ""
}

// condFacts returns the facts that hold when cond is true and when it is
// false. Only comparisons of a path with nil, joined by !, &&, and ||, prove
// anything.
func condFacts(cond syntax.Expr) (whenTrue, whenFalse nilFacts) {
	switch c := cond.(type) {
	case *syntax.ParenExpr:
		return condFacts(c.X)
	case *syntax.Operation:
		return operationFacts(c)
	}
	return nilFacts{}, nilFacts{}
}

// operationFacts is condFacts for an operation. The facts of a false &&, and
// of a true ||, are the facts both operands share, which is sound without
// tracking which operand decided.
func operationFacts(op *syntax.Operation) (whenTrue, whenFalse nilFacts) {
	switch {
	case op.Y == nil && op.Op == syntax.Not:
		whenFalse, whenTrue = condFacts(op.X)
	case op.Op == syntax.AndAnd:
		tx, fx := condFacts(op.X)
		ty, fy := condFacts(op.Y)
		whenTrue, whenFalse = tx.union(ty), fx.intersect(fy)
	case op.Op == syntax.OrOr:
		tx, fx := condFacts(op.X)
		ty, fy := condFacts(op.Y)
		whenTrue, whenFalse = tx.intersect(ty), fx.union(fy)
	case op.Op == syntax.Eql || op.Op == syntax.Neq:
		return nilComparison(op)
	default:
		return nilFacts{}, nilFacts{}
	}
	return whenTrue, whenFalse
}

// nilComparison returns the facts of x == nil or x != nil, in either operand
// order, for a path x.
func nilComparison(op *syntax.Operation) (whenTrue, whenFalse nilFacts) {
	key := pathKey(op.X)
	if isNilName(op.X) {
		key = pathKey(op.Y)
	} else if !isNilName(op.Y) {
		key = ""
	}
	if key == "" {
		return nilFacts{}, nilFacts{}
	}
	present := nilFacts{key: true}
	if op.Op == syntax.Neq {
		return present, nilFacts{}
	}
	return nilFacts{}, present
}

// isNilName reports whether e is the identifier nil.
func isNilName(e syntax.Expr) bool {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && n.Value == "nil"
}

// kills returns every write to a path inside n, function literals included:
// a closure that assigns a path may run at any time.
func kills(n syntax.Node) []kill {
	var out []kill
	syntax.Inspect(n, func(m syntax.Node) bool {
		out = append(out, writes(m)...)
		return true
	})
	return out
}

// writes returns the paths the node n itself assigns or declares.
func writes(n syntax.Node) []kill {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		return targetKills(exprList(n.Lhs))
	case *syntax.RangeClause:
		if n.Lhs != nil {
			return targetKills(exprList(n.Lhs))
		}
	case *syntax.VarDecl:
		var out []kill
		for _, name := range n.NameList {
			out = append(out, kill{key: name.Value, self: true})
		}
		return out
	}
	return nil
}

// targetKills returns the writes of the assignment targets list.
func targetKills(list []syntax.Expr) []kill {
	var out []kill
	for _, e := range list {
		if key := pathKey(e); key != "" {
			out = append(out, kill{key: key, self: true})
		} else if key := storeBase(e); key != "" {
			out = append(out, kill{key: key})
		}
	}
	return out
}

// storeBase returns the path an element or pointer store writes through, or
// "" when the target is not such a store.
func storeBase(e syntax.Expr) string {
	switch e := e.(type) {
	case *syntax.IndexExpr:
		return pathKey(e.X)
	case *syntax.Operation:
		if e.Y == nil {
			return pathKey(e.X)
		}
	}
	return ""
}

// flow walks the statements of one function body in control-flow order with
// the nilFacts that hold before each, and calls onReturn for every return
// statement. A function literal is a separate function and is not entered:
// it may run when the facts of its creator no longer hold.
type flow struct {
	pass     *Pass
	onReturn func(*syntax.ReturnStmt, nilFacts)
}

// stmts walks list and returns the facts that hold after it.
func (w *flow) stmts(list []syntax.Stmt, in nilFacts) nilFacts {
	for _, s := range list {
		in = w.stmt(s, in)
	}
	return in
}

// stmt walks s and returns the facts that hold after it.
func (w *flow) stmt(s syntax.Stmt, in nilFacts) nilFacts {
	switch s := s.(type) {
	case *syntax.ReturnStmt:
		w.onReturn(s, in)
		return in
	case *syntax.IfStmt:
		return w.ifStmt(s, in)
	case *syntax.ForStmt:
		return w.forStmt(s, in)
	case *syntax.BlockStmt:
		out := w.stmts(s.List, in)
		return out.without(kills(s))
	case *syntax.LabeledStmt:
		return w.stmt(s.Stmt, in)
	}
	return w.opaque(s, in)
}

// opaque walks a statement that proves nothing itself, such as a switch, a
// select, or an assignment. Every path written anywhere inside it is dropped
// before its nested statements are walked.
func (w *flow) opaque(s syntax.Stmt, in nilFacts) nilFacts {
	out := in.without(kills(s))
	syntax.Inspect(s, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.ReturnStmt:
			w.onReturn(n, out)
			return false
		case *syntax.BlockStmt:
			w.stmts(n.List, out)
			return false
		case *syntax.CaseClause:
			w.stmts(n.Body, out)
			return false
		case *syntax.CommClause:
			w.stmts(n.Body, out)
			return false
		}
		return true
	})
	return out
}

// ifStmt walks an if statement. The facts of its condition hold in the then
// branch, the facts of the negated condition hold in the else branch, and a
// branch that cannot fall through leaves the facts of the other one behind.
func (w *flow) ifStmt(s *syntax.IfStmt, in nilFacts) nilFacts {
	if s.Init != nil {
		in = w.stmt(s.Init, in)
	}
	whenTrue, whenFalse := condFacts(s.Cond)
	w.stmts(s.Then.List, in.union(whenTrue))
	if s.Else != nil {
		w.stmt(s.Else, in.union(whenFalse))
	}
	after := in
	switch thenExits, elseExits := w.exits(s.Then.List), w.exitsStmt(s.Else); {
	case thenExits && !elseExits:
		after = in.union(whenFalse)
	case elseExits && !thenExits:
		after = in.union(whenTrue)
	}
	return after.without(kills(s))
}

// forStmt walks a for statement. Its condition holds inside the body; a path
// the loop writes is proved nowhere in it, because the writes may come from a
// previous iteration.
func (w *flow) forStmt(s *syntax.ForStmt, in nilFacts) nilFacts {
	out := in.without(kills(s))
	body := out
	if s.Cond != nil {
		whenTrue, _ := condFacts(s.Cond)
		body = out.union(whenTrue)
	}
	w.stmts(s.Body.List, body)
	return out
}

// exits reports whether the last statement of list leaves the enclosing
// block: a return, a branch, a panic, or a call that ends the process.
func (w *flow) exits(list []syntax.Stmt) bool {
	return len(list) > 0 && w.exitsStmt(list[len(list)-1])
}

// exitsStmt reports whether s leaves the enclosing block on every path.
func (w *flow) exitsStmt(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.ReturnStmt, *syntax.BranchStmt:
		return true
	case *syntax.ExprStmt:
		call := callOf(s.X)
		return call != nil && endsFlow(w.pass, call)
	case *syntax.BlockStmt:
		return w.exits(s.List)
	case *syntax.IfStmt:
		return s.Else != nil && w.exits(s.Then.List) && w.exitsStmt(s.Else)
	case *syntax.LabeledStmt:
		return w.exitsStmt(s.Stmt)
	}
	return false
}

// endsFlow reports whether call is a call of panic or of a function that
// never returns normally.
func endsFlow(pass *Pass, call *syntax.CallExpr) bool {
	if name, ok := call.Fun.(*syntax.Name); ok && name.Value == "panic" {
		return true
	}
	fn := calleeFunc(pass, call)
	if fn == nil {
		return false
	}
	full := fn.FullName()
	return slices.ContainsFunc(terminators, func(t string) bool { return strings.HasPrefix(full, t) })
}
