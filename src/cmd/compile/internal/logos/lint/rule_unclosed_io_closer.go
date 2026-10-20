// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// unclosedIOCloser is correctness::unclosed_io_closer (SPEC Sec 1.2): a local
// variable that holds an io.Closer returned by a call, in a function that
// neither defers its Close nor hands the value to another owner.
//
// A variable is held when an assignment or declaration inside the function
// sets it from a call, directly or through a loGo/s ?, and its type has a
// Close() error method. Parameters, named results, and variables of another
// function are not held; a conversion and a call of a builtin are not calls
// that return an owner. The function closes the variable when a defer
// statement, or a loGo/s scope defer, calls its Close, either directly, with
// or without ?, or inside a deferred function literal. An errdefer does not
// close it: it runs only on failure. The function hands the variable on when
// the variable appears anywhere except as the receiver of a method call, a
// field selection, an operand of == or !=, or the target of an assignment:
// returned, stored, sent, passed as an argument, aliased, taken as a method
// value, or captured by a function literal. One finding is reported for each
// variable, at its first assignment from a call.
var unclosedIOCloser = Rule{
	ID:      "correctness::unclosed_io_closer",
	Group:   Correctness,
	Default: Deny,
	Go:      true,
	Check:   checkUnclosedIOCloser,
}

// checkUnclosedIOCloser is the Check of unclosedIOCloser. Every function
// declaration and function literal is analyzed on its own.
func checkUnclosedIOCloser(pass *Pass) {
	closer := closerInterface()
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if n.Body != nil {
					newCloserScan(pass, closer, n.Body).run()
				}
			case *syntax.FuncLit:
				newCloserScan(pass, closer, n.Body).run()
			}
			return true
		})
	}
}

// closerInterface returns io.Closer, built from its method so that the rule
// does not need io among the imports of the package.
func closerInterface() *types2.Interface {
	errType := types2.Universe.Lookup("error").Type()
	results := types2.NewTuple(types2.NewVar(syntax.Pos{}, nil, "", errType))
	sig := types2.NewSignatureType(nil, nil, nil, nil, results, false)
	return types2.NewInterfaceType([]*types2.Func{types2.NewFunc(syntax.Pos{}, nil, "Close", sig)}, nil)
}

// heldCloser is a variable set from a call: the name at its first such
// assignment, and the call.
type heldCloser struct {
	name *syntax.Name
	call *syntax.CallExpr
}

// closerScan is the analysis of the body of one function.
type closerScan struct {
	pass   *Pass
	closer *types2.Interface
	body   *syntax.BlockStmt
	held   map[*types2.Var]heldCloser
	order  []*types2.Var
	// declared holds the variables declared inside the function, so that a
	// parameter, a named result, and a variable of an enclosing function
	// are not held.
	declared map[*types2.Var]bool
	// plain holds the names whose occurrence neither closes nor hands the
	// variable on.
	plain  map[*syntax.Name]bool
	closed map[*types2.Var]bool
	moved  map[*types2.Var]bool
}

// newCloserScan prepares the analysis of the function with the given body.
func newCloserScan(pass *Pass, closer *types2.Interface, body *syntax.BlockStmt) *closerScan {
	return &closerScan{
		pass:   pass,
		closer: closer,
		body:   body,
		held:   make(map[*types2.Var]heldCloser),
		plain:  make(map[*syntax.Name]bool),
		closed: make(map[*types2.Var]bool),
		moved:  make(map[*types2.Var]bool),
	}
}

// run finds the held variables, then reports each one that is neither closed
// nor handed on.
func (c *closerScan) run() {
	syntax.Inspect(c.body, c.define)
	if len(c.order) == 0 {
		return
	}
	syntax.Inspect(c.body, c.mark)
	syntax.Inspect(c.body, c.discard)
	syntax.Inspect(c.body, c.transfer)
	for _, v := range c.order {
		if c.closed[v] || c.moved[v] {
			continue
		}
		h := c.held[v]
		c.pass.Reportf(h.name, "%s holds an io.Closer returned by %s: defer %s.Close() or hand it to another owner",
			h.name.Value, syntax.String(h.call), h.name.Value)
	}
}

// isCloser reports whether t has a Close() error method. A basic type has
// no method.
func (c *closerScan) isCloser(t types2.Type) bool {
	if _, basic := t.(*types2.Basic); basic {
		return false
	}
	return types2.Implements(t, c.closer)
}

// declaredHere reports whether v is declared inside the function body. The
// declarations are collected when the first candidate asks.
func (c *closerScan) declaredHere(v *types2.Var) bool {
	if c.declared == nil {
		c.declared = make(map[*types2.Var]bool)
		syntax.Inspect(c.body, c.declare)
	}
	return c.declared[v]
}

// declare records the variables the body declares.
func (c *closerScan) declare(n syntax.Node) bool {
	if id, ok := n.(*syntax.Name); ok {
		if v, ok := c.pass.Info.Defs[id].(*types2.Var); ok {
			c.declared[v] = true
		}
	}
	return true
}

// define records the variables a statement sets from a call. A nested
// function is analyzed on its own.
func (c *closerScan) define(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncLit:
		return false
	case *syntax.AssignStmt:
		if lhs, rhs, ok := assignPairs(n); ok {
			c.defineAll(lhs, rhs)
		}
	case *syntax.VarDecl:
		if lhs, rhs, ok := declPairs(n); ok {
			c.defineAll(lhs, rhs)
		}
	}
	return true
}

// defineAll pairs each target with the value it receives, whether the values
// are one for one or one call with several results.
func (c *closerScan) defineAll(lhs, rhs []syntax.Expr) {
	for i, l := range lhs {
		value, t := closerValue(rhs, len(lhs), i)
		c.hold(l, value, t)
	}
}

// closerValue returns the expression and the type that the i-th of n targets
// receives from rhs, or nil when rhs does not pair with the targets.
func closerValue(rhs []syntax.Expr, n, i int) (syntax.Expr, types2.Type) {
	if len(rhs) == n {
		return rhs[i], TypeOf(rhs[i])
	}
	if len(rhs) != 1 {
		return nil, nil
	}
	types := elemTypes(TypeOf(rhs[0]))
	if len(types) != n {
		return nil, nil
	}
	return rhs[0], types[i]
}

// closerCall returns the call that value is, looking through parentheses and
// a loGo/s ?.
func closerCall(value syntax.Expr) *syntax.CallExpr {
	if try, ok := bareExpr(value).(*syntax.TryExpr); ok {
		value = try.X
	}
	return callOf(value)
}

// hold records target as held when value is a call that returns a closer.
func (c *closerScan) hold(target, value syntax.Expr, t types2.Type) {
	id, ok := bareExpr(target).(*syntax.Name)
	if !ok || isBlank(id) || value == nil || t == nil {
		return
	}
	call := closerCall(value)
	if call == nil || convertsType(call) || call.Fun.GetTypeInfo().IsBuiltin() || !c.isCloser(t) {
		return
	}
	v := localVar(c.pass, id)
	if v == nil || !c.declaredHere(v) {
		return
	}
	if _, seen := c.held[v]; !seen {
		c.held[v] = heldCloser{name: id, call: call}
		c.order = append(c.order, v)
	}
}

// mark records the occurrences that are not a transfer and the deferred
// closes. A nested function is left to transfer.
func (c *closerScan) mark(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncLit:
		return false
	case *syntax.AssignStmt:
		c.markTargets(n)
	case *syntax.CallExpr:
		c.markReceiver(n)
	case *syntax.SelectorExpr:
		c.markField(n)
	case *syntax.Operation:
		c.markCompare(n)
	case *syntax.CallStmt:
		if n.Tok == syntax.Defer {
			c.deferred(n.Call)
		}
	case *syntax.ScopeDeferStmt:
		c.deferred(n.Call)
	}
	return true
}

// discard marks the values a blank assignment drops: `_ = r` hands r to
// no owner, so the occurrence is not a transfer. A name nested in a call
// still transfers: `_ = f(r)` passes r on.
func (c *closerScan) discard(n syntax.Node) bool {
	a, ok := n.(*syntax.AssignStmt)
	if !ok {
		return true
	}
	lhs, rhs, ok := assignPairs(a)
	if !ok {
		return true
	}
	for _, l := range lhs {
		if !isBlank(bareExpr(l)) {
			return true
		}
	}
	for _, r := range rhs {
		c.markName(r)
	}
	return true
}

// markTargets marks the variables an assignment sets.
func (c *closerScan) markTargets(a *syntax.AssignStmt) {
	for _, l := range exprList(a.Lhs) {
		c.markName(l)
	}
}

// markReceiver marks the receiver of a method call.
func (c *closerScan) markReceiver(call *syntax.CallExpr) {
	sel, ok := bareExpr(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return
	}
	if s := c.pass.Info.Selections[sel]; s != nil && s.Kind() == types2.MethodVal {
		c.markName(sel.X)
	}
}

// markField marks the base of a field selection.
func (c *closerScan) markField(sel *syntax.SelectorExpr) {
	if s := c.pass.Info.Selections[sel]; s != nil && s.Kind() == types2.FieldVal {
		c.markName(sel.X)
	}
}

// markCompare marks the operands of an equality comparison.
func (c *closerScan) markCompare(op *syntax.Operation) {
	if op.Y == nil || (op.Op != syntax.Eql && op.Op != syntax.Neq) {
		return
	}
	c.markName(op.X)
	c.markName(op.Y)
}

// markName marks e when it is a plain name.
func (c *closerScan) markName(e syntax.Expr) {
	if id, ok := bareExpr(e).(*syntax.Name); ok {
		c.plain[id] = true
	}
}

// deferred records the variables closed by a deferred call: x.Close() itself,
// or any x.Close() inside a deferred function literal.
func (c *closerScan) deferred(x syntax.Expr) {
	call := closerCall(x)
	if call == nil {
		return
	}
	lit, ok := bareExpr(call.Fun).(*syntax.FuncLit)
	if !ok {
		c.closes(call)
		return
	}
	syntax.Inspect(lit.Body, func(n syntax.Node) bool {
		if inner, ok := n.(*syntax.CallExpr); ok {
			c.closes(inner)
		}
		return true
	})
}

// closes records the variable that call closes, when call is x.Close().
func (c *closerScan) closes(call *syntax.CallExpr) {
	sel, ok := bareExpr(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Close" || len(call.ArgList) != 0 {
		return
	}
	id, ok := bareExpr(sel.X).(*syntax.Name)
	if !ok {
		return
	}
	if v := localVar(c.pass, id); v != nil {
		c.closed[v] = true
	}
}

// transfer marks the held variables that the function hands on: any
// occurrence that is not plain, and every occurrence inside a nested function.
func (c *closerScan) transfer(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncLit:
		syntax.Inspect(n.Body, c.capture)
		return false
	case *syntax.Name:
		if !c.plain[n] {
			c.move(n)
		}
	}
	return true
}

// capture marks the held variable a name in a nested function refers to.
func (c *closerScan) capture(n syntax.Node) bool {
	if id, ok := n.(*syntax.Name); ok {
		c.move(id)
	}
	return true
}

// move marks the variable id uses as handed on, when it is held.
func (c *closerScan) move(id *syntax.Name) {
	if v, ok := c.pass.Info.Uses[id].(*types2.Var); ok {
		if _, held := c.held[v]; held {
			c.moved[v] = true
		}
	}
}
