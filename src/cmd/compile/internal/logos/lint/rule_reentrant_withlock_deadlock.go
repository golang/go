// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// logosSyncPath is the import path of the loGo/s synchronization package.
const logosSyncPath = "logos/sync"

// reentrantWithLockDeadlock is concurrency::reentrant_withlock_deadlock
// (SPEC Sec 1.2, #167): inside the function literal or lambda passed to
// WithLock of a logos/sync.Locked, a call to WithLock on the same receiver
// expression, which blocks forever on the mutex the outer call holds. The
// inner call counts when it appears directly in the callback, in the
// callback of another WithLock call inside it, or in the body of a
// same-package function or method that the callback calls, one call deep.
// A function literal that is not itself a WithLock callback is skipped:
// it may run later, after the lock is released. Two receiver expressions
// are the same when they name the same variable through the same chain of
// field selections, ignoring address-of and dereference; a receiver that is
// any other expression never matches, so the rule fails safe. The analysis
// reads syntax and checker objects only, so it holds for Go and loGo/s
// packages alike; a lambda exists in loGo/s only.
var reentrantWithLockDeadlock = Rule{
	ID:    "concurrency::reentrant_withlock_deadlock",
	Group: Concurrency,
	Go:    true,
	Check: checkReentrantWithLock,
}

// lockKey names a receiver expression: the variable at its root and the
// chain of field names selected from it.
type lockKey struct {
	root types2.Object
	path string
}

// lockCall is one call of WithLock on a logos/sync.Locked, on the receiver
// expression recv. Body is the body of its callback when the argument is a
// function literal or a lambda, and nil otherwise.
type lockCall struct {
	call *syntax.CallExpr
	recv syntax.Expr
	key  lockKey
	body syntax.Node
}

// lockCheck is the analysis of one package. File is the file being
// checked.
type lockCheck struct {
	pass      *Pass
	file      *syntax.File
	calls     map[*syntax.CallExpr]lockCall
	callbacks map[syntax.Node]bool
	decls     map[*types2.Func]*syntax.FuncDecl
	reported  map[*syntax.CallExpr]bool
}

// checkReentrantWithLock is the Check of reentrantWithLockDeadlock.
func checkReentrantWithLock(pass *Pass) {
	c := &lockCheck{
		pass:      pass,
		calls:     make(map[*syntax.CallExpr]lockCall),
		callbacks: make(map[syntax.Node]bool),
		decls:     make(map[*types2.Func]*syntax.FuncDecl),
		reported:  make(map[*syntax.CallExpr]bool),
	}
	for _, f := range pass.Files {
		c.index(f)
	}
	for _, f := range pass.Files {
		c.file = f
		syntax.Inspect(f, func(n syntax.Node) bool {
			if call, ok := n.(*syntax.CallExpr); ok {
				c.checkOuter(call)
			}
			return true
		})
	}
}

// index records the WithLock calls, their callbacks, and the function
// declarations of f.
func (c *lockCheck) index(f *syntax.File) {
	for _, d := range f.DeclList {
		if fd, ok := d.(*syntax.FuncDecl); ok && fd.Body != nil {
			if fn, _ := c.pass.Info.Defs[fd.Name].(*types2.Func); fn != nil {
				c.decls[fn] = fd
			}
		}
	}
	syntax.Inspect(f, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			c.noteLockCall(call)
		}
		return true
	})
}

// noteLockCall records call when it calls WithLock on a logos/sync.Locked
// with a receiver expression the rule can match.
func (c *lockCheck) noteLockCall(call *syntax.CallExpr) {
	fn := calleeFunc(c.pass, call)
	sel, ok := unparenExpr(call.Fun).(*syntax.SelectorExpr)
	if fn == nil || !ok || fn.Name() != "WithLock" || fn.Pkg() == nil || fn.Pkg().Path() != logosSyncPath {
		return
	}
	key := c.keyOf(sel.X)
	if key.root == nil {
		return
	}
	lc := lockCall{call: call, recv: sel.X, key: key}
	if len(call.ArgList) == 1 {
		lc.body = callbackBody(call.ArgList[0])
	}
	c.calls[call] = lc
	if lc.body != nil {
		c.callbacks[unparenExpr(call.ArgList[0])] = true
	}
}

// callbackBody returns the body of a function literal or a lambda, or nil.
func callbackBody(e syntax.Expr) syntax.Node {
	switch f := unparenExpr(e).(type) {
	case *syntax.FuncLit:
		return f.Body
	case *syntax.LambdaExpr:
		return f.Body
	}
	return nil
}

// unparenExpr returns e without enclosing parentheses.
func unparenExpr(e syntax.Expr) syntax.Expr {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// checkOuter reports the re-entrant calls inside the callback of call, when
// call is a WithLock call with a callback.
func (c *lockCheck) checkOuter(call *syntax.CallExpr) {
	lc, ok := c.calls[call]
	if !ok || lc.body == nil {
		return
	}
	syntax.Inspect(lc.body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return c.callbacks[n]
		case *syntax.CallExpr:
			c.checkInner(n, lc.key)
		}
		return true
	})
}

// checkInner reports call when it re-enters the lock named key: it is a
// WithLock call on key, or it calls a same-package function whose body is.
func (c *lockCheck) checkInner(call *syntax.CallExpr, key lockKey) {
	if lc, ok := c.calls[call]; ok {
		if lc.key == key {
			c.flag(call, "WithLock on "+syntax.String(lc.recv)+" inside a WithLock callback of the same receiver deadlocks: the mutex is already held")
		}
		return
	}
	fn := calleeFunc(c.pass, call)
	if fn == nil {
		return
	}
	if decl := c.decls[fn.Origin()]; decl != nil && c.reenters(call, decl, key) {
		c.flag(call, "call to "+fn.Name()+" takes WithLock on the receiver of the enclosing WithLock callback and deadlocks: the mutex is already held")
	}
}

// reenters reports whether the body of decl, called by call, runs WithLock
// on the receiver named key.
func (c *lockCheck) reenters(call *syntax.CallExpr, decl *syntax.FuncDecl, key lockKey) bool {
	found := false
	syntax.Inspect(decl.Body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return c.callbacks[n]
		case *syntax.CallExpr:
			lc, ok := c.calls[n]
			found = found || ok && c.substitute(lc.key, call, decl) == key
		}
		return !found
	})
	return found
}

// substitute returns key, a receiver key in the body of decl, as seen from
// call: a key rooted at the receiver or a parameter of decl takes the key of
// the matching operand of call. A key that cannot be mapped has no root.
func (c *lockCheck) substitute(key lockKey, call *syntax.CallExpr, decl *syntax.FuncDecl) lockKey {
	operand := c.operandFor(key.root, call, decl)
	if operand == nil {
		if key.root != nil && c.isDeclared(key.root, decl) {
			return lockKey{}
		}
		return key
	}
	outer := c.keyOf(operand)
	if outer.root == nil {
		return lockKey{}
	}
	return lockKey{root: outer.root, path: outer.path + key.path}
}

// isDeclared reports whether obj is declared inside decl: its receiver, a
// parameter, or a local variable.
func (c *lockCheck) isDeclared(obj types2.Object, decl *syntax.FuncDecl) bool {
	pos := obj.Pos()
	return pos.IsKnown() && within(pos, syntax.StartPos(decl), syntax.EndPos(decl))
}

// operandFor returns the operand of call that binds the receiver or the
// parameter of decl that root names, or nil when root is neither.
func (c *lockCheck) operandFor(root types2.Object, call *syntax.CallExpr, decl *syntax.FuncDecl) syntax.Expr {
	if decl.Recv != nil && c.definedBy(decl.Recv) == root {
		if sel, ok := unparenExpr(call.Fun).(*syntax.SelectorExpr); ok {
			return sel.X
		}
		return nil
	}
	for i, p := range decl.Type.ParamList {
		if c.definedBy(p) == root && i < len(call.ArgList) {
			return call.ArgList[i]
		}
	}
	return nil
}

// definedBy returns the object that a named parameter declares, or nil.
func (c *lockCheck) definedBy(f *syntax.Field) types2.Object {
	if f.Name == nil {
		return nil
	}
	return c.pass.Info.Defs[f.Name]
}

// keyOf returns the key of receiver expression e, or a key with no root
// when e is not a variable, a field selection chain from one, or an
// address-of or dereference of those.
func (c *lockCheck) keyOf(e syntax.Expr) lockKey {
	path := ""
	for e != nil {
		if name, ok := e.(*syntax.Name); ok {
			return lockKey{root: c.variableOf(name), path: path}
		}
		var field string
		e, field = c.keyStep(e)
		if field != "" {
			path = "." + field + path
		}
	}
	return lockKey{}
}

// variableOf returns the variable that name uses, or nil.
func (c *lockCheck) variableOf(name *syntax.Name) types2.Object {
	if v, ok := c.pass.Info.Uses[name].(*types2.Var); ok {
		return v
	}
	return nil
}

// keyStep returns the operand of e and the field e selects from it. An
// expression that cannot continue a key returns a nil operand.
func (c *lockCheck) keyStep(e syntax.Expr) (syntax.Expr, string) {
	switch x := e.(type) {
	case *syntax.ParenExpr:
		return x.X, ""
	case *syntax.Operation:
		if x.Y == nil && (x.Op == syntax.And || x.Op == syntax.Mul) {
			return x.X, ""
		}
	case *syntax.SelectorExpr:
		if sel := c.pass.Info.Selections[x]; sel != nil && sel.Kind() == types2.FieldVal {
			return x.X, x.Sel.Value
		}
	}
	return nil, ""
}

// flag reports call once.
func (c *lockCheck) flag(call *syntax.CallExpr, msg string) {
	if c.reported[call] {
		return
	}
	c.reported[call] = true
	c.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(call),
		End:     c.pass.CallEnd(c.file, call),
		Message: msg,
	})
}
