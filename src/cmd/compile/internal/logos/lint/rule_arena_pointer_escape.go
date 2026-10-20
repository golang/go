// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// arenaPointerEscape is correctness::arena_pointer_escape (SPEC Sec 1.2): the
// address of a frame object, kept past the frame through an unsafe address.
// A frame object is storage the compiler allocates on the stack or in the
// goroutine arena: a local variable, a field of a local struct, an element of
// a local array, and the result of new or of an address-of composite
// literal. A local variable that every assignment in the function sets to
// such an address, or to unsafe.Pointer of one, stands for it as well. A
// slice element, a map entry, and anything reached through a pointer is not
// frame storage.
//
// Two forms are reported. An integer form, uintptr(unsafe.Pointer(&x)), is
// reported wherever the value is stored, including arithmetic on it: an
// integer keeps nothing alive and does not follow the object when its frame
// moves or dies. A pointer form, unsafe.Pointer(&x), is reported only when it
// leaves the frame: returned, sent, assigned to a named result, or stored in
// a location that is not frame storage. A pointer form held in a local
// variable stays in the frame and is exempt. A conversion back, as in
// unsafe.Pointer(uintptr(unsafe.Pointer(&x)) + off), turns an integer form
// into a pointer form. A call argument, a comparison, and an assignment to
// the blank identifier store nothing. The analysis reads the syntax tree and
// the checker's objects only, so it holds for Go and loGo/s packages alike.
var arenaPointerEscape = Rule{
	ID:    "correctness::arena_pointer_escape",
	Group: Correctness,
	Go:    true,
	Check: checkArenaPointerEscape,
}

// addrKind says how an expression carries the address of a frame object.
type addrKind uint8

const (
	addrNone addrKind = iota
	addrUintptr
	addrPointer
)

// checkArenaPointerEscape is the Check of arenaPointerEscape. Every function
// declaration, and every function literal outside one, is analyzed with the
// literals nested in it. A file that does not import unsafe has no unsafe
// address to store.
func checkArenaPointerEscape(pass *Pass) {
	for _, f := range pass.Files {
		if !importsUnsafe(f) {
			continue
		}
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if n.Body != nil {
					newFrameScan(pass, f, n).run()
				}
				return false
			case *syntax.FuncLit:
				newFrameScan(pass, f, n).run()
				return false
			}
			return true
		})
	}
}

// importsUnsafe reports whether f imports package unsafe.
func importsUnsafe(f *syntax.File) bool {
	for _, d := range f.DeclList {
		if imp, ok := d.(*syntax.ImportDecl); ok && imp.Path != nil && imp.Path.Value == `"unsafe"` {
			return true
		}
	}
	return false
}

// frameScan is the analysis of one function and the literals inside it.
type frameScan struct {
	pass *Pass
	file *syntax.File
	root syntax.Node
	// results holds the named results of the function and of its literals.
	results map[*types2.Var]bool
	// pointers records what the assignments to each local variable set it
	// to.
	pointers map[*types2.Var]pointerState
	// addrs records the pointer-form conversion each frame-address
	// variable was assigned, so a later use reports that conversion.
	addrs map[*types2.Var]*syntax.CallExpr
}

// pointerState is what every assignment seen so far sets a variable to.
type pointerState uint8

const (
	pointerUnset pointerState = iota
	pointerFrame
	pointerOther
)

// newFrameScan prepares the analysis of the function rooted at root.
func newFrameScan(pass *Pass, f *syntax.File, root syntax.Node) *frameScan {
	return &frameScan{
		pass:     pass,
		file:     f,
		root:     root,
		results:  make(map[*types2.Var]bool),
		pointers: make(map[*types2.Var]pointerState),
		addrs:    make(map[*types2.Var]*syntax.CallExpr),
	}
}

// run collects the named results and the address variables, then reports
// every store of a frame address.
func (s *frameScan) run() {
	syntax.Inspect(s.root, s.collect)
	syntax.Inspect(s.root, s.scan)
}

// collect records named results and the assignments to local variables.
func (s *frameScan) collect(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncType:
		s.noteSignature(n)
	case *syntax.AssignStmt:
		s.noteAssign(n)
	case *syntax.VarDecl:
		s.noteDecl(n)
	case *syntax.RangeClause:
		s.noteTargets(exprList(n.Lhs), nil)
	}
	return true
}

// noteSignature records the named results of a function type, and marks its
// parameters as holding values from the caller.
func (s *frameScan) noteSignature(ft *syntax.FuncType) {
	for _, f := range ft.ParamList {
		if v := s.fieldVar(f); v != nil {
			s.pointers[v] = pointerOther
		}
	}
	for _, f := range ft.ResultList {
		if v := s.fieldVar(f); v != nil {
			s.results[v] = true
		}
	}
}

// fieldVar returns the variable a parameter or result declares, or nil for
// an unnamed one.
func (s *frameScan) fieldVar(f *syntax.Field) *types2.Var {
	if f.Name == nil {
		return nil
	}
	return localVar(s.pass, f.Name)
}

// noteAssign records the targets of an assignment statement.
func (s *frameScan) noteAssign(a *syntax.AssignStmt) {
	lhs, rhs, ok := assignPairs(a)
	if !ok {
		s.noteTargets(exprList(a.Lhs), nil)
		return
	}
	s.noteTargets(lhs, rhs)
}

// noteDecl records the names of a variable declaration with values.
func (s *frameScan) noteDecl(d *syntax.VarDecl) {
	if lhs, rhs, ok := declPairs(d); ok {
		s.noteTargets(lhs, rhs)
	}
}

// noteTargets records each variable of targets as set to the matching value
// of values. A variable every assignment sets to a frame address, directly
// or through unsafe.Pointer of one, stands for that address; the conversion
// of the pointer form is kept so a later use reports it. Without a matching
// value, or after any other assignment, the variable holds something else.
func (s *frameScan) noteTargets(targets, values []syntax.Expr) {
	for i, t := range targets {
		id, ok := bareExpr(t).(*syntax.Name)
		if !ok {
			continue
		}
		v := localVar(s.pass, id)
		if v == nil {
			continue
		}
		delete(s.addrs, v)
		if len(values) != len(targets) || s.pointers[v] == pointerOther {
			s.pointers[v] = pointerOther
			continue
		}
		if kind, call := s.carry(values[i]); kind == addrPointer {
			s.pointers[v] = pointerFrame
			s.addrs[v] = call
		} else if s.directAddr(values[i]) {
			s.pointers[v] = pointerFrame
		} else {
			s.pointers[v] = pointerOther
		}
	}
}

// scan reports the stores of a frame address in one node.
func (s *frameScan) scan(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		s.assign(n)
	case *syntax.VarDecl:
		s.declare(n)
	case *syntax.ReturnStmt:
		s.leaveAll(n.Results)
	case *syntax.SendStmt:
		s.leave(n.Value)
	case *syntax.LambdaExpr:
		if body, ok := n.Body.(syntax.Expr); ok {
			s.leave(body)
		}
	}
	return true
}

// assign checks the pairs of an assignment statement.
func (s *frameScan) assign(a *syntax.AssignStmt) {
	if a.Rhs == nil {
		return
	}
	lhs, rhs := exprList(a.Lhs), exprList(a.Rhs)
	if len(lhs) != len(rhs) {
		return
	}
	for i, l := range lhs {
		s.store(l, rhs[i])
	}
}

// declare checks the pairs of a variable declaration inside a function.
func (s *frameScan) declare(d *syntax.VarDecl) {
	names, values, ok := declPairs(d)
	if !ok || len(values) != len(names) {
		return
	}
	for i, name := range names {
		s.store(name, values[i])
	}
}

// store checks the value v stored in the target lhs.
func (s *frameScan) store(lhs, v syntax.Expr) {
	switch {
	case isBlank(lhs):
	case s.withinFrame(lhs):
		s.keep(v)
	default:
		s.leave(v)
	}
}

// withinFrame reports whether lhs is frame storage that is not a named
// result.
func (s *frameScan) withinFrame(lhs syntax.Expr) bool {
	v := s.frameRoot(lhs)
	return v != nil && !s.results[v]
}

// leaveAll checks every result of a return statement.
func (s *frameScan) leaveAll(results syntax.Expr) {
	if results == nil {
		return
	}
	for _, r := range exprList(results) {
		s.leave(r)
	}
}

// leave reports a value that leaves the frame and carries a frame address of
// either form.
func (s *frameScan) leave(x syntax.Expr) {
	if kind, call := s.carry(x); kind != addrNone {
		s.report(kind, call)
	}
}

// keep reports a value stored in the frame that carries the integer form.
func (s *frameScan) keep(x syntax.Expr) {
	if kind, call := s.carry(x); kind == addrUintptr {
		s.report(kind, call)
	}
}

// report records the finding for the conversion call.
func (s *frameScan) report(kind addrKind, call *syntax.CallExpr) {
	text := syntax.String(call)
	msg := text + " keeps the address of a frame object as an integer the compiler cannot track; keep an unsafe.Pointer"
	if kind == addrPointer {
		msg = text + " takes the address of a frame object and stores it outside its frame"
	}
	s.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(call),
		End:     s.pass.CallEnd(s.file, call),
		Message: msg,
	})
}

// carry returns how x carries a frame address, and the conversion call that
// does so. A local that every assignment sets through unsafe.Pointer of a
// frame address carries that conversion past its declaration.
func (s *frameScan) carry(x syntax.Expr) (addrKind, *syntax.CallExpr) {
	switch x := bareExpr(x).(type) {
	case *syntax.Name:
		if v := localVar(s.pass, x); v != nil && s.pointers[v] == pointerFrame {
			if call := s.addrs[v]; call != nil {
				return addrPointer, call
			}
		}
		return addrNone, nil
	case *syntax.CallExpr:
		return s.carryConversion(x)
	case *syntax.Operation:
		return s.carryArith(x)
	case *syntax.CompositeLit:
		return s.carryElems(x)
	}
	return addrNone, nil
}

// carryConversion classifies a conversion to unsafe.Pointer or to uintptr.
func (s *frameScan) carryConversion(c *syntax.CallExpr) (addrKind, *syntax.CallExpr) {
	if !convertsType(c) {
		return addrNone, nil
	}
	t := TypeOf(c)
	switch {
	case hasBasicKind(t, types2.UnsafePointer):
		return s.toPointer(c)
	case hasBasicKind(t, types2.Uintptr):
		return s.toUintptr(c)
	}
	return addrNone, nil
}

// toPointer classifies unsafe.Pointer(arg): a frame address, or an integer
// form turned back into a pointer.
func (s *frameScan) toPointer(c *syntax.CallExpr) (addrKind, *syntax.CallExpr) {
	arg := c.ArgList[0]
	if s.frameAddr(arg) {
		return addrPointer, c
	}
	if kind, _ := s.carry(arg); kind != addrNone {
		return addrPointer, c
	}
	return addrNone, nil
}

// toUintptr classifies uintptr(arg): an integer form when arg carries any.
func (s *frameScan) toUintptr(c *syntax.CallExpr) (addrKind, *syntax.CallExpr) {
	if kind, _ := s.carry(c.ArgList[0]); kind != addrNone {
		return addrUintptr, c
	}
	return addrNone, nil
}

// carryArith classifies arithmetic on an integer form.
func (s *frameScan) carryArith(op *syntax.Operation) (addrKind, *syntax.CallExpr) {
	if op.Y == nil || !isArithmetic(op.Op) {
		return addrNone, nil
	}
	if kind, call := s.carry(op.X); kind != addrNone {
		return kind, call
	}
	return s.carry(op.Y)
}

// carryElems classifies the elements of a composite literal.
func (s *frameScan) carryElems(lit *syntax.CompositeLit) (addrKind, *syntax.CallExpr) {
	for _, e := range lit.ElemList {
		if kv, ok := e.(*syntax.KeyValueExpr); ok {
			e = kv.Value
		}
		if kind, call := s.carry(e); kind != addrNone {
			return kind, call
		}
	}
	return addrNone, nil
}

// frameAddr reports whether x is the address of a frame object, or a
// variable that holds one.
func (s *frameScan) frameAddr(x syntax.Expr) bool {
	if s.directAddr(x) {
		return true
	}
	id, ok := bareExpr(x).(*syntax.Name)
	if !ok {
		return false
	}
	v := localVar(s.pass, id)
	return v != nil && s.pointers[v] == pointerFrame
}

// directAddr reports whether x is &y for frame storage y, &T{...}, or a
// call of new.
func (s *frameScan) directAddr(x syntax.Expr) bool {
	switch x := bareExpr(x).(type) {
	case *syntax.Operation:
		if x.Y != nil || x.Op != syntax.And {
			return false
		}
		if _, ok := bareExpr(x.X).(*syntax.CompositeLit); ok {
			return true
		}
		return s.frameRoot(x.X) != nil
	case *syntax.CallExpr:
		return callsBuiltin(s.pass, x, "new")
	}
	return false
}

// frameRoot returns the local variable whose frame storage x denotes: x names the
// variable, a field of it through no pointer, or an element of it that is an
// array. It returns nil for any other expression.
func (s *frameScan) frameRoot(x syntax.Expr) *types2.Var {
	switch x := bareExpr(x).(type) {
	case *syntax.Name:
		return localVar(s.pass, x)
	case *syntax.SelectorExpr:
		if sel := s.pass.Info.Selections[x]; sel != nil && sel.Kind() == types2.FieldVal && !sel.Indirect() {
			return s.frameRoot(x.X)
		}
	case *syntax.IndexExpr:
		if _, ok := underOrNil(TypeOf(x.X)).(*types2.Array); ok {
			return s.frameRoot(x.X)
		}
	}
	return nil
}

// underOrNil returns the underlying type of t, or nil.
func underOrNil(t types2.Type) types2.Type {
	if t == nil {
		return nil
	}
	return t.Underlying()
}

// hasBasicKind reports whether the underlying type of t is the basic kind k.
func hasBasicKind(t types2.Type, k types2.BasicKind) bool {
	b, ok := underOrNil(t).(*types2.Basic)
	return ok && b.Kind() == k
}

// isArithmetic reports whether op is an arithmetic or bitwise operator.
func isArithmetic(op syntax.Operator) bool {
	switch op {
	case syntax.Add, syntax.Sub, syntax.Mul, syntax.Div, syntax.Rem,
		syntax.And, syntax.Or, syntax.Xor, syntax.AndNot, syntax.Shl, syntax.Shr:
		return true
	}
	return false
}
