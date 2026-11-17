// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// arenaPointerEscape is correctness::arena_pointer_escape (SPEC Sec 1.2).
var arenaPointerEscape = Rule{
	ID:    "correctness::arena_pointer_escape",
	Group: Correctness,
	Go:    true,
	Check: checkArenaPointerEscape,
}

type addrKind uint8

const (
	addrNone addrKind = iota
	addrUintptr
	addrPointer
)

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

func importsUnsafe(f *syntax.File) bool {
	for _, d := range f.DeclList {
		if imp, ok := d.(*syntax.ImportDecl); ok && imp.Path != nil && imp.Path.Value == `"unsafe"` {
			return true
		}
	}
	return false
}

type frameScan struct {
	pass     *Pass
	file     *syntax.File
	root     syntax.Node
	results  map[*types2.Var]bool
	pointers map[*types2.Var]pointerState
	addrs    map[*types2.Var]*syntax.CallExpr
}

type pointerState uint8

const (
	pointerUnset pointerState = iota
	pointerFrame
	pointerOther
)

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

func (s *frameScan) run() {
	syntax.Inspect(s.root, s.collect)
	syntax.Inspect(s.root, s.scan)
}

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

func (s *frameScan) fieldVar(f *syntax.Field) *types2.Var {
	if f.Name == nil {
		return nil
	}
	return localVar(s.pass, f.Name)
}

func (s *frameScan) noteAssign(a *syntax.AssignStmt) {
	lhs, rhs, ok := assignPairs(a)
	if !ok {
		s.noteTargets(exprList(a.Lhs), nil)
		return
	}
	s.noteTargets(lhs, rhs)
}

func (s *frameScan) noteDecl(d *syntax.VarDecl) {
	if lhs, rhs, ok := declPairs(d); ok {
		s.noteTargets(lhs, rhs)
	}
}

func (s *frameScan) noteTargets(targets, values []syntax.Expr) {
	for i, t := range targets {
		id, ok := syntax.Unparen(t).(*syntax.Name)
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

func (s *frameScan) declare(d *syntax.VarDecl) {
	names, values, ok := declPairs(d)
	if !ok || len(values) != len(names) {
		return
	}
	for i, name := range names {
		s.store(name, values[i])
	}
}

func (s *frameScan) store(lhs, v syntax.Expr) {
	switch {
	case isBlank(lhs):
	case s.withinFrame(lhs):
		s.keep(v)
	default:
		s.leave(v)
	}
}

func (s *frameScan) withinFrame(lhs syntax.Expr) bool {
	v := s.frameRoot(lhs)
	return v != nil && !s.results[v]
}

func (s *frameScan) leaveAll(results syntax.Expr) {
	if results == nil {
		return
	}
	for _, r := range exprList(results) {
		s.leave(r)
	}
}

func (s *frameScan) leave(x syntax.Expr) {
	if kind, call := s.carry(x); kind != addrNone {
		s.report(kind, call)
	}
}

func (s *frameScan) keep(x syntax.Expr) {
	if kind, call := s.carry(x); kind == addrUintptr {
		s.report(kind, call)
	}
}

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

func (s *frameScan) carry(x syntax.Expr) (addrKind, *syntax.CallExpr) {
	switch x := syntax.Unparen(x).(type) {
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

func (s *frameScan) toUintptr(c *syntax.CallExpr) (addrKind, *syntax.CallExpr) {
	if kind, _ := s.carry(c.ArgList[0]); kind != addrNone {
		return addrUintptr, c
	}
	return addrNone, nil
}

func (s *frameScan) carryArith(op *syntax.Operation) (addrKind, *syntax.CallExpr) {
	if op.Y == nil || !isArithmetic(op.Op) {
		return addrNone, nil
	}
	if kind, call := s.carry(op.X); kind != addrNone {
		return kind, call
	}
	return s.carry(op.Y)
}

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

func (s *frameScan) frameAddr(x syntax.Expr) bool {
	if s.directAddr(x) {
		return true
	}
	id, ok := syntax.Unparen(x).(*syntax.Name)
	if !ok {
		return false
	}
	v := localVar(s.pass, id)
	return v != nil && s.pointers[v] == pointerFrame
}

func (s *frameScan) directAddr(x syntax.Expr) bool {
	switch x := syntax.Unparen(x).(type) {
	case *syntax.Operation:
		if x.Y != nil || x.Op != syntax.And {
			return false
		}
		if _, ok := syntax.Unparen(x.X).(*syntax.CompositeLit); ok {
			return true
		}
		return s.frameRoot(x.X) != nil
	case *syntax.CallExpr:
		return callsBuiltin(s.pass, x, "new")
	}
	return false
}

func (s *frameScan) frameRoot(x syntax.Expr) *types2.Var {
	switch x := syntax.Unparen(x).(type) {
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

func underOrNil(t types2.Type) types2.Type {
	if t == nil {
		return nil
	}
	return t.Underlying()
}

func hasBasicKind(t types2.Type, k types2.BasicKind) bool {
	b, ok := underOrNil(t).(*types2.Basic)
	return ok && b.Kind() == k
}

func isArithmetic(op syntax.Operator) bool {
	switch op {
	case syntax.Add, syntax.Sub, syntax.Mul, syntax.Div, syntax.Rem,
		syntax.And, syntax.Or, syntax.Xor, syntax.AndNot, syntax.Shl, syntax.Shr:
		return true
	}
	return false
}
