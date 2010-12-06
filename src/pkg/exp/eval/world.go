// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package is the beginning of an interpreter for Go.
// It can run simple Go programs but does not implement
// interface values or packages.
package eval

import (
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"os"
)

type World struct {
	scope *Scope
	frame *Frame
}

func NewWorld() *World {
	w := new(World)
	w.scope = universe.ChildScope()
	w.scope.global = true // this block's vars allocate directly
	return w
}

type Code interface {
	// The type of the value Run returns, or nil if Run returns nil.
	Type() Type

	// Run runs the code; if the code is a single expression
	// with a value, it returns the value; otherwise it returns nil.
	Run() (Value, os.Error)
}

type stmtCode struct {
	w    *World
	code code
}

func (w *World) CompileStmtList(fset *token.FileSet, stmts []ast.Stmt) (Code, os.Error) {
	if len(stmts) == 1 {
		if s, ok := stmts[0].(*ast.ExprStmt); ok {
			return w.CompileExpr(fset, s.X)
		}
	}
	errors := new(scanner.ErrorVector)
	cc := &compiler{fset, errors, 0, 0}
	cb := newCodeBuf()
	fc := &funcCompiler{
		compiler:     cc,
		fnType:       nil,
		outVarsNamed: false,
		codeBuf:      cb,
		flow:         newFlowBuf(cb),
		labels:       make(map[string]*label),
	}
	bc := &blockCompiler{
		funcCompiler: fc,
		block:        w.scope.block,
	}
	nerr := cc.numError()
	for _, stmt := range stmts {
		bc.compileStmt(stmt)
	}
	fc.checkLabels()
	if nerr != cc.numError() {
		return nil, errors.GetError(scanner.Sorted)
	}
	return &stmtCode{w, fc.get()}, nil
}

func (w *World) CompileDeclList(fset *token.FileSet, decls []ast.Decl) (Code, os.Error) {
	stmts := make([]ast.Stmt, len(decls))
	for i, d := range decls {
		stmts[i] = &ast.DeclStmt{d}
	}
	return w.CompileStmtList(fset, stmts)
}

func (s *stmtCode) Type() Type { return nil }

func (s *stmtCode) Run() (Value, os.Error) {
	t := new(Thread)
	t.f = s.w.scope.NewFrame(nil)
	return nil, t.Try(func(t *Thread) { s.code.exec(t) })
}

type exprCode struct {
	w    *World
	e    *expr
	eval func(Value, *Thread)
}

func (w *World) CompileExpr(fset *token.FileSet, e ast.Expr) (Code, os.Error) {
	errors := new(scanner.ErrorVector)
	cc := &compiler{fset, errors, 0, 0}

	ec := cc.compileExpr(w.scope.block, false, e)
	if ec == nil {
		return nil, errors.GetError(scanner.Sorted)
	}
	var eval func(Value, *Thread)
	switch t := ec.t.(type) {
	case *idealIntType:
		// nothing
	case *idealFloatType:
		// nothing
	default:
		if tm, ok := t.(*MultiType); ok && len(tm.Elems) == 0 {
			return &stmtCode{w, code{ec.exec}}, nil
		}
		eval = genAssign(ec.t, ec)
	}
	return &exprCode{w, ec, eval}, nil
}

func (e *exprCode) Type() Type { return e.e.t }

func (e *exprCode) Run() (Value, os.Error) {
	t := new(Thread)
	t.f = e.w.scope.NewFrame(nil)
	switch e.e.t.(type) {
	case *idealIntType:
		return &idealIntV{e.e.asIdealInt()()}, nil
	case *idealFloatType:
		return &idealFloatV{e.e.asIdealFloat()()}, nil
	}
	v := e.e.t.Zero()
	eval := e.eval
	err := t.Try(func(t *Thread) { eval(v, t) })
	return v, err
}

func (w *World) Compile(fset *token.FileSet, text string) (Code, os.Error) {
	stmts, err := parser.ParseStmtList(fset, "input", text)
	if err == nil {
		return w.CompileStmtList(fset, stmts)
	}

	// Otherwise try as DeclList.
	decls, err1 := parser.ParseDeclList(fset, "input", text)
	if err1 == nil {
		return w.CompileDeclList(fset, decls)
	}

	// Have to pick an error.
	// Parsing as statement list admits more forms,
	// its error is more likely to be useful.
	return nil, err
}

type RedefinitionError struct {
	Name string
	Prev Def
}

func (e *RedefinitionError) String() string {
	res := "identifier " + e.Name + " redeclared"
	pos := e.Prev.Pos()
	if pos.IsValid() {
		// TODO: fix this - currently this code is not reached by the tests
		//       need to get a file set (fset) from somewhere
		//res += "; previous declaration at " + fset.Position(pos).String()
		panic(0)
	}
	return res
}

func (w *World) DefineConst(name string, t Type, val Value) os.Error {
	_, prev := w.scope.DefineConst(name, token.NoPos, t, val)
	if prev != nil {
		return &RedefinitionError{name, prev}
	}
	return nil
}

func (w *World) DefineVar(name string, t Type, val Value) os.Error {
	v, prev := w.scope.DefineVar(name, token.NoPos, t)
	if prev != nil {
		return &RedefinitionError{name, prev}
	}
	v.Init = val
	return nil
}
