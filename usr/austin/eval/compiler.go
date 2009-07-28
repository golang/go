// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"fmt";
	"go/ast";
	"go/scanner";
	"go/token";
)


type positioned interface {
	Pos() token.Position;
}


// A compiler captures information used throughout an entire
// compilation.  Currently it includes only the error handler.
//
// TODO(austin) This might actually represent package level, in which
// case it should be package compiler.
type compiler struct {
	errors scanner.ErrorHandler;
}

func (a *compiler) diagAt(pos positioned, format string, args ...) {
	a.errors.Error(pos.Pos(), fmt.Sprintf(format, args));
}

type FuncDecl struct
func (a *compiler) compileFunc(scope *Scope, decl *FuncDecl, body *ast.BlockStmt) (func (f *Frame) Func)
type exprCompiler struct
func (a *compiler) compileExpr(scope *Scope, expr ast.Expr, constant bool) *exprCompiler
type assignCompiler struct
func (a *compiler) checkAssign(pos token.Position, rs []*exprCompiler, errOp, errPosName string) (*assignCompiler, bool)
func (a *compiler) compileAssign(pos token.Position, lt Type, rs []*exprCompiler, errOp, errPosName string) (func(lv Value, f *Frame))
func (a *compiler) compileType(scope *Scope, typ ast.Expr) Type
func (a *compiler) compileFuncType(scope *Scope, typ *ast.FuncType) *FuncDecl

func (a *compiler) compileArrayLen(scope *Scope, expr ast.Expr) (int64, bool)


type codeBuf struct
type FuncType struct
// A funcCompiler captures information used throughout the compilation
// of a single function body.
type funcCompiler struct {
	*compiler;
	fnType *FuncType;
	// Whether the out variables are named.  This affects what
	// kinds of return statements are legal.
	outVarsNamed bool;
	*codeBuf;
	err bool;
}


// A blockCompiler captures information used throughout the compilation
// of a single block within a function.
type blockCompiler struct {
	*funcCompiler;
	scope *Scope;
	returned bool;
}

func (a *blockCompiler) compileBlock(body *ast.BlockStmt)


// An exprContext stores information used throughout the compilation
// of a single expression.  It does not embed funcCompiler because
// expressions can appear at top level.
//
// TODO(austin) Rename exprCompiler to exprNodeCompiler and rename
// this to exprCompiler.
type exprContext struct {
	*compiler;
	scope *Scope;
	constant bool;
}
