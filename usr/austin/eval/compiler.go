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
func (a *compiler) compileFunc(b *block, decl *FuncDecl, body *ast.BlockStmt) (func (f *Frame) Func)
type exprCompiler struct
func (a *compiler) compileExpr(b *block, expr ast.Expr, constant bool) *exprCompiler
type assignCompiler struct
func (a *compiler) checkAssign(pos token.Position, rs []*exprCompiler, errOp, errPosName string) (*assignCompiler, bool)
func (a *compiler) compileAssign(pos token.Position, lt Type, rs []*exprCompiler, errOp, errPosName string) (func(lv Value, f *Frame))
func (a *compiler) compileType(b *block, typ ast.Expr) Type
func (a *compiler) compileFuncType(b *block, typ *ast.FuncType) *FuncDecl

func (a *compiler) compileArrayLen(b *block, expr ast.Expr) (int64, bool)


type label struct {
	name string;
	desc string;
	// The PC goto statements should jump to, or nil if this label
	// cannot be goto'd (such as an anonymous for loop label).
	gotoPC *uint;
	// The PC break statements should jump to, or nil if a break
	// statement is invalid.
	breakPC *uint;
	// The PC continue statements should jump to, or nil if a
	// continue statement is invalid.
	continuePC *uint;
	// The position where this label was resolved.  If it has not
	// been resolved yet, an invalid position.
	resolved token.Position;
	// The position where this label was first jumped to.
	used token.Position;
}

type codeBuf struct
type flowBuf struct
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
	flow *flowBuf;
	labels map[string] *label;
	err bool;
}

func (a *funcCompiler) checkLabels()

// A blockCompiler captures information used throughout the compilation
// of a single block within a function.
type blockCompiler struct {
	*funcCompiler;
	block *block;
	// The label of this block, used for finding break and
	// continue labels.
	label *label;
	// The blockCompiler for the block enclosing this one, or nil
	// for a function-level block.
	parent *blockCompiler;
}

func (a *blockCompiler) compileStmt(s ast.Stmt)
func (a *blockCompiler) compileStmts(body *ast.BlockStmt)
func (a *blockCompiler) enterChild() *blockCompiler
func (a *blockCompiler) exit()

// An exprContext stores information used throughout the compilation
// of a single expression.  It does not embed funcCompiler because
// expressions can appear at top level.
//
// TODO(austin) Rename exprCompiler to exprNodeCompiler and rename
// this to exprCompiler.
type exprContext struct {
	*compiler;
	block *block;
	constant bool;
}
