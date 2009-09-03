// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"os";
	"go/ast";
	"go/parser";
)

// TODO: Make CompileExpr and CompileStmts 
// methods on World.

type World struct {
	scope *Scope;
	frame *Frame;
}

func NewWorld() (*World) {
	w := new(World);
	w.scope = universe.ChildScope();
	w.scope.offset = -1;	// this block's vars allocate directly
	w.scope.numVars = 1;	// inner blocks have frames: offset+numVars >= 0
	return w;
}


type Code struct {
	w *World;
	stmt *Stmt;
	expr *Expr;
}

func (w *World) Compile(text string) (*Code, os.Error) {
	asts, err := parser.ParseStmtList("input", text);
	if err != nil {
		return nil, err;
	}
	if len(asts) == 1 {
		if s, ok := asts[0].(*ast.ExprStmt); ok {
			expr, err := CompileExpr(w.scope, s.X);
			if err != nil {
				return nil, err;
			}
			return &Code{w: w, expr: expr}, nil;
		}
	}
	stmt, err := CompileStmts(w.scope, asts);
	if err != nil {
		return nil, err;
	}
	return &Code{w: w, stmt: stmt}, nil;
}

func (c *Code) Run() (Value, os.Error) {
	w := c.w;
	w.frame = w.scope.NewFrame(nil);
	if c.stmt != nil {
		return nil, c.stmt.Exec(w.frame);
	}
	val, err := c.expr.Eval(w.frame);
	return val, err;
}

