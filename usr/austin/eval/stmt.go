// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"log";
	"os";
	"go/ast";
	"go/scanner";
	"go/token";
	"strconv";
)

/*
 * Statement compiler
 */

type stmtCompiler struct {
	*blockCompiler;
	pos token.Position;
	// err should be initialized to true before visiting and set
	// to false when the statement is compiled successfully.  The
	// function invoking Visit should or this with
	// blockCompiler.err.  This is less error prone than setting
	// blockCompiler.err on every failure path.
	err bool;
}

func (a *stmtCompiler) diag(format string, args ...) {
	a.diagAt(&a.pos, format, args);
}

/*
 * Statement visitors
 */

func (a *stmtCompiler) DoBadStmt(s *ast.BadStmt) {
	// Do nothing.  Already reported by parser.
}

func (a *stmtCompiler) DoDeclStmt(s *ast.DeclStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoEmptyStmt(s *ast.EmptyStmt) {
	a.err = false;
}

func (a *stmtCompiler) DoLabeledStmt(s *ast.LabeledStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoExprStmt(s *ast.ExprStmt) {
	// TODO(austin) Permit any 0 or more valued function call
	e := a.compileExpr(a.scope, s.X, false);
	if e == nil {
		return;
	}

	if e.exec == nil {
		a.diag("%s cannot be used as expression statement", e.desc);
		return;
	}

	exec := e.exec;
	a.push(func(v *vm) {
		exec(v.f);
	});
	a.err = false;
}

func (a *stmtCompiler) DoIncDecStmt(s *ast.IncDecStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) doAssign(s *ast.AssignStmt) {
	bad := false;

	// Compile right side first so we have the types when
	// compiling the left side and so we don't see definitions
	// made on the left side.
	rs := make([]*exprCompiler, len(s.Rhs));
	for i, re := range s.Rhs {
		rs[i] = a.compileExpr(a.scope, re, false);
		if rs[i] == nil {
			bad = true;
			continue;
		}
	}

	// Check the assignment count
	if len(s.Lhs) != len(s.Rhs) {
		log.Crashf("Unbalanced assignment not implemented %v %v %v", len(s.Lhs), s.Tok, len(s.Rhs));
	}

	// Compile left side and generate assigners
	ls := make([]*exprCompiler, len(s.Lhs));
	as := make([]func(lv Value, f *Frame), len(s.Lhs));
	nDefs := 0;
	for i, le := range s.Lhs {
		errPos := i + 1;
		if len(s.Lhs) == 1 {
			errPos = 0;
		}

		if s.Tok == token.DEFINE {
			// Check that it's an identifier
			ident, ok := le.(*ast.Ident);
			if !ok {
				a.diagAt(le, "left side of := must be a name");
				bad = true;
				// Suppress new defitions errors
				nDefs++;
				continue;
			}

			// Is this simply an assignment?
			if _, ok := a.scope.defs[ident.Value]; ok {
				goto assignment;
			}
			nDefs++;

			if rs[i] == nil {
				// TODO(austin) Define a placeholder.
				continue;
			}

			// Generate assigner and get type
			var lt Type;
			lt, as[i] = mkAssign(nil, rs[i], "assignment", "position", errPos);
			if lt == nil {
				bad = true;
				continue;
			}

			// Define identifier
			v := a.scope.DefineVar(ident.Value, lt);
			if v == nil {
				log.Crashf("Failed to define %s", ident.Value);
			}
		}

	assignment:
		ls[i] = a.compileExpr(a.scope, le, false);
		if ls[i] == nil {
			bad = true;
			continue;
		}

		if ls[i].evalAddr == nil {
			ls[i].diag("cannot assign to %s", ls[i].desc);
			bad = true;
			continue;
		}

		// Generate assigner
		if as[i] == nil {
			var lt Type;
			lt, as[i] = mkAssign(ls[i].t, rs[i], "assignment", "position", errPos);
			if lt == nil {
				bad = true;
				continue;
			}
		}
	}

	// A short variable declaration may redeclare variables
	// provided they were originally declared in the same block
	// with the same type, and at least one of the variables is
	// new.
	if s.Tok == token.DEFINE && nDefs == 0 {
		a.diag("at least one new variable must be declared");
		return;
	}

	if bad {
		return;
	}

	n := len(s.Lhs);
	if n == 1 {
		lf := ls[0].evalAddr;
		assign := as[0];
		a.push(func(v *vm) { assign(lf(v.f), v.f) });
	} else {
		a.push(func(v *vm) {
			temps := make([]Value, n);
			// Assign to temporaries
			for i := 0; i < n; i++ {
				// TODO(austin) Don't capture ls
				temps[i] = ls[i].t.Zero();
				as[i](temps[i], v.f);
			}
			// Copy to destination
			for i := 0; i < n; i++ {
				ls[i].evalAddr(v.f).Assign(temps[i]);
			}
		});
	}
	a.err = false;
}

var assignOpToOp = map[token.Token] token.Token {
	token.ADD_ASSIGN : token.ADD,
	token.SUB_ASSIGN : token.SUB,
	token.MUL_ASSIGN : token.MUL,
	token.QUO_ASSIGN : token.QUO,
	token.REM_ASSIGN : token.REM,

	token.AND_ASSIGN : token.AND,
	token.OR_ASSIGN  : token.OR,
        token.XOR_ASSIGN : token.XOR,
        token.SHL_ASSIGN : token.SHL,
        token.SHR_ASSIGN : token.SHR,
        token.AND_NOT_ASSIGN : token.AND_NOT,
}

func (a *stmtCompiler) doAssignOp(s *ast.AssignStmt) {
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		a.diag("tuple assignment cannot be combined with an arithmetic operation");
		return;
	}

	l := a.compileExpr(a.scope, s.Lhs[0], false);
	r := a.compileExpr(a.scope, s.Rhs[0], false);
	if l == nil || r == nil {
		return;
	}

	if l.evalAddr == nil {
		l.diag("cannot assign to %s", l.desc);
		return;
	}

	effect, l := l.extractEffect();

	binop := r.copy();
	binop.pos = s.TokPos;
	binop.doBinaryExpr(assignOpToOp[s.Tok], l, r);
	if binop.t == nil {
		return;
	}

	_, assign := mkAssign(l.t, binop, "assignment", "", 0);
	if assign == nil {
		return;
	}

	lf := l.evalAddr;
	a.push(func(v *vm) {
		effect(v.f);
		assign(lf(v.f), v.f);
	});
	a.err = false;
}

func (a *stmtCompiler) DoAssignStmt(s *ast.AssignStmt) {
	switch s.Tok {
	case token.ASSIGN, token.DEFINE:
		a.doAssign(s);

	default:
		a.doAssignOp(s);
	}
}

func (a *stmtCompiler) DoGoStmt(s *ast.GoStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoDeferStmt(s *ast.DeferStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoReturnStmt(s *ast.ReturnStmt) {
	// Supress return errors even if we fail to compile this
	// return statement.
	a.returned = true;

	if len(s.Results) == 0 && (len(a.outVars) == 0 || a.outVarsNamed) {
		// Simple case.  Simply exit from the function.
		a.push(func(v *vm) { v.pc = ^uint(0) });
		a.err = false;
		return;
	}

	// TODO(austin) Might be a call of a multi-valued function.
	// It might be possible to combine this code with the
	// assignment code.
	if len(s.Results) != len(a.outVars) {
		a.diag("Unbalanced return not implemented");
		return;
	}

	// Compile expressions and create assigners
	bad := false;
	rs := make([]*exprCompiler, len(s.Results));
	as := make([]func(lv Value, f *Frame), len(s.Results));
	for i, re := range s.Results {
		rs[i] = a.compileExpr(a.scope, re, false);
		if rs[i] == nil {
			bad = true;
			continue;
		}

		errPos := i + 1;
		if len(s.Results) == 1 {
			errPos = 0;
		}
		var lt Type;
		lt, as[i] = mkAssign(a.outVars[i].Type, rs[i], "return", "value", errPos);
		if as[i] == nil {
			bad = true;
		}
	}

	if bad {
		return;
	}

	// Save indexes of return values
	idxs := make([]int, len(s.Results));
	for i, outVar := range a.outVars {
		idxs[i] = outVar.Index;
	}

	// Compile
	a.push(func(v *vm) {
		for i, assign := range as {
			assign(v.activation.Vars[idxs[i]], v.f);
		}
		v.pc = ^uint(0);
	});
	a.err = false;
}

func (a *stmtCompiler) DoBranchStmt(s *ast.BranchStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoBlockStmt(s *ast.BlockStmt) {
	blockScope := a.scope.Fork();
	bc := &blockCompiler{a.funcCompiler, blockScope, false};

	a.push(func(v *vm) {
		v.f = blockScope.NewFrame(v.f);
	});
	bc.compileBlock(s);
	a.push(func(v *vm) {
		v.f = v.f.Outer;
	});

	a.returned = a.returned || bc.returned;
	a.err = false;
}

func (a *stmtCompiler) DoIfStmt(s *ast.IfStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoCaseClause(s *ast.CaseClause) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoSwitchStmt(s *ast.SwitchStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoTypeCaseClause(s *ast.TypeCaseClause) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoTypeSwitchStmt(s *ast.TypeSwitchStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoCommClause(s *ast.CommClause) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoSelectStmt(s *ast.SelectStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoForStmt(s *ast.ForStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoRangeStmt(s *ast.RangeStmt) {
	log.Crash("Not implemented");
}

func (a *blockCompiler) compileBlock(block *ast.BlockStmt) {
	for i, sub := range block.List {
		sc := &stmtCompiler{a, sub.Pos(), true};
		sub.Visit(sc);
		a.err = a.err || sc.err;
	}
}

func (a *compiler) compileFunc(scope *Scope, decl *FuncDecl, body *ast.BlockStmt) (func (f *Frame) Func) {
	// Create body scope
	//
	// The scope of a parameter or result is the body of the
	// corresponding function.
	bodyScope := scope.Fork();
	for i, t := range decl.Type.In {
		bodyScope.DefineVar(decl.InNames[i].Value, t);
	}
	outVars := make([]*Variable, len(decl.Type.Out));
	for i, t := range decl.Type.Out {
		if decl.OutNames[i] != nil {
			outVars[i] = bodyScope.DefineVar(decl.OutNames[i].Value, t);
		} else {
			// TODO(austin) It would be nice to have a
			// better way to define unnamed slots.
			outVars[i] = bodyScope.DefineVar(":out" + strconv.Itoa(i), t);
		}
	}

	// Create block context
	fc := &funcCompiler{a, outVars, false, newCodeBuf(), false};
	if len(decl.OutNames) > 0 && decl.OutNames[0] != nil {
		fc.outVarsNamed = true;
	}
	bc := &blockCompiler{fc, bodyScope, false};

	// Compile body
	bc.compileBlock(body);
	if fc.err {
		return nil;
	}

	// TODO(austin) Check that all gotos were linked?

	// Check that the body returned if necessary
	if len(decl.Type.Out) > 0 && !bc.returned {
		// XXX(Spec) Not specified.
		a.diagAt(&body.Rbrace, "function ends without a return statement");
		return nil;
	}

	code := fc.get();
	return func(f *Frame) Func { return &evalFunc{bodyScope, f, code} };
}

/*
 * Testing interface
 */

type Stmt struct {
	f func (f *Frame);
}

func (s *Stmt) Exec(f *Frame) {
	s.f(f);
}

func CompileStmt(scope *Scope, stmt ast.Stmt) (*Stmt, os.Error) {
	errors := scanner.NewErrorVector();
	cc := &compiler{errors};
	fc := &funcCompiler{cc, nil, false, newCodeBuf(), false};
	bc := &blockCompiler{fc, scope, false};
	sc := &stmtCompiler{bc, stmt.Pos(), true};
	stmt.Visit(sc);
	fc.err = fc.err || sc.err;
	if fc.err {
		return nil, errors.GetError(scanner.Sorted);
	}
	code := fc.get();
	return &Stmt{func(f *Frame) { code.exec(f); }}, nil;
}
