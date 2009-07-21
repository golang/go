// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"eval";
	"fmt";
	"log";
	"os";
	"go/ast";
	"go/scanner";
	"go/token";
)

type stmtCompiler struct {
	scope *Scope;
	errors scanner.ErrorHandler;
	pos token.Position;
	f func (f *Frame);
}

func (a *stmtCompiler) diagAt(pos token.Position, format string, args ...) {
	a.errors.Error(pos, fmt.Sprintf(format, args));
}

func (a *stmtCompiler) diag(format string, args ...) {
	a.diagAt(a.pos, format, args);
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
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoLabeledStmt(s *ast.LabeledStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoExprStmt(s *ast.ExprStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoIncDecStmt(s *ast.IncDecStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) doAssign(s *ast.AssignStmt) {
	if len(s.Lhs) != len(s.Rhs) {
		log.Crashf("Unbalanced assignment not implemented %v %v %v", len(s.Lhs), s.Tok, len(s.Rhs));
	}

	bad := false;

	// Compile right side first so we have the types when
	// compiling the left side and so we don't see definitions
	// made on the left side.
	rs := make([]*exprCompiler, len(s.Rhs));
	for i, re := range s.Rhs {
		rs[i] = compileExpr(re, a.scope, a.errors);
		if rs[i] == nil {
			bad = true;
		}
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
				a.diagAt(le.Pos(), "left side of := must be a name");
				bad = true;
				continue;
			}

			// Is this simply an assignment?
			if _, ok := a.scope.defs[ident.Value]; ok {
				goto assignment;
			}

			if rs[i] == nil {
				// TODO(austin) Define a placeholder.
				continue;
			}

			// Generate assigner and get type
			var lt Type;
			lt, as[i] = mkAssign(nil, rs[i], "assignment", errPos, "position");
			if lt == nil {
				bad = true;
				continue;
			}

			// Define identifier
			v := a.scope.DefineVar(ident.Value, lt);
			nDefs++;
			if v == nil {
				log.Crashf("Failed to define %s", ident.Value);
			}
		}

	assignment:
		ls[i] = compileExpr(le, a.scope, a.errors);
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
			lt, as[i] = mkAssign(ls[i].t, rs[i], "assignment", errPos, "position");
			if lt == nil {
				bad = true;
				continue;
			}
		}
	}

	if bad {
		return;
	}


	// A short variable declaration may redeclare variables
	// provided they were originally declared in the same block
	// with the same type, and at least one of the variables is
	// new.
	if s.Tok == token.DEFINE && nDefs == 0 {
		a.diag("at least one new variable must be declared");
		return;
	}

	n := len(s.Lhs);
	if n == 1 {
		lf := ls[0].evalAddr;
		assign := as[0];
		a.f = func(f *Frame) { assign(lf(f), f) };
	} else {
		a.f = func(f *Frame) {
			temps := make([]Value, n);
			// Assign to temporaries
			for i := 0; i < n; i++ {
				// TODO(austin) Don't capture ls
				temps[i] = ls[i].t.Zero();
				as[i](temps[i], f);
			}
			// Copy to destination
			for i := 0; i < n; i++ {
				ls[i].evalAddr(f).Assign(temps[i]);
			}
		}
	}
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

	l := compileExpr(s.Lhs[0], a.scope, a.errors);
	r := compileExpr(s.Rhs[0], a.scope, a.errors);
	if l == nil || r == nil {
		return;
	}

	if l.evalAddr == nil {
		l.diag("cannot assign to %s", l.desc);
		return;
	}

	ec := r.copy();
	ec.pos = s.TokPos;
	ec.doBinaryExpr(assignOpToOp[s.Tok], l, r);
	if ec.t == nil {
		return;
	}

	lf := l.evalAddr;
	_, assign := mkAssign(l.t, r, "assignment", 0, "");
	if assign == nil {
		return;
	}
	a.f = func(f *Frame) { assign(lf(f), f) };
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
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoBranchStmt(s *ast.BranchStmt) {
	log.Crash("Not implemented");
}

func (a *stmtCompiler) DoBlockStmt(s *ast.BlockStmt) {
	log.Crash("Not implemented");
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

/*
 * Public interface
 */

type Stmt struct {
	f func (f *Frame);
}

func (s *Stmt) Exec(f *Frame) {
	s.f(f);
}

func CompileStmt(stmt ast.Stmt, scope *Scope) (*Stmt, os.Error) {
	errors := scanner.NewErrorVector();
	sc := &stmtCompiler{scope, errors, stmt.Pos(), nil};
	stmt.Visit(sc);
	if sc.f == nil {
		return nil, errors.GetError(scanner.Sorted);
	}
	return &Stmt{sc.f}, nil;
}
