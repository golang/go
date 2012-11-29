// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/token"
)

func (check *checker) assignOperand(z, x *operand) {
	if t, ok := x.typ.(*tuple); ok {
		// TODO(gri) elsewhere we use "assignment count mismatch" (consolidate)
		check.errorf(x.pos(), "%d-valued expression %s used as single value", len(t.list), x)
		x.mode = invalid
		return
	}

	check.convertUntyped(x, z.typ)

	if !x.isAssignable(z.typ) {
		check.errorf(x.pos(), "cannot assign %s to %s", x, z)
		x.mode = invalid
	}
}

// assign1to1 typechecks a single assignment of the form lhs := rhs (if rhs != nil),
// or lhs := x (if rhs == nil). If decl is set, the lhs operand must be an identifier.
// If its type is not set, it is deduced from the type or value of x. If lhs has a
// type it is used as a hint when evaluating rhs, if present.
//
func (check *checker) assign1to1(lhs, rhs ast.Expr, x *operand, decl bool, iota int) {
	ident, _ := lhs.(*ast.Ident)
	if x == nil {
		assert(rhs != nil)
		x = new(operand)
	}

	if ident != nil && ident.Name == "_" {
		// anything can be assigned to a blank identifier - check rhs only, if present
		if rhs != nil {
			check.expr(x, rhs, nil, iota)
		}
		return
	}

	if !decl {
		// regular assignment - start with lhs to obtain a type hint
		var z operand
		check.expr(&z, lhs, nil, -1)
		if z.mode == invalid {
			z.typ = nil // so we can proceed with rhs
		}

		if rhs != nil {
			check.expr(x, rhs, z.typ, -1)
			if x.mode == invalid {
				return
			}
		}

		check.assignOperand(&z, x)
		if x.mode != invalid && z.mode == constant {
			check.errorf(x.pos(), "cannot assign %s to %s", x, &z)
		}
		return
	}

	// declaration - lhs must be an identifier
	if ident == nil {
		check.errorf(lhs.Pos(), "cannot declare %s", lhs)
		return
	}

	// lhs may or may not be typed yet
	obj := ident.Obj
	var typ Type
	if obj.Type != nil {
		typ = obj.Type.(Type)
	}

	if rhs != nil {
		check.expr(x, rhs, typ, iota)
		// continue even if x.mode == invalid
	}

	if typ == nil {
		// determine lhs type from rhs expression;
		// for variables, convert untyped types to
		// default types
		typ = Typ[Invalid]
		if x.mode != invalid {
			typ = x.typ
			if obj.Kind == ast.Var && isUntyped(typ) {
				typ = defaultType(typ)
			}
		}
		obj.Type = typ
	}

	if x.mode != invalid {
		var z operand
		switch obj.Kind {
		case ast.Con:
			z.mode = constant
		case ast.Var:
			z.mode = variable
		default:
			unreachable()
		}
		z.expr = ident
		z.typ = typ
		check.assignOperand(&z, x)
	}

	// for constants, set their value
	if obj.Kind == ast.Con {
		assert(obj.Data == nil)
		if x.mode != invalid {
			if x.mode == constant {
				if isConstType(x.typ) {
					obj.Data = x.val
				} else {
					check.errorf(x.pos(), "%s has invalid constant type", x)
				}
			} else {
				check.errorf(x.pos(), "%s is not constant", x)
			}
		}
		if obj.Data == nil {
			// set the constant to its type's zero value to reduce spurious errors
			switch typ := underlying(obj.Type.(Type)); {
			case typ == Typ[Invalid]:
				// ignore
			case isBoolean(typ):
				obj.Data = false
			case isNumeric(typ):
				obj.Data = int64(0)
			case isString(typ):
				obj.Data = ""
			case hasNil(typ):
				obj.Data = nilConst
			default:
				// in all other cases just prevent use of the constant
				obj.Kind = ast.Bad
			}
		}
	}
}

// assignNtoM typechecks a general assignment. If decl is set, the lhs operands
// must be identifiers. If their types are not set, they are deduced from the
// types of the corresponding rhs expressions. iota >= 0 indicates that the
// "assignment" is part of a constant declaration.
// Precondition: len(lhs) > 0 .
//
func (check *checker) assignNtoM(lhs, rhs []ast.Expr, decl bool, iota int) {
	assert(len(lhs) > 0)

	if len(lhs) == len(rhs) {
		for i, e := range rhs {
			check.assign1to1(lhs[i], e, nil, decl, iota)
		}
		return
	}

	if len(rhs) == 1 {
		// len(lhs) > 1, therefore a correct rhs expression
		// cannot be a shift and we don't need a type hint;
		// ok to evaluate rhs first
		var x operand
		check.expr(&x, rhs[0], nil, iota)
		if x.mode == invalid {
			return
		}

		if t, ok := x.typ.(*tuple); ok && len(lhs) == len(t.list) {
			// function result
			x.mode = value
			for i, typ := range t.list {
				x.expr = nil // TODO(gri) should do better here
				x.typ = typ
				check.assign1to1(lhs[i], nil, &x, decl, iota)
			}
			return
		}

		if x.mode == valueok && len(lhs) == 2 {
			// comma-ok expression
			x.mode = value
			check.assign1to1(lhs[0], nil, &x, decl, iota)

			x.mode = value
			x.typ = Typ[UntypedBool]
			check.assign1to1(lhs[1], nil, &x, decl, iota)
			return
		}
	}

	check.errorf(lhs[0].Pos(), "assignment count mismatch: %d = %d", len(lhs), len(rhs))

	// avoid checking the same declaration over and over
	// again for each lhs identifier that has no type yet
	if iota >= 0 {
		// declaration
		for _, e := range lhs {
			if ident, ok := e.(*ast.Ident); ok {
				ident.Obj.Type = Typ[Invalid]
			}
		}
	}
}

func (check *checker) optionalStmt(s ast.Stmt) {
	if s != nil {
		check.stmt(s)
	}
}

func (check *checker) stmtList(list []ast.Stmt) {
	for _, s := range list {
		check.stmt(s)
	}
}

func (check *checker) call(c ast.Expr) {
	call, _ := c.(*ast.CallExpr)
	if call == nil {
		// For go/defer, the parser makes sure that we have a function call,
		// so if we don't, the AST was created incorrectly elsewhere.
		// TODO(gri) consider removing the checks from the parser.
		check.invalidAST(c.Pos(), "%s is not a function call", c)
		return
	}
	var x operand
	check.rawExpr(&x, call, nil, -1, false) // don't check if value is used
	// TODO(gri) If a builtin is called, the builtin must be valid in statement
	//           context. However, the spec doesn't say that explicitly.
}

// stmt typechecks statement s.
func (check *checker) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.BadStmt, *ast.EmptyStmt:
		// ignore

	case *ast.DeclStmt:
		check.decl(s.Decl)

	case *ast.LabeledStmt:
		// TODO(gri) anything to do with label itself?
		check.stmt(s.Stmt)

	case *ast.ExprStmt:
		var x operand
		used := false
		switch e := unparen(s.X).(type) {
		case *ast.CallExpr:
			// function calls are permitted
			used = true
			// but some builtins are excluded
			check.expr(&x, e.Fun, nil, -1)
			if x.mode != invalid {
				if b, ok := x.typ.(*builtin); ok && !b.isStatement {
					used = false
				}
			}
		case *ast.UnaryExpr:
			// receive operations are permitted
			if e.Op == token.ARROW {
				used = true
			}
		}
		if !used {
			check.errorf(s.Pos(), "%s not used", s.X)
			// ok to continue
		}
		check.rawExpr(&x, s.X, nil, -1, false)
		if x.mode == typexpr {
			check.errorf(x.pos(), "%s is not an expression", x)
		}

	case *ast.SendStmt:
		var ch, x operand
		check.expr(&ch, s.Chan, nil, -1)
		check.expr(&x, s.Value, nil, -1)
		if ch.mode == invalid || x.mode == invalid {
			return
		}
		if tch, ok := underlying(ch.typ).(*Chan); !ok || tch.Dir&ast.SEND == 0 || !x.isAssignable(tch.Elt) {
			check.invalidOp(ch.pos(), "cannot send %s to channel %s", &x, &ch)
		}

	case *ast.IncDecStmt:
		var op token.Token
		switch s.Tok {
		case token.INC:
			op = token.ADD
		case token.DEC:
			op = token.SUB
		default:
			check.invalidAST(s.TokPos, "unknown inc/dec operation %s", s.Tok)
			return
		}
		var x, y operand
		check.expr(&x, s.X, nil, -1)
		check.expr(&y, &ast.BasicLit{ValuePos: x.pos(), Kind: token.INT, Value: "1"}, nil, -1) // use x's position
		check.binary(&x, &y, op, nil)
		check.assign1to1(s.X, nil, &x, false, -1)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			if len(s.Lhs) == 0 {
				check.invalidAST(s.Pos(), "missing lhs in assignment")
				return
			}
			check.assignNtoM(s.Lhs, s.Rhs, s.Tok == token.DEFINE, -1)
		default:
			// assignment operations
			if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
				check.errorf(s.TokPos, "assignment operation %s requires single-valued expressions", s.Tok)
				return
			}
			// TODO(gri) make this conversion more efficient
			var op token.Token
			switch s.Tok {
			case token.ADD_ASSIGN:
				op = token.ADD
			case token.SUB_ASSIGN:
				op = token.SUB
			case token.MUL_ASSIGN:
				op = token.MUL
			case token.QUO_ASSIGN:
				op = token.QUO
			case token.REM_ASSIGN:
				op = token.REM
			case token.AND_ASSIGN:
				op = token.AND
			case token.OR_ASSIGN:
				op = token.OR
			case token.XOR_ASSIGN:
				op = token.XOR
			case token.SHL_ASSIGN:
				op = token.SHL
			case token.SHR_ASSIGN:
				op = token.SHR
			case token.AND_NOT_ASSIGN:
				op = token.AND_NOT
			default:
				check.invalidAST(s.TokPos, "unknown assignment operation %s", s.Tok)
				return
			}
			var x, y operand
			check.expr(&x, s.Lhs[0], nil, -1)
			check.expr(&y, s.Rhs[0], nil, -1)
			check.binary(&x, &y, op, nil)
			check.assign1to1(s.Lhs[0], nil, &x, false, -1)
		}

	case *ast.GoStmt:
		check.call(s.Call)

	case *ast.DeferStmt:
		check.call(s.Call)

	case *ast.ReturnStmt:
		sig := check.functypes[len(check.functypes)-1]
		if n := len(sig.Results); n > 0 {
			// TODO(gri) should not have to compute lhs, named every single time - clean this up
			lhs := make([]ast.Expr, n)
			named := false // if set, function has named results
			for i, res := range sig.Results {
				if len(res.Name) > 0 {
					// a blank (_) result parameter is a named result parameter!
					named = true
				}
				name := ast.NewIdent(res.Name)
				name.NamePos = s.Pos()
				name.Obj = res
				lhs[i] = name
			}
			if len(s.Results) > 0 || !named {
				// TODO(gri) assignNtoM should perhaps not require len(lhs) > 0
				check.assignNtoM(lhs, s.Results, false, -1)
			}
		} else if len(s.Results) > 0 {
			check.errorf(s.Pos(), "no result values expected")
		}

	case *ast.BranchStmt:
		unimplemented()

	case *ast.BlockStmt:
		check.stmtList(s.List)

	case *ast.IfStmt:
		check.optionalStmt(s.Init)
		var x operand
		check.expr(&x, s.Cond, nil, -1)
		if !isBoolean(x.typ) {
			check.errorf(s.Cond.Pos(), "non-boolean condition in if statement")
		}
		check.stmt(s.Body)
		check.optionalStmt(s.Else)

	case *ast.SwitchStmt:
		check.optionalStmt(s.Init)
		var x operand
		if s.Tag != nil {
			check.expr(&x, s.Tag, nil, -1)
		} else {
			// TODO(gri) should provide a position (see IncDec) for good error messages
			x.mode = constant
			x.typ = Typ[UntypedBool]
			x.val = true
		}
		for _, s := range s.Body.List {
			if clause, ok := s.(*ast.CaseClause); ok {
				for _, expr := range clause.List {
					var y operand
					check.expr(&y, expr, nil, -1)
					// TODO(gri) x and y must be comparable
				}
				check.stmtList(clause.Body)
			} else {
				check.errorf(s.Pos(), "invalid AST: case clause expected")
			}
		}

	case *ast.TypeSwitchStmt:
		unimplemented()

	case *ast.SelectStmt:
		for _, s := range s.Body.List {
			c, ok := s.(*ast.CommClause)
			if !ok {
				check.invalidAST(s.Pos(), "communication clause expected")
				continue
			}
			check.optionalStmt(c.Comm) // TODO(gri) check correctness of c.Comm (must be Send/RecvStmt)
			check.stmtList(c.Body)
		}

	case *ast.ForStmt:
		check.optionalStmt(s.Init)
		if s.Cond != nil {
			var x operand
			check.expr(&x, s.Cond, nil, -1)
			if !isBoolean(x.typ) {
				check.errorf(s.Cond.Pos(), "non-boolean condition in for statement")
			}
		}
		check.optionalStmt(s.Post)
		check.stmt(s.Body)

	case *ast.RangeStmt:
		unimplemented()

	default:
		check.errorf(s.Pos(), "invalid statement")
	}
}
