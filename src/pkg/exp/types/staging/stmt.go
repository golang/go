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

// assignment typechecks a single assignment of the form lhs := x. If decl is set,
// the lhs operand must be an identifier. If its type is not set, it is deduced
// from the type or value of x.
//
func (check *checker) assignment(lhs ast.Expr, x *operand, decl bool) {
	if decl {
		ident, ok := lhs.(*ast.Ident)
		if !ok {
			check.errorf(lhs.Pos(), "cannot declare %s", lhs)
			return
		}

		obj := ident.Obj
		if obj.Type == nil {
			// determine type from rhs expression
			var typ Type = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
				// determine the default type for variables
				if obj.Kind == ast.Var && isUntyped(typ) {
					typ = defaultType(typ)
				}
			}
			obj.Type = typ
		}

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
		z.typ = obj.Type.(Type)

		check.assignOperand(&z, x)

		// for constants, set the constant value
		if obj.Kind == ast.Con {
			assert(obj.Data == nil)
			if x.mode != invalid && x.mode != constant {
				check.errorf(x.pos(), "%s is not constant", x) // TODO(gri) better error position
				x.mode = invalid
			}
			if x.mode == constant {
				obj.Data = x.val
			} else {
				// set the constant to the type's zero value to reduce spurious errors
				// TODO(gri) factor this out - useful elsewhere
				switch typ := underlying(obj.Type.(Type)); {
				case typ == Typ[Invalid]:
					// ignore
				case isBoolean(typ):
					obj.Data = false
				case isNumeric(typ):
					obj.Data = int64(0)
				case isString(typ):
					obj.Data = ""
				default:
					check.dump("%s: typ(%s) = %s", obj.Pos(), obj.Name, typ)
					unreachable()
				}
			}
		}

		return
	}

	// regular assignment
	var z operand
	check.expr(&z, lhs, nil, -1)
	check.assignOperand(&z, x)
	if x.mode != invalid && z.mode == constant {
		check.errorf(x.pos(), "cannot assign %s to %s", x, z)
	}
}

func (check *checker) assign1to1(lhs, rhs ast.Expr, decl bool, iota int) {
	if !decl {
		// regular assignment - start with lhs[0] to obtain a type hint
		var z operand
		check.expr(&z, lhs, nil, -1)
		if z.mode == invalid {
			z.typ = nil // so we can proceed with rhs
		}

		var x operand
		check.expr(&x, rhs, z.typ, -1)
		if x.mode == invalid {
			return
		}

		check.assignOperand(&z, &x)
		return
	}

	// declaration - rhs may or may not be typed yet
	ident, ok := lhs.(*ast.Ident)
	if !ok {
		check.errorf(lhs.Pos(), "cannot declare %s", lhs)
		return
	}

	obj := ident.Obj
	var typ Type
	if obj.Type != nil {
		typ = obj.Type.(Type)
	}

	var x operand
	check.expr(&x, rhs, typ, iota)
	if x.mode == invalid {
		return
	}

	if typ == nil {
		// determine lhs type from rhs expression;
		// for variables, convert untyped types to
		// default types
		typ = x.typ
		if obj.Kind == ast.Var && isUntyped(typ) {
			// TODO(gri) factor this out
			var k BasicKind
			switch typ.(*Basic).Kind {
			case UntypedBool:
				k = Bool
			case UntypedRune:
				k = Rune
			case UntypedInt:
				k = Int
			case UntypedFloat:
				k = Float64
			case UntypedComplex:
				k = Complex128
			case UntypedString:
				k = String
			default:
				unreachable()
			}
			typ = Typ[k]
		}
		obj.Type = typ
	}

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

	check.assignOperand(&z, &x)

	// for constants, set their value
	if obj.Kind == ast.Con {
		assert(obj.Data == nil)
		if x.mode != constant {
			check.errorf(x.pos(), "%s is not constant", x)
			// set the constant to the type's zero value to reduce spurious errors
			// TODO(gri) factor this out - useful elsewhere
			switch typ := underlying(typ); {
			case typ == Typ[Invalid]:
				// ignore
			case isBoolean(typ):
				obj.Data = false
			case isNumeric(typ):
				obj.Data = int64(0)
			case isString(typ):
				obj.Data = ""
			default:
				unreachable()
			}
			return
		}
		obj.Data = x.val
	}
}

// assignNtoM typechecks a general assignment. If decl is set, the lhs operands
// must be identifiers. If their types are not set, they are deduced from the
// types of the corresponding rhs expressions. iota >= 0 indicates that the
// "assignment" is part of a constant declaration.
// Precondition: len(lhs) > 0 .
//
func (check *checker) assignNtoM(lhs, rhs []ast.Expr, decl bool, iota int) {
	assert(len(lhs) >= 1)

	if len(lhs) == len(rhs) {
		for i, e := range rhs {
			check.assign1to1(lhs[i], e, decl, iota)
		}
		return
	}

	if len(rhs) == 1 {
		// len(lhs) >= 2; therefore a correct rhs expression
		// cannot be a shift and we don't need a type hint -
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
				check.assignment(lhs[i], &x, decl)
			}
			return
		}

		if x.mode == valueok && len(lhs) == 2 {
			// comma-ok expression
			x.mode = value
			check.assignment(lhs[0], &x, decl)

			x.mode = value
			x.typ = Typ[UntypedBool]
			check.assignment(lhs[1], &x, decl)
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

// stmt typechecks statement s.
func (check *checker) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.BadStmt, *ast.EmptyStmt:
		// ignore

	case *ast.DeclStmt:
		unimplemented()

	case *ast.LabeledStmt:
		unimplemented()

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
		check.exprOrType(&x, s.X, nil, -1, false)
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
		unimplemented()

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
			}
			var x, y operand
			check.expr(&x, s.Lhs[0], nil, -1)
			check.expr(&y, s.Rhs[0], nil, -1)
			check.binary(&x, &y, op, nil)
			check.assignment(s.Lhs[0], &x, false)
		}

	case *ast.GoStmt:
		unimplemented()

	case *ast.DeferStmt:
		unimplemented()

	case *ast.ReturnStmt:
		unimplemented()

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
		unimplemented()

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
