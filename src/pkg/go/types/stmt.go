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
	if t, ok := x.typ.(*Result); ok {
		// TODO(gri) elsewhere we use "assignment count mismatch" (consolidate)
		check.errorf(x.pos(), "%d-valued expression %s used as single value", len(t.Values), x)
		x.mode = invalid
		return
	}

	check.convertUntyped(x, z.typ)

	if !x.isAssignable(z.typ) {
		check.errorf(x.pos(), "cannot assign %s to %s", x, z)
		x.mode = invalid
	}
}

// assign1to1 typechecks a single assignment of the form lhs = rhs (if rhs != nil),
// or lhs = x (if rhs == nil). If decl is set, the lhs operand must be an identifier.
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
	obj := check.lookup(ident)
	var typ Type
	if t := obj.GetType(); t != nil {
		typ = t
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
			if _, ok := obj.(*Var); ok && isUntyped(typ) {
				if x.isNil() {
					check.errorf(x.pos(), "use of untyped nil")
					x.mode = invalid
				} else {
					typ = defaultType(typ)
				}
			}
		}
		switch obj := obj.(type) {
		case *Const:
			obj.Type = typ
		case *Var:
			obj.Type = typ
		default:
			unreachable()
		}
	}

	if x.mode != invalid {
		var z operand
		switch obj.(type) {
		case *Const:
			z.mode = constant
		case *Var:
			z.mode = variable
		default:
			unreachable()
		}
		z.expr = ident
		z.typ = typ
		check.assignOperand(&z, x)
	}

	// for constants, set their value
	if obj, ok := obj.(*Const); ok {
		assert(obj.Val == nil)
		if x.mode != invalid {
			if x.mode == constant {
				if isConstType(x.typ) {
					obj.Val = x.val
				} else {
					check.errorf(x.pos(), "%s has invalid constant type", x)
				}
			} else {
				check.errorf(x.pos(), "%s is not constant", x)
			}
		}
		if obj.Val == nil {
			// set the constant to its type's zero value to reduce spurious errors
			switch typ := underlying(obj.Type); {
			case typ == Typ[Invalid]:
				// ignore
			case isBoolean(typ):
				obj.Val = false
			case isNumeric(typ):
				obj.Val = int64(0)
			case isString(typ):
				obj.Val = ""
			case hasNil(typ):
				obj.Val = nilConst
			default:
				// in all other cases just prevent use of the constant
				// TODO(gri) re-evaluate this code
				obj.Val = nilConst
			}
		}
	}
}

// assignNtoM typechecks a general assignment. If decl is set, the lhs operands
// must be identifiers. If their types are not set, they are deduced from the
// types of the corresponding rhs expressions. iota >= 0 indicates that the
// "assignment" is part of a constant/variable declaration.
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

		if t, ok := x.typ.(*Result); ok && len(lhs) == len(t.Values) {
			// function result
			x.mode = value
			for i, obj := range t.Values {
				x.expr = nil // TODO(gri) should do better here
				x.typ = obj.Type
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
			if name, ok := e.(*ast.Ident); ok {
				switch obj := check.lookup(name).(type) {
				case *Const:
					obj.Type = Typ[Invalid]
				case *Var:
					obj.Type = Typ[Invalid]
				default:
					unreachable()
				}
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

func (check *checker) call(call *ast.CallExpr) {
	var x operand
	check.rawExpr(&x, call, nil, -1, false) // don't check if value is used
	// TODO(gri) If a builtin is called, the builtin must be valid in statement context.
}

func (check *checker) multipleDefaults(list []ast.Stmt) {
	var first ast.Stmt
	for _, s := range list {
		var d ast.Stmt
		switch c := s.(type) {
		case *ast.CaseClause:
			if len(c.List) == 0 {
				d = s
			}
		case *ast.CommClause:
			if c.Comm == nil {
				d = s
			}
		default:
			check.invalidAST(s.Pos(), "case/communication clause expected")
		}
		if d != nil {
			if first != nil {
				check.errorf(d.Pos(), "multiple defaults (first at %s)", first.Pos())
			} else {
				first = d
			}
		}
	}
}

// stmt typechecks statement s.
func (check *checker) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.BadStmt, *ast.EmptyStmt:
		// ignore

	case *ast.DeclStmt:
		d, _ := s.Decl.(*ast.GenDecl)
		if d == nil || (d.Tok != token.CONST && d.Tok != token.TYPE && d.Tok != token.VAR) {
			check.invalidAST(token.NoPos, "const, type, or var declaration expected")
			return
		}
		if d.Tok == token.CONST {
			check.assocInitvals(d)
		}
		check.decl(d)

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
			// (Caution: This evaluates e.Fun twice, once here and once
			//           below as part of s.X. This has consequences for
			//           check.register. Perhaps this can be avoided.)
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
			check.errorf(x.pos(), "%s is not an expression", &x)
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
			if x.mode == invalid {
				return
			}
			check.expr(&y, s.Rhs[0], nil, -1)
			if y.mode == invalid {
				return
			}
			check.binary(&x, &y, op, x.typ)
			check.assign1to1(s.Lhs[0], nil, &x, false, -1)
		}

	case *ast.GoStmt:
		check.call(s.Call)

	case *ast.DeferStmt:
		check.call(s.Call)

	case *ast.ReturnStmt:
		sig := check.funcsig
		if n := len(sig.Results); n > 0 {
			// TODO(gri) should not have to compute lhs, named every single time - clean this up
			lhs := make([]ast.Expr, n)
			named := false // if set, function has named results
			for i, res := range sig.Results {
				if len(res.Name) > 0 {
					// a blank (_) result parameter is a named result
					named = true
				}
				name := ast.NewIdent(res.Name)
				name.NamePos = s.Pos()
				check.register(name, &Var{Name: res.Name, Type: res.Type})
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
		// TODO(gri) implement this

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
		tag := s.Tag
		if tag == nil {
			// use fake true tag value and position it at the opening { of the switch
			ident := &ast.Ident{NamePos: s.Body.Lbrace, Name: "true"}
			check.register(ident, Universe.Lookup("true"))
			tag = ident
		}
		check.expr(&x, tag, nil, -1)

		check.multipleDefaults(s.Body.List)
		seen := make(map[interface{}]token.Pos)
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CaseClause)
			if clause == nil {
				continue // error reported before
			}
			if x.mode != invalid {
				for _, expr := range clause.List {
					x := x // copy of x (don't modify original)
					var y operand
					check.expr(&y, expr, nil, -1)
					if y.mode == invalid {
						continue // error reported before
					}
					// If we have a constant case value, it must appear only
					// once in the switch statement. Determine if there is a
					// duplicate entry, but only report an error if there are
					// no other errors.
					var dupl token.Pos
					var yy operand
					if y.mode == constant {
						// TODO(gri) This code doesn't work correctly for
						//           large integer, floating point, or
						//           complex values - the respective struct
						//           comparisons are shallow. Need to use a
						//           hash function to index the map.
						dupl = seen[y.val]
						seen[y.val] = y.pos()
						yy = y // remember y
					}
					// TODO(gri) The convertUntyped call pair below appears in other places. Factor!
					// Order matters: By comparing y against x, error positions are at the case values.
					check.convertUntyped(&y, x.typ)
					if y.mode == invalid {
						continue // error reported before
					}
					check.convertUntyped(&x, y.typ)
					if x.mode == invalid {
						continue // error reported before
					}
					check.comparison(&y, &x, token.EQL)
					if y.mode != invalid && dupl.IsValid() {
						check.errorf(yy.pos(), "%s is duplicate case (previous at %s)",
							&yy, check.fset.Position(dupl))
					}
				}
			}
			check.stmtList(clause.Body)
		}

	case *ast.TypeSwitchStmt:
		check.optionalStmt(s.Init)

		// A type switch guard must be of the form:
		//
		//     TypeSwitchGuard = [ identifier ":=" ] PrimaryExpr "." "(" "type" ")" .
		//
		// The parser is checking syntactic correctness;
		// remaining syntactic errors are considered AST errors here.
		// TODO(gri) better factoring of error handling (invalid ASTs)
		//
		var lhs *Var // lhs variable or nil
		var rhs ast.Expr
		switch guard := s.Assign.(type) {
		case *ast.ExprStmt:
			rhs = guard.X
		case *ast.AssignStmt:
			if len(guard.Lhs) != 1 || guard.Tok != token.DEFINE || len(guard.Rhs) != 1 {
				check.invalidAST(s.Pos(), "incorrect form of type switch guard")
				return
			}
			ident, _ := guard.Lhs[0].(*ast.Ident)
			if ident == nil {
				check.invalidAST(s.Pos(), "incorrect form of type switch guard")
				return
			}
			lhs = check.lookup(ident).(*Var)
			rhs = guard.Rhs[0]
		default:
			check.invalidAST(s.Pos(), "incorrect form of type switch guard")
			return
		}

		// rhs must be of the form: expr.(type) and expr must be an interface
		expr, _ := rhs.(*ast.TypeAssertExpr)
		if expr == nil || expr.Type != nil {
			check.invalidAST(s.Pos(), "incorrect form of type switch guard")
			return
		}
		var x operand
		check.expr(&x, expr.X, nil, -1)
		if x.mode == invalid {
			return
		}
		var T *Interface
		if T, _ = underlying(x.typ).(*Interface); T == nil {
			check.errorf(x.pos(), "%s is not an interface", &x)
			return
		}

		check.multipleDefaults(s.Body.List)
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CaseClause)
			if clause == nil {
				continue // error reported before
			}
			// Check each type in this type switch case.
			var typ Type
			for _, expr := range clause.List {
				typ = check.typOrNil(expr, false)
				if typ != nil && typ != Typ[Invalid] {
					if method, wrongType := missingMethod(typ, T); method != nil {
						var msg string
						if wrongType {
							msg = "%s cannot have dynamic type %s (wrong type for method %s)"
						} else {
							msg = "%s cannot have dynamic type %s (missing method %s)"
						}
						check.errorf(expr.Pos(), msg, &x, typ, method.Name)
						// ok to continue
					}
				}
			}
			// If lhs exists, set its type for each clause.
			if lhs != nil {
				// In clauses with a case listing exactly one type, the variable has that type;
				// otherwise, the variable has the type of the expression in the TypeSwitchGuard.
				if len(clause.List) != 1 || typ == nil {
					typ = x.typ
				}
				lhs.Type = typ
			}
			check.stmtList(clause.Body)
		}

		// There is only one object (lhs) associated with a lhs identifier, but that object
		// assumes different types for different clauses. Set it back to the type of the
		// TypeSwitchGuard expression so that that variable always has a valid type.
		if lhs != nil {
			lhs.Type = x.typ
		}

	case *ast.SelectStmt:
		check.multipleDefaults(s.Body.List)
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CommClause)
			if clause == nil {
				continue // error reported before
			}
			check.optionalStmt(clause.Comm) // TODO(gri) check correctness of c.Comm (must be Send/RecvStmt)
			check.stmtList(clause.Body)
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
		// check expression to iterate over
		decl := s.Tok == token.DEFINE
		var x operand
		check.expr(&x, s.X, nil, -1)
		if x.mode == invalid {
			// if we don't have a declaration, we can still check the loop's body
			if !decl {
				check.stmt(s.Body)
			}
			return
		}

		// determine key/value types
		var key, val Type
		switch typ := underlying(x.typ).(type) {
		case *Basic:
			if isString(typ) {
				key = Typ[UntypedInt]
				val = Typ[UntypedRune]
			}
		case *Array:
			key = Typ[UntypedInt]
			val = typ.Elt
		case *Slice:
			key = Typ[UntypedInt]
			val = typ.Elt
		case *Pointer:
			if typ, _ := underlying(typ.Base).(*Array); typ != nil {
				key = Typ[UntypedInt]
				val = typ.Elt
			}
		case *Map:
			key = typ.Key
			val = typ.Elt
		case *Chan:
			key = typ.Elt
			if typ.Dir&ast.RECV == 0 {
				check.errorf(x.pos(), "cannot range over send-only channel %s", &x)
				// ok to continue
			}
			if s.Value != nil {
				check.errorf(s.Value.Pos(), "iteration over %s permits only one iteration variable", &x)
				// ok to continue
			}
		}

		if key == nil {
			check.errorf(x.pos(), "cannot range over %s", &x)
			// if we don't have a declaration, we can still check the loop's body
			if !decl {
				check.stmt(s.Body)
			}
			return
		}

		// check assignment to/declaration of iteration variables
		// TODO(gri) The error messages/positions are not great here,
		//           they refer to the expression in the range clause.
		//           Should give better messages w/o too much code
		//           duplication (assignment checking).
		x.mode = value
		if s.Key != nil {
			x.typ = key
			check.assign1to1(s.Key, nil, &x, decl, -1)
		} else {
			check.invalidAST(s.Pos(), "range clause requires index iteration variable")
			// ok to continue
		}
		if s.Value != nil {
			x.typ = val
			check.assign1to1(s.Value, nil, &x, decl, -1)
		}

		check.stmt(s.Body)

	default:
		check.errorf(s.Pos(), "invalid statement")
	}
}
