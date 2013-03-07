// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/token"
)

// assigment reports whether x can be assigned to a variable of type 'to',
// if necessary by attempting to convert untyped values to the appropriate
// type. If x.mode == invalid upon return, then assignment has already
// issued an error message and the caller doesn't have to report another.
// TODO(gri) This latter behavior is for historic reasons and complicates
// callers. Needs to be cleaned up.
func (check *checker) assignment(x *operand, to Type) bool {
	if x.mode == invalid {
		return false
	}

	if t, ok := x.typ.(*Result); ok {
		// TODO(gri) elsewhere we use "assignment count mismatch" (consolidate)
		check.errorf(x.pos(), "%d-valued expression %s used as single value", len(t.Values), x)
		x.mode = invalid
		return false
	}

	check.convertUntyped(x, to)

	return x.mode != invalid && x.isAssignable(check.ctxt, to)
}

// assign1to1 typechecks a single assignment of the form lhs = rhs (if rhs != nil), or
// lhs = x (if rhs == nil). If decl is set, the lhs expression must be an identifier;
// if its type is not set, it is deduced from the type of x or set to Typ[Invalid] in
// case of an error.
//
func (check *checker) assign1to1(lhs, rhs ast.Expr, x *operand, decl bool, iota int) {
	// Start with rhs so we have an expression type
	// for declarations with implicit type.
	if x == nil {
		x = new(operand)
		check.expr(x, rhs, nil, iota)
		// don't exit for declarations - we need the lhs first
		if x.mode == invalid && !decl {
			return
		}
	}
	// x.mode == valid || decl

	// lhs may be an identifier
	ident, _ := lhs.(*ast.Ident)

	// regular assignment; we know x is valid
	if !decl {
		// anything can be assigned to the blank identifier
		if ident != nil && ident.Name == "_" {
			return
		}

		var z operand
		check.expr(&z, lhs, nil, -1)
		if z.mode == invalid {
			return
		}

		// TODO(gri) verify that all other z.mode values
		//           that may appear here are legal
		if z.mode == constant || !check.assignment(x, z.typ) {
			if x.mode != invalid {
				check.errorf(x.pos(), "cannot assign %s to %s", x, &z)
			}
		}
		return
	}

	// declaration with initialization; lhs must be an identifier
	if ident == nil {
		check.errorf(lhs.Pos(), "cannot declare %s", lhs)
		return
	}

	// Determine typ of lhs: If the object doesn't have a type
	// yet, determine it from the type of x; if x is invalid,
	// set the object type to Typ[Invalid].
	var typ Type
	obj := check.lookup(ident)
	switch obj := obj.(type) {
	default:
		unreachable()

	case nil:
		// TODO(gri) is this really unreachable?
		unreachable()

	case *Const:
		typ = obj.Type // may already be Typ[Invalid]
		if typ == nil {
			typ = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
			}
			obj.Type = typ
		}

	case *Var:
		typ = obj.Type // may already be Typ[Invalid]
		if typ == nil {
			typ = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
				if isUntyped(typ) {
					// convert untyped types to default types
					if typ == Typ[UntypedNil] {
						check.errorf(x.pos(), "use of untyped nil")
						typ = Typ[Invalid]
					} else {
						typ = defaultType(typ)
					}
				}
			}
			obj.Type = typ
		}
	}

	// nothing else to check if we don't have a valid lhs or rhs
	if typ == Typ[Invalid] || x.mode == invalid {
		return
	}

	if !check.assignment(x, typ) {
		if x.mode != invalid {
			if x.typ != Typ[Invalid] && typ != Typ[Invalid] {
				check.errorf(x.pos(), "cannot initialize %s (type %s) with %s", ident.Name, typ, x)
			}
		}
		return
	}

	// for constants, set their value
	if obj, _ := obj.(*Const); obj != nil {
		obj.Val = nil // failure case: we don't know the constant value
		if x.mode == constant {
			if isConstType(x.typ) {
				obj.Val = x.val
			} else if x.typ != Typ[Invalid] {
				check.errorf(x.pos(), "%s has invalid constant type", x)
			}
		} else if x.mode != invalid {
			check.errorf(x.pos(), "%s is not constant", x)
		}
	}
}

// assignNtoM typechecks a general assignment. If decl is set, the lhs expressions
// must be identifiers; if their types are not set, they are deduced from the types
// of the corresponding rhs expressions, or set to Typ[Invalid] in case of an error.
// Precondition: len(lhs) > 0 .
//
func (check *checker) assignNtoM(lhs, rhs []ast.Expr, decl bool, iota int) {
	assert(len(lhs) > 0)

	// If the lhs and rhs have corresponding expressions, treat each
	// matching pair as an individual pair.
	if len(lhs) == len(rhs) {
		for i, e := range rhs {
			check.assign1to1(lhs[i], e, nil, decl, iota)
		}
		return
	}

	// Otherwise, the rhs must be a single expression (possibly
	// a function call returning multiple values, or a comma-ok
	// expression).
	if len(rhs) == 1 {
		// len(lhs) > 1
		// Start with rhs so we have expression types
		// for declarations with implicit types.
		var x operand
		check.expr(&x, rhs[0], nil, iota)
		if x.mode == invalid {
			goto Error
		}

		if t, _ := x.typ.(*Result); t != nil && len(lhs) == len(t.Values) {
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

			x.typ = Typ[UntypedBool]
			check.assign1to1(lhs[1], nil, &x, decl, iota)
			return
		}
	}

	check.errorf(lhs[0].Pos(), "assignment count mismatch: %d = %d", len(lhs), len(rhs))

Error:
	// In case of a declaration, set all lhs types to Typ[Invalid].
	if decl {
		for _, e := range lhs {
			ident, _ := e.(*ast.Ident)
			if ident == nil {
				check.errorf(e.Pos(), "cannot declare %s", e)
				continue
			}
			switch obj := check.lookup(ident).(type) {
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
		if tch, ok := underlying(ch.typ).(*Chan); !ok || tch.Dir&ast.SEND == 0 || !check.assignment(&x, tch.Elt) {
			if x.mode != invalid {
				check.invalidOp(ch.pos(), "cannot send %s to channel %s", &x, &ch)
			}
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
		var x operand
		Y := &ast.BasicLit{ValuePos: s.X.Pos(), Kind: token.INT, Value: "1"} // use x's position
		check.binary(&x, s.X, Y, op, -1)
		if x.mode == invalid {
			return
		}
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
			var x operand
			check.binary(&x, s.Lhs[0], s.Rhs[0], op, -1)
			if x.mode == invalid {
				return
			}
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
				check.register(name, &Var{Name: res.Name, Type: res.Type}) // Pkg == nil
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
		if x.mode != invalid && !isBoolean(x.typ) {
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
		// TODO(gri) check also correct use of fallthrough
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
			if x.mode != invalid && !isBoolean(x.typ) {
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
			x.expr = s.Key
			check.assign1to1(s.Key, nil, &x, decl, -1)
		} else {
			check.invalidAST(s.Pos(), "range clause requires index iteration variable")
			// ok to continue
		}
		if s.Value != nil {
			x.typ = val
			x.expr = s.Value
			check.assign1to1(s.Value, nil, &x, decl, -1)
		}

		check.stmt(s.Body)

	default:
		check.errorf(s.Pos(), "invalid statement")
	}
}
