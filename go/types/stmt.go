// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
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

	if t, ok := x.typ.(*Tuple); ok {
		// TODO(gri) elsewhere we use "assignment count mismatch" (consolidate)
		check.errorf(x.pos(), "%d-valued expression %s used as single value", t.Len(), x)
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
func (check *checker) assign1to1(lhs, rhs ast.Expr, x *operand, decl bool, iota int, isConst bool) {
	assert(!isConst || decl)

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
			check.callIdent(ident, nil)
			// the rhs has its final type
			check.updateExprType(rhs, x.typ, true)
			return
		}

		var z operand
		check.expr(&z, lhs, nil, -1)
		if z.mode == invalid || z.typ == Typ[Invalid] {
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
	var obj Object
	var typ Type
	if isConst {
		obj = &Const{pos: ident.Pos(), pkg: check.pkg, name: ident.Name}
	} else {
		obj = &Var{pos: ident.Pos(), pkg: check.pkg, name: ident.Name}
	}
	defer check.declare(check.topScope, ident, obj)

	// TODO(gri) remove this switch, combine with code above
	switch obj := obj.(type) {
	default:
		unreachable()

	case *Const:
		typ = obj.typ // may already be Typ[Invalid]
		if typ == nil {
			typ = Typ[Invalid]
			if x.mode != invalid {
				typ = x.typ
			}
			obj.typ = typ
		}

	case *Var:
		typ = obj.typ // may already be Typ[Invalid]
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
			obj.typ = typ
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
		obj.val = exact.MakeUnknown() // failure case: we don't know the constant value
		if x.mode == constant {
			if isConstType(x.typ) {
				obj.val = x.val
			} else if x.typ != Typ[Invalid] {
				check.errorf(x.pos(), "%s has invalid constant type", x)
			}
		} else if x.mode != invalid {
			check.errorf(x.pos(), "%s is not constant", x)
		}
	}
}

// TODO(gri) assignNtoM is only used in one place now. remove and consolidate with other assignment functions.

// assignNtoM typechecks a general assignment. If decl is set, the lhs expressions
// must be identifiers; if their types are not set, they are deduced from the types
// of the corresponding rhs expressions, or set to Typ[Invalid] in case of an error.
// Precondition: len(lhs) > 0 .
//
func (check *checker) assignNtoM(lhs, rhs []ast.Expr, decl bool, iota int, isConst bool) {
	assert(len(lhs) > 0)
	assert(!isConst || decl)

	// If the lhs and rhs have corresponding expressions, treat each
	// matching pair as an individual pair.
	if len(lhs) == len(rhs) {
		for i, e := range rhs {
			check.assign1to1(lhs[i], e, nil, decl, iota, isConst)
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

		if t, ok := x.typ.(*Tuple); ok && len(lhs) == t.Len() {
			// function result
			x.mode = value
			for i := 0; i < len(lhs); i++ {
				obj := t.At(i)
				x.expr = nil // TODO(gri) should do better here
				x.typ = obj.typ
				check.assign1to1(lhs[i], nil, &x, decl, iota, isConst)
			}
			return
		}

		if x.mode == valueok && len(lhs) == 2 {
			// comma-ok expression
			x.mode = value
			check.assign1to1(lhs[0], nil, &x, decl, iota, isConst)

			x.typ = Typ[UntypedBool]
			check.assign1to1(lhs[1], nil, &x, decl, iota, isConst)
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

			var obj Object
			if isConst {
				obj = &Const{pos: ident.Pos(), pkg: check.pkg, name: ident.Name}
			} else {
				obj = &Var{pos: ident.Pos(), pkg: check.pkg, name: ident.Name}
			}
			defer check.declare(check.topScope, ident, obj)

			switch obj := obj.(type) {
			case *Const:
				obj.typ = Typ[Invalid]
			case *Var:
				obj.typ = Typ[Invalid]
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
		check.declStmt(s.Decl)

	case *ast.LabeledStmt:
		scope := check.funcsig.labels
		if scope == nil {
			scope = new(Scope) // no label scope chain
			check.funcsig.labels = scope
		}
		label := s.Label
		check.declare(scope, label, &Label{pos: label.Pos(), name: label.Name})
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
				if b, ok := x.typ.(*Builtin); ok && !b.isStatement {
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
		if tch, ok := ch.typ.Underlying().(*Chan); !ok || tch.dir&ast.SEND == 0 || !check.assignment(&x, tch.elt) {
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
		check.assign1to1(s.X, nil, &x, false, -1, false)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			if len(s.Lhs) == 0 {
				check.invalidAST(s.Pos(), "missing lhs in assignment")
				return
			}
			if s.Tok == token.DEFINE {
				// short variable declaration
				lhs := make([]Object, len(s.Lhs))
				for i, x := range s.Lhs {
					var obj Object
					if ident, ok := x.(*ast.Ident); ok {
						// use the correct obj if the ident is redeclared
						obj = &Var{pos: ident.Pos(), pkg: check.pkg, name: ident.Name}
						if alt := check.topScope.Lookup(ident.Name); alt != nil {
							obj = alt
						}
						check.callIdent(ident, obj)
					} else {
						check.errorf(x.Pos(), "cannot declare %s", x)
						// create a dummy variable
						obj = &Var{pos: x.Pos(), pkg: check.pkg, name: "_"}
					}
					lhs[i] = obj
				}
				check.assignMulti(lhs, s.Rhs)
				check.declareShort(check.topScope, lhs) // scope starts after the assignment

			} else {
				check.assignNtoM(s.Lhs, s.Rhs, s.Tok == token.DEFINE, -1, false)
			}

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
			check.assign1to1(s.Lhs[0], nil, &x, false, -1, false)
		}

	case *ast.GoStmt:
		check.call(s.Call)

	case *ast.DeferStmt:
		check.call(s.Call)

	case *ast.ReturnStmt:
		sig := check.funcsig
		if n := sig.results.Len(); n > 0 {
			// determine if the function has named results
			{
				named := false
				lhs := make([]Object, len(sig.results.vars))
				for i, res := range sig.results.vars {
					if res.name != "" {
						// a blank (_) result parameter is a named result
						named = true
					}
					lhs[i] = res
				}
				if len(s.Results) > 0 || !named {
					check.assignMulti(lhs, s.Results)
					return
				}
			}

			// TODO(gri) should not have to compute lhs, named every single time - clean this up
			lhs := make([]ast.Expr, n)
			named := false // if set, function has named results
			for i, res := range sig.results.vars {
				if len(res.name) > 0 {
					// a blank (_) result parameter is a named result
					named = true
				}
				name := ast.NewIdent(res.name)
				name.NamePos = s.Pos()
				lhs[i] = name
			}
			if len(s.Results) > 0 || !named {
				// TODO(gri) assignNtoM should perhaps not require len(lhs) > 0
				check.assignNtoM(lhs, s.Results, false, -1, false)
			}
		} else if len(s.Results) > 0 {
			check.errorf(s.Pos(), "no result values expected")
		}

	case *ast.BranchStmt:
		// TODO(gri) implement this

	case *ast.BlockStmt:
		check.openScope()
		check.stmtList(s.List)
		check.closeScope()

	case *ast.IfStmt:
		check.openScope()
		check.optionalStmt(s.Init)
		var x operand
		check.expr(&x, s.Cond, nil, -1)
		if x.mode != invalid && !isBoolean(x.typ) {
			check.errorf(s.Cond.Pos(), "non-boolean condition in if statement")
		}
		check.stmt(s.Body)
		check.optionalStmt(s.Else)
		check.closeScope()

	case *ast.SwitchStmt:
		check.openScope()
		check.optionalStmt(s.Init)
		var x operand
		tag := s.Tag
		if tag == nil {
			// use fake true tag value and position it at the opening { of the switch
			ident := &ast.Ident{NamePos: s.Body.Lbrace, Name: "true"}
			check.callIdent(ident, Universe.Lookup("true"))
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
			check.openScope()
			check.stmtList(clause.Body)
			check.closeScope()
		}
		check.closeScope()

	case *ast.TypeSwitchStmt:
		check.openScope()
		check.optionalStmt(s.Init)

		// A type switch guard must be of the form:
		//
		//     TypeSwitchGuard = [ identifier ":=" ] PrimaryExpr "." "(" "type" ")" .
		//
		// The parser is checking syntactic correctness;
		// remaining syntactic errors are considered AST errors here.
		// TODO(gri) better factoring of error handling (invalid ASTs)
		//
		var lhs *ast.Ident // lhs identifier or nil
		var rhs ast.Expr
		switch guard := s.Assign.(type) {
		case *ast.ExprStmt:
			rhs = guard.X
		case *ast.AssignStmt:
			if len(guard.Lhs) != 1 || guard.Tok != token.DEFINE || len(guard.Rhs) != 1 {
				check.invalidAST(s.Pos(), "incorrect form of type switch guard")
				return
			}

			lhs, _ = guard.Lhs[0].(*ast.Ident)
			if lhs == nil {
				check.invalidAST(s.Pos(), "incorrect form of type switch guard")
				return
			}

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
		if T, _ = x.typ.Underlying().(*Interface); T == nil {
			check.errorf(x.pos(), "%s is not an interface", &x)
			return
		}

		var obj Object
		if lhs != nil {
			obj = &Var{pos: lhs.Pos(), pkg: check.pkg, name: lhs.Name, typ: x.typ}
			check.declare(check.topScope, lhs, obj)
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
						check.errorf(expr.Pos(), msg, &x, typ, method.name)
						// ok to continue
					}
				}
			}
			check.openScope()
			// If lhs exists, declare a corresponding object in the case-local scope if necessary.
			if lhs != nil {
				// A single-type case clause implicitly declares a new variable shadowing lhs.
				if len(clause.List) == 1 && typ != nil {
					obj := &Var{pos: lhs.Pos(), pkg: check.pkg, name: lhs.Name, typ: typ}
					check.declare(check.topScope, nil, obj)
					check.callImplicitObj(clause, obj)
				}
			}
			check.stmtList(clause.Body)
			check.closeScope()
		}
		check.closeScope()

	case *ast.SelectStmt:
		check.multipleDefaults(s.Body.List)
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CommClause)
			if clause == nil {
				continue // error reported before
			}
			check.openScope()
			check.optionalStmt(clause.Comm) // TODO(gri) check correctness of c.Comm (must be Send/RecvStmt)
			check.stmtList(clause.Body)
			check.closeScope()
		}

	case *ast.ForStmt:
		check.openScope()
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
		check.closeScope()

	case *ast.RangeStmt:
		check.openScope()
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
		switch typ := x.typ.Underlying().(type) {
		case *Basic:
			if isString(typ) {
				key = Typ[UntypedInt]
				val = Typ[UntypedRune]
			}
		case *Array:
			key = Typ[UntypedInt]
			val = typ.elt
		case *Slice:
			key = Typ[UntypedInt]
			val = typ.elt
		case *Pointer:
			if typ, _ := typ.base.Underlying().(*Array); typ != nil {
				key = Typ[UntypedInt]
				val = typ.elt
			}
		case *Map:
			key = typ.key
			val = typ.elt
		case *Chan:
			key = typ.elt
			if typ.dir&ast.RECV == 0 {
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
			check.assign1to1(s.Key, nil, &x, decl, -1, false)
		} else {
			check.invalidAST(s.Pos(), "range clause requires index iteration variable")
			// ok to continue
		}
		if s.Value != nil {
			x.typ = val
			x.expr = s.Value
			check.assign1to1(s.Value, nil, &x, decl, -1, false)
		}

		check.stmt(s.Body)
		check.closeScope()

	default:
		check.errorf(s.Pos(), "invalid statement")
	}
}
