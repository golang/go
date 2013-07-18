// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/token"
)

func (check *checker) optionalStmt(s ast.Stmt) {
	if s != nil {
		scope := check.topScope
		check.stmt(s)
		assert(check.topScope == scope)
	}
}

func (check *checker) stmtList(list []ast.Stmt) {
	for _, s := range list {
		scope := check.topScope
		check.stmt(s)
		assert(check.topScope == scope)
	}
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

func (check *checker) openScope(node ast.Node) {
	s := NewScope(check.topScope)
	if retainASTLinks {
		s.node = node
	}
	check.topScope = s
}

func (check *checker) closeScope() {
	check.topScope = check.topScope.Parent()
}

// stmt typechecks statement s.
func (check *checker) stmt(s ast.Stmt) {
	// statements cannot use iota in general
	// (constant declarations set it explicitly)
	assert(check.iota == nil)

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
		check.declare(scope, label, NewLabel(label.Pos(), label.Name))
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
			//           below as part of s.X. Perhaps this can be avoided.)
			check.expr(&x, e.Fun)
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
		check.rawExpr(&x, s.X, nil)
		if x.mode == typexpr {
			check.errorf(x.pos(), "%s is not an expression", &x)
		}

	case *ast.SendStmt:
		var ch, x operand
		check.expr(&ch, s.Chan)
		check.expr(&x, s.Value)
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
		check.binary(&x, s.X, Y, op)
		if x.mode == invalid {
			return
		}
		check.assignVar(s.X, &x)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			if len(s.Lhs) == 0 {
				check.invalidAST(s.Pos(), "missing lhs in assignment")
				return
			}
			if s.Tok == token.DEFINE {
				check.shortVarDecl(s.Lhs, s.Rhs)
			} else {
				// regular assignment
				check.assignVars(s.Lhs, s.Rhs)
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
			check.binary(&x, s.Lhs[0], s.Rhs[0], op)
			if x.mode == invalid {
				return
			}
			check.assignVar(s.Lhs[0], &x)
		}

	case *ast.GoStmt:
		var x operand
		check.call(&x, s.Call)
		// TODO(gri) If a builtin is called, the builtin must be valid this context.

	case *ast.DeferStmt:
		var x operand
		check.call(&x, s.Call)
		// TODO(gri) If a builtin is called, the builtin must be valid this context.

	case *ast.ReturnStmt:
		sig := check.funcsig
		if n := sig.results.Len(); n > 0 {
			// determine if the function has named results
			named := false
			lhs := make([]*Var, n)
			for i, res := range sig.results.vars {
				if res.name != "" {
					// a blank (_) result parameter is a named result
					named = true
				}
				lhs[i] = res
			}
			if len(s.Results) > 0 || !named {
				check.initVars(lhs, s.Results, false)
				return
			}
		} else if len(s.Results) > 0 {
			check.errorf(s.Pos(), "no result values expected")
		}

	case *ast.BranchStmt:
		// TODO(gri) implement this

	case *ast.BlockStmt:
		check.openScope(s)
		check.stmtList(s.List)
		check.closeScope()

	case *ast.IfStmt:
		check.openScope(s)
		check.optionalStmt(s.Init)
		var x operand
		check.expr(&x, s.Cond)
		if x.mode != invalid && !isBoolean(x.typ) {
			check.errorf(s.Cond.Pos(), "non-boolean condition in if statement")
		}
		check.stmt(s.Body)
		check.optionalStmt(s.Else)
		check.closeScope()

	case *ast.SwitchStmt:
		check.openScope(s)
		check.optionalStmt(s.Init)
		var x operand
		tag := s.Tag
		if tag == nil {
			// use fake true tag value and position it at the opening { of the switch
			ident := &ast.Ident{NamePos: s.Body.Lbrace, Name: "true"}
			check.recordObject(ident, Universe.Lookup(nil, "true"))
			tag = ident
		}
		check.expr(&x, tag)

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
					check.expr(&y, expr)
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
			check.openScope(clause)
			check.stmtList(clause.Body)
			check.closeScope()
		}
		check.closeScope()

	case *ast.TypeSwitchStmt:
		check.openScope(s)
		defer check.closeScope()
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
		check.expr(&x, expr.X)
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
			obj = NewVar(lhs.Pos(), check.pkg, lhs.Name, x.typ)
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
				typ = check.typOrNil(expr)
				if typ != nil && typ != Typ[Invalid] {
					if method, wrongType := MissingMethod(typ, T); method != nil {
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
			check.openScope(clause)
			// If lhs exists, declare a corresponding object in the case-local scope if necessary.
			if lhs != nil {
				// A single-type case clause implicitly declares a new variable shadowing lhs.
				if len(clause.List) == 1 && typ != nil {
					obj := NewVar(lhs.Pos(), check.pkg, lhs.Name, typ)
					check.declare(check.topScope, nil, obj)
					check.recordImplicit(clause, obj)
				}
			}
			check.stmtList(clause.Body)
			check.closeScope()
		}

	case *ast.SelectStmt:
		check.multipleDefaults(s.Body.List)
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CommClause)
			if clause == nil {
				continue // error reported before
			}
			check.openScope(clause)
			check.optionalStmt(clause.Comm) // TODO(gri) check correctness of c.Comm (must be Send/RecvStmt)
			check.stmtList(clause.Body)
			check.closeScope()
		}

	case *ast.ForStmt:
		check.openScope(s)
		check.optionalStmt(s.Init)
		if s.Cond != nil {
			var x operand
			check.expr(&x, s.Cond)
			if x.mode != invalid && !isBoolean(x.typ) {
				check.errorf(s.Cond.Pos(), "non-boolean condition in for statement")
			}
		}
		check.optionalStmt(s.Post)
		check.stmt(s.Body)
		check.closeScope()

	case *ast.RangeStmt:
		check.openScope(s)
		defer check.closeScope()

		// check expression to iterate over
		decl := s.Tok == token.DEFINE
		var x operand
		check.expr(&x, s.X)
		if x.mode == invalid {
			// if we don't have a declaration, we can still check the loop's body
			// (otherwise we can't because we are missing the declared variables)
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
		// (irregular assignment, cannot easily map to existing assignment checks)
		if s.Key == nil {
			check.invalidAST(s.Pos(), "range clause requires index iteration variable")
			// ok to continue
		}

		// lhs expressions and initialization value (rhs) types
		lhs := [2]ast.Expr{s.Key, s.Value}
		rhs := [2]Type{key, val}

		if decl {
			// declaration; variable scope starts after the range clause
			var idents []*ast.Ident
			var vars []*Var
			for i, lhs := range lhs {
				if lhs == nil {
					continue
				}

				// determine lhs variable
				name := "_" // dummy, in case lhs is not an identifier
				ident, _ := lhs.(*ast.Ident)
				if ident != nil {
					name = ident.Name
				} else {
					check.errorf(lhs.Pos(), "cannot declare %s", lhs)
				}
				idents = append(idents, ident)

				obj := NewVar(lhs.Pos(), check.pkg, name, nil)
				vars = append(vars, obj)

				// initialize lhs variable
				x.mode = value
				x.expr = lhs // we don't have a better rhs expression to use here
				x.typ = rhs[i]
				check.initVar(obj, &x)
			}

			// declare variables
			for i, ident := range idents {
				check.declare(check.topScope, ident, vars[i])
			}
		} else {
			// ordinary assignment
			for i, lhs := range lhs {
				if lhs == nil {
					continue
				}
				x.mode = value
				x.expr = lhs // we don't have a better rhs expression to use here
				x.typ = rhs[i]
				check.assignVar(lhs, &x)
			}
		}

		check.stmt(s.Body)

	default:
		check.errorf(s.Pos(), "invalid statement")
	}
}
