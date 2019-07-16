// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
	"sort"
)

func (check *Checker) funcBody(decl *declInfo, name string, sig *Signature, body *ast.BlockStmt, iota constant.Value) {
	if trace {
		check.trace(body.Pos(), "--- %s: %s", name, sig)
		defer func() {
			check.trace(body.End(), "--- <end>")
		}()
	}

	// set function scope extent
	sig.scope.pos = body.Pos()
	sig.scope.end = body.End()

	// save/restore current context and setup function context
	// (and use 0 indentation at function start)
	defer func(ctxt context, indent int) {
		check.context = ctxt
		check.indent = indent
	}(check.context, check.indent)
	check.context = context{
		decl:  decl,
		scope: sig.scope,
		iota:  iota,
		sig:   sig,
	}
	check.indent = 0

	check.stmtList(0, body.List)

	if check.hasLabel {
		check.labels(body)
	}

	if sig.results.Len() > 0 && !check.isTerminating(body, "") {
		check.error(body.Rbrace, "missing return")
	}

	// spec: "Implementation restriction: A compiler may make it illegal to
	// declare a variable inside a function body if the variable is never used."
	check.usage(sig.scope)
}

func (check *Checker) usage(scope *Scope) {
	var unused []*Var
	for _, elem := range scope.elems {
		if v, _ := elem.(*Var); v != nil && !v.used {
			unused = append(unused, v)
		}
	}
	sort.Slice(unused, func(i, j int) bool {
		return unused[i].pos < unused[j].pos
	})
	for _, v := range unused {
		check.softErrorf(v.pos, "%s declared but not used", v.name)
	}

	for _, scope := range scope.children {
		// Don't go inside function literal scopes a second time;
		// they are handled explicitly by funcBody.
		if !scope.isFunc {
			check.usage(scope)
		}
	}
}

// stmtContext is a bitset describing which
// control-flow statements are permissible,
// and provides additional context information
// for better error messages.
type stmtContext uint

const (
	// permissible control-flow statements
	breakOk stmtContext = 1 << iota
	continueOk
	fallthroughOk

	// additional context information
	finalSwitchCase
)

func (check *Checker) simpleStmt(s ast.Stmt) {
	if s != nil {
		check.stmt(0, s)
	}
}

func trimTrailingEmptyStmts(list []ast.Stmt) []ast.Stmt {
	for i := len(list); i > 0; i-- {
		if _, ok := list[i-1].(*ast.EmptyStmt); !ok {
			return list[:i]
		}
	}
	return nil
}

func (check *Checker) stmtList(ctxt stmtContext, list []ast.Stmt) {
	ok := ctxt&fallthroughOk != 0
	inner := ctxt &^ fallthroughOk
	list = trimTrailingEmptyStmts(list) // trailing empty statements are "invisible" to fallthrough analysis
	for i, s := range list {
		inner := inner
		if ok && i+1 == len(list) {
			inner |= fallthroughOk
		}
		check.stmt(inner, s)
	}
}

func (check *Checker) multipleDefaults(list []ast.Stmt) {
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
				check.errorf(d.Pos(), "multiple defaults (first at %s)", check.fset.Position(first.Pos()))
			} else {
				first = d
			}
		}
	}
}

func (check *Checker) openScope(s ast.Stmt, comment string) {
	scope := NewScope(check.scope, s.Pos(), s.End(), comment)
	check.recordScope(s, scope)
	check.scope = scope
}

func (check *Checker) closeScope() {
	check.scope = check.scope.Parent()
}

func assignOp(op token.Token) token.Token {
	// token_test.go verifies the token ordering this function relies on
	if token.ADD_ASSIGN <= op && op <= token.AND_NOT_ASSIGN {
		return op + (token.ADD - token.ADD_ASSIGN)
	}
	return token.ILLEGAL
}

func (check *Checker) suspendedCall(keyword string, call *ast.CallExpr) {
	var x operand
	var msg string
	switch check.rawExpr(&x, call, nil) {
	case conversion:
		msg = "requires function call, not conversion"
	case expression:
		msg = "discards result of"
	case statement:
		return
	default:
		unreachable()
	}
	check.errorf(x.pos(), "%s %s %s", keyword, msg, &x)
}

// goVal returns the Go value for val, or nil.
func goVal(val constant.Value) interface{} {
	// val should exist, but be conservative and check
	if val == nil {
		return nil
	}
	// Match implementation restriction of other compilers.
	// gc only checks duplicates for integer, floating-point
	// and string values, so only create Go values for these
	// types.
	switch val.Kind() {
	case constant.Int:
		if x, ok := constant.Int64Val(val); ok {
			return x
		}
		if x, ok := constant.Uint64Val(val); ok {
			return x
		}
	case constant.Float:
		if x, ok := constant.Float64Val(val); ok {
			return x
		}
	case constant.String:
		return constant.StringVal(val)
	}
	return nil
}

// A valueMap maps a case value (of a basic Go type) to a list of positions
// where the same case value appeared, together with the corresponding case
// types.
// Since two case values may have the same "underlying" value but different
// types we need to also check the value's types (e.g., byte(1) vs myByte(1))
// when the switch expression is of interface type.
type (
	valueMap  map[interface{}][]valueType // underlying Go value -> valueType
	valueType struct {
		pos token.Pos
		typ Type
	}
)

func (check *Checker) caseValues(x *operand, values []ast.Expr, seen valueMap) {
L:
	for _, e := range values {
		var v operand
		check.expr(&v, e)
		if x.mode == invalid || v.mode == invalid {
			continue L
		}
		check.convertUntyped(&v, x.typ)
		if v.mode == invalid {
			continue L
		}
		// Order matters: By comparing v against x, error positions are at the case values.
		res := v // keep original v unchanged
		check.comparison(&res, x, token.EQL)
		if res.mode == invalid {
			continue L
		}
		if v.mode != constant_ {
			continue L // we're done
		}
		// look for duplicate values
		if val := goVal(v.val); val != nil {
			// look for duplicate types for a given value
			// (quadratic algorithm, but these lists tend to be very short)
			for _, vt := range seen[val] {
				if Identical(v.typ, vt.typ) {
					check.errorf(v.pos(), "duplicate case %s in expression switch", &v)
					check.error(vt.pos, "\tprevious case") // secondary error, \t indented
					continue L
				}
			}
			seen[val] = append(seen[val], valueType{v.pos(), v.typ})
		}
	}
}

func (check *Checker) caseTypes(x *operand, xtyp *Interface, types []ast.Expr, seen map[Type]token.Pos) (T Type) {
L:
	for _, e := range types {
		T = check.typOrNil(e)
		if T == Typ[Invalid] {
			continue L
		}
		// look for duplicate types
		// (quadratic algorithm, but type switches tend to be reasonably small)
		for t, pos := range seen {
			if T == nil && t == nil || T != nil && t != nil && Identical(T, t) {
				// talk about "case" rather than "type" because of nil case
				Ts := "nil"
				if T != nil {
					Ts = T.String()
				}
				check.errorf(e.Pos(), "duplicate case %s in type switch", Ts)
				check.error(pos, "\tprevious case") // secondary error, \t indented
				continue L
			}
		}
		seen[T] = e.Pos()
		if T != nil {
			check.typeAssertion(e.Pos(), x, xtyp, T)
		}
	}
	return
}

// stmt typechecks statement s.
func (check *Checker) stmt(ctxt stmtContext, s ast.Stmt) {
	// statements must end with the same top scope as they started with
	if debug {
		defer func(scope *Scope) {
			// don't check if code is panicking
			if p := recover(); p != nil {
				panic(p)
			}
			assert(scope == check.scope)
		}(check.scope)
	}

	// process collected function literals before scope changes
	defer check.processDelayed(len(check.delayed))

	inner := ctxt &^ (fallthroughOk | finalSwitchCase)
	switch s := s.(type) {
	case *ast.BadStmt, *ast.EmptyStmt:
		// ignore

	case *ast.DeclStmt:
		check.declStmt(s.Decl)

	case *ast.LabeledStmt:
		check.hasLabel = true
		check.stmt(ctxt, s.Stmt)

	case *ast.ExprStmt:
		// spec: "With the exception of specific built-in functions,
		// function and method calls and receive operations can appear
		// in statement context. Such statements may be parenthesized."
		var x operand
		kind := check.rawExpr(&x, s.X, nil)
		var msg string
		switch x.mode {
		default:
			if kind == statement {
				return
			}
			msg = "is not used"
		case builtin:
			msg = "must be called"
		case typexpr:
			msg = "is not an expression"
		}
		check.errorf(x.pos(), "%s %s", &x, msg)

	case *ast.SendStmt:
		var ch, x operand
		check.expr(&ch, s.Chan)
		check.expr(&x, s.Value)
		if ch.mode == invalid || x.mode == invalid {
			return
		}

		tch, ok := ch.typ.Underlying().(*Chan)
		if !ok {
			check.invalidOp(s.Arrow, "cannot send to non-chan type %s", ch.typ)
			return
		}

		if tch.dir == RecvOnly {
			check.invalidOp(s.Arrow, "cannot send to receive-only type %s", tch)
			return
		}

		check.assignment(&x, tch.elem, "send")

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
		check.expr(&x, s.X)
		if x.mode == invalid {
			return
		}
		if !isNumeric(x.typ) {
			check.invalidOp(s.X.Pos(), "%s%s (non-numeric type %s)", s.X, s.Tok, x.typ)
			return
		}

		Y := &ast.BasicLit{ValuePos: s.X.Pos(), Kind: token.INT, Value: "1"} // use x's position
		check.binary(&x, nil, s.X, Y, op)
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
				check.shortVarDecl(s.TokPos, s.Lhs, s.Rhs)
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
			op := assignOp(s.Tok)
			if op == token.ILLEGAL {
				check.invalidAST(s.TokPos, "unknown assignment operation %s", s.Tok)
				return
			}
			var x operand
			check.binary(&x, nil, s.Lhs[0], s.Rhs[0], op)
			if x.mode == invalid {
				return
			}
			check.assignVar(s.Lhs[0], &x)
		}

	case *ast.GoStmt:
		check.suspendedCall("go", s.Call)

	case *ast.DeferStmt:
		check.suspendedCall("defer", s.Call)

	case *ast.ReturnStmt:
		res := check.sig.results
		if res.Len() > 0 {
			// function returns results
			// (if one, say the first, result parameter is named, all of them are named)
			if len(s.Results) == 0 && res.vars[0].name != "" {
				// spec: "Implementation restriction: A compiler may disallow an empty expression
				// list in a "return" statement if a different entity (constant, type, or variable)
				// with the same name as a result parameter is in scope at the place of the return."
				for _, obj := range res.vars {
					if alt := check.lookup(obj.name); alt != nil && alt != obj {
						check.errorf(s.Pos(), "result parameter %s not in scope at return", obj.name)
						check.errorf(alt.Pos(), "\tinner declaration of %s", obj)
						// ok to continue
					}
				}
			} else {
				// return has results or result parameters are unnamed
				check.initVars(res.vars, s.Results, s.Return)
			}
		} else if len(s.Results) > 0 {
			check.error(s.Results[0].Pos(), "no result values expected")
			check.use(s.Results...)
		}

	case *ast.BranchStmt:
		if s.Label != nil {
			check.hasLabel = true
			return // checked in 2nd pass (check.labels)
		}
		switch s.Tok {
		case token.BREAK:
			if ctxt&breakOk == 0 {
				check.error(s.Pos(), "break not in for, switch, or select statement")
			}
		case token.CONTINUE:
			if ctxt&continueOk == 0 {
				check.error(s.Pos(), "continue not in for statement")
			}
		case token.FALLTHROUGH:
			if ctxt&fallthroughOk == 0 {
				msg := "fallthrough statement out of place"
				if ctxt&finalSwitchCase != 0 {
					msg = "cannot fallthrough final case in switch"
				}
				check.error(s.Pos(), msg)
			}
		default:
			check.invalidAST(s.Pos(), "branch statement: %s", s.Tok)
		}

	case *ast.BlockStmt:
		check.openScope(s, "block")
		defer check.closeScope()

		check.stmtList(inner, s.List)

	case *ast.IfStmt:
		check.openScope(s, "if")
		defer check.closeScope()

		check.simpleStmt(s.Init)
		var x operand
		check.expr(&x, s.Cond)
		if x.mode != invalid && !isBoolean(x.typ) {
			check.error(s.Cond.Pos(), "non-boolean condition in if statement")
		}
		check.stmt(inner, s.Body)
		// The parser produces a correct AST but if it was modified
		// elsewhere the else branch may be invalid. Check again.
		switch s.Else.(type) {
		case nil, *ast.BadStmt:
			// valid or error already reported
		case *ast.IfStmt, *ast.BlockStmt:
			check.stmt(inner, s.Else)
		default:
			check.error(s.Else.Pos(), "invalid else branch in if statement")
		}

	case *ast.SwitchStmt:
		inner |= breakOk
		check.openScope(s, "switch")
		defer check.closeScope()

		check.simpleStmt(s.Init)
		var x operand
		if s.Tag != nil {
			check.expr(&x, s.Tag)
			// By checking assignment of x to an invisible temporary
			// (as a compiler would), we get all the relevant checks.
			check.assignment(&x, nil, "switch expression")
		} else {
			// spec: "A missing switch expression is
			// equivalent to the boolean value true."
			x.mode = constant_
			x.typ = Typ[Bool]
			x.val = constant.MakeBool(true)
			x.expr = &ast.Ident{NamePos: s.Body.Lbrace, Name: "true"}
		}

		check.multipleDefaults(s.Body.List)

		seen := make(valueMap) // map of seen case values to positions and types
		for i, c := range s.Body.List {
			clause, _ := c.(*ast.CaseClause)
			if clause == nil {
				check.invalidAST(c.Pos(), "incorrect expression switch case")
				continue
			}
			check.caseValues(&x, clause.List, seen)
			check.openScope(clause, "case")
			inner := inner
			if i+1 < len(s.Body.List) {
				inner |= fallthroughOk
			} else {
				inner |= finalSwitchCase
			}
			check.stmtList(inner, clause.Body)
			check.closeScope()
		}

	case *ast.TypeSwitchStmt:
		inner |= breakOk
		check.openScope(s, "type switch")
		defer check.closeScope()

		check.simpleStmt(s.Init)

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

			if lhs.Name == "_" {
				// _ := x.(type) is an invalid short variable declaration
				check.softErrorf(lhs.Pos(), "no new variable on left side of :=")
				lhs = nil // avoid declared but not used error below
			} else {
				check.recordDef(lhs, nil) // lhs variable is implicitly declared in each cause clause
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
		xtyp, _ := x.typ.Underlying().(*Interface)
		if xtyp == nil {
			check.errorf(x.pos(), "%s is not an interface", &x)
			return
		}

		check.multipleDefaults(s.Body.List)

		var lhsVars []*Var               // list of implicitly declared lhs variables
		seen := make(map[Type]token.Pos) // map of seen types to positions
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CaseClause)
			if clause == nil {
				check.invalidAST(s.Pos(), "incorrect type switch case")
				continue
			}
			// Check each type in this type switch case.
			T := check.caseTypes(&x, xtyp, clause.List, seen)
			check.openScope(clause, "case")
			// If lhs exists, declare a corresponding variable in the case-local scope.
			if lhs != nil {
				// spec: "The TypeSwitchGuard may include a short variable declaration.
				// When that form is used, the variable is declared at the beginning of
				// the implicit block in each clause. In clauses with a case listing
				// exactly one type, the variable has that type; otherwise, the variable
				// has the type of the expression in the TypeSwitchGuard."
				if len(clause.List) != 1 || T == nil {
					T = x.typ
				}
				obj := NewVar(lhs.Pos(), check.pkg, lhs.Name, T)
				scopePos := clause.Pos() + token.Pos(len("default")) // for default clause (len(List) == 0)
				if n := len(clause.List); n > 0 {
					scopePos = clause.List[n-1].End()
				}
				check.declare(check.scope, nil, obj, scopePos)
				check.recordImplicit(clause, obj)
				// For the "declared but not used" error, all lhs variables act as
				// one; i.e., if any one of them is 'used', all of them are 'used'.
				// Collect them for later analysis.
				lhsVars = append(lhsVars, obj)
			}
			check.stmtList(inner, clause.Body)
			check.closeScope()
		}

		// If lhs exists, we must have at least one lhs variable that was used.
		if lhs != nil {
			var used bool
			for _, v := range lhsVars {
				if v.used {
					used = true
				}
				v.used = true // avoid usage error when checking entire function
			}
			if !used {
				check.softErrorf(lhs.Pos(), "%s declared but not used", lhs.Name)
			}
		}

	case *ast.SelectStmt:
		inner |= breakOk

		check.multipleDefaults(s.Body.List)

		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CommClause)
			if clause == nil {
				continue // error reported before
			}

			// clause.Comm must be a SendStmt, RecvStmt, or default case
			valid := false
			var rhs ast.Expr // rhs of RecvStmt, or nil
			switch s := clause.Comm.(type) {
			case nil, *ast.SendStmt:
				valid = true
			case *ast.AssignStmt:
				if len(s.Rhs) == 1 {
					rhs = s.Rhs[0]
				}
			case *ast.ExprStmt:
				rhs = s.X
			}

			// if present, rhs must be a receive operation
			if rhs != nil {
				if x, _ := unparen(rhs).(*ast.UnaryExpr); x != nil && x.Op == token.ARROW {
					valid = true
				}
			}

			if !valid {
				check.error(clause.Comm.Pos(), "select case must be send or receive (possibly with assignment)")
				continue
			}

			check.openScope(s, "case")
			if clause.Comm != nil {
				check.stmt(inner, clause.Comm)
			}
			check.stmtList(inner, clause.Body)
			check.closeScope()
		}

	case *ast.ForStmt:
		inner |= breakOk | continueOk
		check.openScope(s, "for")
		defer check.closeScope()

		check.simpleStmt(s.Init)
		if s.Cond != nil {
			var x operand
			check.expr(&x, s.Cond)
			if x.mode != invalid && !isBoolean(x.typ) {
				check.error(s.Cond.Pos(), "non-boolean condition in for statement")
			}
		}
		check.simpleStmt(s.Post)
		// spec: "The init statement may be a short variable
		// declaration, but the post statement must not."
		if s, _ := s.Post.(*ast.AssignStmt); s != nil && s.Tok == token.DEFINE {
			check.softErrorf(s.Pos(), "cannot declare in post statement")
			// Don't call useLHS here because we want to use the lhs in
			// this erroneous statement so that we don't get errors about
			// these lhs variables being declared but not used.
			check.use(s.Lhs...) // avoid follow-up errors
		}
		check.stmt(inner, s.Body)

	case *ast.RangeStmt:
		inner |= breakOk | continueOk
		check.openScope(s, "for")
		defer check.closeScope()

		// check expression to iterate over
		var x operand
		check.expr(&x, s.X)

		// determine key/value types
		var key, val Type
		if x.mode != invalid {
			switch typ := x.typ.Underlying().(type) {
			case *Basic:
				if isString(typ) {
					key = Typ[Int]
					val = universeRune // use 'rune' name
				}
			case *Array:
				key = Typ[Int]
				val = typ.elem
			case *Slice:
				key = Typ[Int]
				val = typ.elem
			case *Pointer:
				if typ, _ := typ.base.Underlying().(*Array); typ != nil {
					key = Typ[Int]
					val = typ.elem
				}
			case *Map:
				key = typ.key
				val = typ.elem
			case *Chan:
				key = typ.elem
				val = Typ[Invalid]
				if typ.dir == SendOnly {
					check.errorf(x.pos(), "cannot range over send-only channel %s", &x)
					// ok to continue
				}
				if s.Value != nil {
					check.errorf(s.Value.Pos(), "iteration over %s permits only one iteration variable", &x)
					// ok to continue
				}
			}
		}

		if key == nil {
			check.errorf(x.pos(), "cannot range over %s", &x)
			// ok to continue
		}

		// check assignment to/declaration of iteration variables
		// (irregular assignment, cannot easily map to existing assignment checks)

		// lhs expressions and initialization value (rhs) types
		lhs := [2]ast.Expr{s.Key, s.Value}
		rhs := [2]Type{key, val} // key, val may be nil

		if s.Tok == token.DEFINE {
			// short variable declaration; variable scope starts after the range clause
			// (the for loop opens a new scope, so variables on the lhs never redeclare
			// previously declared variables)
			var vars []*Var
			for i, lhs := range lhs {
				if lhs == nil {
					continue
				}

				// determine lhs variable
				var obj *Var
				if ident, _ := lhs.(*ast.Ident); ident != nil {
					// declare new variable
					name := ident.Name
					obj = NewVar(ident.Pos(), check.pkg, name, nil)
					check.recordDef(ident, obj)
					// _ variables don't count as new variables
					if name != "_" {
						vars = append(vars, obj)
					}
				} else {
					check.errorf(lhs.Pos(), "cannot declare %s", lhs)
					obj = NewVar(lhs.Pos(), check.pkg, "_", nil) // dummy variable
				}

				// initialize lhs variable
				if typ := rhs[i]; typ != nil {
					x.mode = value
					x.expr = lhs // we don't have a better rhs expression to use here
					x.typ = typ
					check.initVar(obj, &x, "range clause")
				} else {
					obj.typ = Typ[Invalid]
					obj.used = true // don't complain about unused variable
				}
			}

			// declare variables
			if len(vars) > 0 {
				scopePos := s.X.End()
				for _, obj := range vars {
					// spec: "The scope of a constant or variable identifier declared inside
					// a function begins at the end of the ConstSpec or VarSpec (ShortVarDecl
					// for short variable declarations) and ends at the end of the innermost
					// containing block."
					check.declare(check.scope, nil /* recordDef already called */, obj, scopePos)
				}
			} else {
				check.error(s.TokPos, "no new variables on left side of :=")
			}
		} else {
			// ordinary assignment
			for i, lhs := range lhs {
				if lhs == nil {
					continue
				}
				if typ := rhs[i]; typ != nil {
					x.mode = value
					x.expr = lhs // we don't have a better rhs expression to use here
					x.typ = typ
					check.assignVar(lhs, &x)
				}
			}
		}

		check.stmt(inner, s.Body)

	default:
		check.error(s.Pos(), "invalid statement")
	}
}
