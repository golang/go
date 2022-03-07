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
	if check.conf.IgnoreFuncBodies {
		panic("function body not ignored")
	}

	if trace {
		check.trace(body.Pos(), "--- %s: %s", name, sig)
		defer func() {
			check.trace(body.End(), "--- <end>")
		}()
	}

	// set function scope extent
	sig.scope.pos = body.Pos()
	sig.scope.end = body.End()

	// save/restore current environment and set up function environment
	// (and use 0 indentation at function start)
	defer func(env environment, indent int) {
		check.environment = env
		check.indent = indent
	}(check.environment, check.indent)
	check.environment = environment{
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
		check.error(atPos(body.Rbrace), _MissingReturn, "missing return")
	}

	// spec: "Implementation restriction: A compiler may make it illegal to
	// declare a variable inside a function body if the variable is never used."
	check.usage(sig.scope)
}

func (check *Checker) usage(scope *Scope) {
	var unused []*Var
	for name, elem := range scope.elems {
		elem = resolve(name, elem)
		if v, _ := elem.(*Var); v != nil && !v.used {
			unused = append(unused, v)
		}
	}
	sort.Slice(unused, func(i, j int) bool {
		return unused[i].pos < unused[j].pos
	})
	for _, v := range unused {
		check.softErrorf(v, _UnusedVar, "%s declared but not used", v.name)
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
			check.invalidAST(s, "case/communication clause expected")
		}
		if d != nil {
			if first != nil {
				check.errorf(d, _DuplicateDefault, "multiple defaults (first at %s)", check.fset.Position(first.Pos()))
			} else {
				first = d
			}
		}
	}
}

func (check *Checker) openScope(node ast.Node, comment string) {
	scope := NewScope(check.scope, node.Pos(), node.End(), comment)
	check.recordScope(node, scope)
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
	var code errorCode
	switch check.rawExpr(&x, call, nil, false) {
	case conversion:
		msg = "requires function call, not conversion"
		code = _InvalidDefer
		if keyword == "go" {
			code = _InvalidGo
		}
	case expression:
		msg = "discards result of"
		code = _UnusedResults
	case statement:
		return
	default:
		unreachable()
	}
	check.errorf(&x, code, "%s %s %s", keyword, msg, &x)
}

// goVal returns the Go value for val, or nil.
func goVal(val constant.Value) any {
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
	valueMap  map[any][]valueType // underlying Go value -> valueType
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
		check.comparison(&res, x, token.EQL, true)
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
					check.errorf(&v, _DuplicateCase, "duplicate case %s in expression switch", &v)
					check.error(atPos(vt.pos), _DuplicateCase, "\tprevious case") // secondary error, \t indented
					continue L
				}
			}
			seen[val] = append(seen[val], valueType{v.Pos(), v.typ})
		}
	}
}

// isNil reports whether the expression e denotes the predeclared value nil.
func (check *Checker) isNil(e ast.Expr) bool {
	// The only way to express the nil value is by literally writing nil (possibly in parentheses).
	if name, _ := unparen(e).(*ast.Ident); name != nil {
		_, ok := check.lookup(name.Name).(*Nil)
		return ok
	}
	return false
}

// If the type switch expression is invalid, x is nil.
func (check *Checker) caseTypes(x *operand, types []ast.Expr, seen map[Type]ast.Expr) (T Type) {
	var dummy operand
L:
	for _, e := range types {
		// The spec allows the value nil instead of a type.
		if check.isNil(e) {
			T = nil
			check.expr(&dummy, e) // run e through expr so we get the usual Info recordings
		} else {
			T = check.varType(e)
			if T == Typ[Invalid] {
				continue L
			}
		}
		// look for duplicate types
		// (quadratic algorithm, but type switches tend to be reasonably small)
		for t, other := range seen {
			if T == nil && t == nil || T != nil && t != nil && Identical(T, t) {
				// talk about "case" rather than "type" because of nil case
				Ts := "nil"
				if T != nil {
					Ts = TypeString(T, check.qualifier)
				}
				check.errorf(e, _DuplicateCase, "duplicate case %s in type switch", Ts)
				check.error(other, _DuplicateCase, "\tprevious case") // secondary error, \t indented
				continue L
			}
		}
		seen[T] = e
		if x != nil && T != nil {
			check.typeAssertion(e, x, T, true)
		}
	}
	return
}

// TODO(gri) Once we are certain that typeHash is correct in all situations, use this version of caseTypes instead.
//           (Currently it may be possible that different types have identical names and import paths due to ImporterFrom.)
//
// func (check *Checker) caseTypes(x *operand, xtyp *Interface, types []ast.Expr, seen map[string]ast.Expr) (T Type) {
// 	var dummy operand
// L:
// 	for _, e := range types {
// 		// The spec allows the value nil instead of a type.
// 		var hash string
// 		if check.isNil(e) {
// 			check.expr(&dummy, e) // run e through expr so we get the usual Info recordings
// 			T = nil
// 			hash = "<nil>" // avoid collision with a type named nil
// 		} else {
// 			T = check.varType(e)
// 			if T == Typ[Invalid] {
// 				continue L
// 			}
// 			hash = typeHash(T, nil)
// 		}
// 		// look for duplicate types
// 		if other := seen[hash]; other != nil {
// 			// talk about "case" rather than "type" because of nil case
// 			Ts := "nil"
// 			if T != nil {
// 				Ts = TypeString(T, check.qualifier)
// 			}
// 			var err error_
// 			err.errorf(e, "duplicate case %s in type switch", Ts)
// 			err.errorf(other, "previous case")
// 			check.report(&err)
// 			continue L
// 		}
// 		seen[hash] = e
// 		if T != nil {
// 			check.typeAssertion(e.Pos(), x, xtyp, T)
// 		}
// 	}
// 	return
// }

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
		kind := check.rawExpr(&x, s.X, nil, false)
		var msg string
		var code errorCode
		switch x.mode {
		default:
			if kind == statement {
				return
			}
			msg = "is not used"
			code = _UnusedExpr
		case builtin:
			msg = "must be called"
			code = _UncalledBuiltin
		case typexpr:
			msg = "is not an expression"
			code = _NotAnExpr
		}
		check.errorf(&x, code, "%s %s", &x, msg)

	case *ast.SendStmt:
		var ch, val operand
		check.expr(&ch, s.Chan)
		check.expr(&val, s.Value)
		if ch.mode == invalid || val.mode == invalid {
			return
		}
		u := coreType(ch.typ)
		if u == nil {
			check.invalidOp(inNode(s, s.Arrow), _InvalidSend, "cannot send to %s: no core type", &ch)
			return
		}
		uch, _ := u.(*Chan)
		if uch == nil {
			check.invalidOp(inNode(s, s.Arrow), _InvalidSend, "cannot send to non-channel %s", &ch)
			return
		}
		if uch.dir == RecvOnly {
			check.invalidOp(inNode(s, s.Arrow), _InvalidSend, "cannot send to receive-only channel %s", &ch)
			return
		}
		check.assignment(&val, uch.elem, "send")

	case *ast.IncDecStmt:
		var op token.Token
		switch s.Tok {
		case token.INC:
			op = token.ADD
		case token.DEC:
			op = token.SUB
		default:
			check.invalidAST(inNode(s, s.TokPos), "unknown inc/dec operation %s", s.Tok)
			return
		}

		var x operand
		check.expr(&x, s.X)
		if x.mode == invalid {
			return
		}
		if !allNumeric(x.typ) {
			check.invalidOp(s.X, _NonNumericIncDec, "%s%s (non-numeric type %s)", s.X, s.Tok, x.typ)
			return
		}

		Y := &ast.BasicLit{ValuePos: s.X.Pos(), Kind: token.INT, Value: "1"} // use x's position
		check.binary(&x, nil, s.X, Y, op, s.TokPos)
		if x.mode == invalid {
			return
		}
		check.assignVar(s.X, &x)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			if len(s.Lhs) == 0 {
				check.invalidAST(s, "missing lhs in assignment")
				return
			}
			if s.Tok == token.DEFINE {
				check.shortVarDecl(inNode(s, s.TokPos), s.Lhs, s.Rhs)
			} else {
				// regular assignment
				check.assignVars(s.Lhs, s.Rhs)
			}

		default:
			// assignment operations
			if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
				check.errorf(inNode(s, s.TokPos), _MultiValAssignOp, "assignment operation %s requires single-valued expressions", s.Tok)
				return
			}
			op := assignOp(s.Tok)
			if op == token.ILLEGAL {
				check.invalidAST(atPos(s.TokPos), "unknown assignment operation %s", s.Tok)
				return
			}
			var x operand
			check.binary(&x, nil, s.Lhs[0], s.Rhs[0], op, s.TokPos)
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
		// Return with implicit results allowed for function with named results.
		// (If one is named, all are named.)
		if len(s.Results) == 0 && res.Len() > 0 && res.vars[0].name != "" {
			// spec: "Implementation restriction: A compiler may disallow an empty expression
			// list in a "return" statement if a different entity (constant, type, or variable)
			// with the same name as a result parameter is in scope at the place of the return."
			for _, obj := range res.vars {
				if alt := check.lookup(obj.name); alt != nil && alt != obj {
					check.errorf(s, _OutOfScopeResult, "result parameter %s not in scope at return", obj.name)
					check.errorf(alt, _OutOfScopeResult, "\tinner declaration of %s", obj)
					// ok to continue
				}
			}
		} else {
			var lhs []*Var
			if res.Len() > 0 {
				lhs = res.vars
			}
			check.initVars(lhs, s.Results, s)
		}

	case *ast.BranchStmt:
		if s.Label != nil {
			check.hasLabel = true
			return // checked in 2nd pass (check.labels)
		}
		switch s.Tok {
		case token.BREAK:
			if ctxt&breakOk == 0 {
				check.error(s, _MisplacedBreak, "break not in for, switch, or select statement")
			}
		case token.CONTINUE:
			if ctxt&continueOk == 0 {
				check.error(s, _MisplacedContinue, "continue not in for statement")
			}
		case token.FALLTHROUGH:
			if ctxt&fallthroughOk == 0 {
				msg := "fallthrough statement out of place"
				code := _MisplacedFallthrough
				if ctxt&finalSwitchCase != 0 {
					msg = "cannot fallthrough final case in switch"
				}
				check.error(s, code, msg)
			}
		default:
			check.invalidAST(s, "branch statement: %s", s.Tok)
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
		if x.mode != invalid && !allBoolean(x.typ) {
			check.error(s.Cond, _InvalidCond, "non-boolean condition in if statement")
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
			check.invalidAST(s.Else, "invalid else branch in if statement")
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
			if x.mode != invalid && !Comparable(x.typ) && !hasNil(x.typ) {
				check.errorf(&x, _InvalidExprSwitch, "cannot switch on %s (%s is not comparable)", &x, x.typ)
				x.mode = invalid
			}
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
				check.invalidAST(c, "incorrect expression switch case")
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
				check.invalidAST(s, "incorrect form of type switch guard")
				return
			}

			lhs, _ = guard.Lhs[0].(*ast.Ident)
			if lhs == nil {
				check.invalidAST(s, "incorrect form of type switch guard")
				return
			}

			if lhs.Name == "_" {
				// _ := x.(type) is an invalid short variable declaration
				check.softErrorf(lhs, _NoNewVar, "no new variable on left side of :=")
				lhs = nil // avoid declared but not used error below
			} else {
				check.recordDef(lhs, nil) // lhs variable is implicitly declared in each cause clause
			}

			rhs = guard.Rhs[0]

		default:
			check.invalidAST(s, "incorrect form of type switch guard")
			return
		}

		// rhs must be of the form: expr.(type) and expr must be an ordinary interface
		expr, _ := rhs.(*ast.TypeAssertExpr)
		if expr == nil || expr.Type != nil {
			check.invalidAST(s, "incorrect form of type switch guard")
			return
		}
		var x operand
		check.expr(&x, expr.X)
		if x.mode == invalid {
			return
		}
		// TODO(gri) we may want to permit type switches on type parameter values at some point
		var sx *operand // switch expression against which cases are compared against; nil if invalid
		if isTypeParam(x.typ) {
			check.errorf(&x, _InvalidTypeSwitch, "cannot use type switch on type parameter value %s", &x)
		} else {
			if _, ok := under(x.typ).(*Interface); ok {
				sx = &x
			} else {
				check.errorf(&x, _InvalidTypeSwitch, "%s is not an interface", &x)
			}
		}

		check.multipleDefaults(s.Body.List)

		var lhsVars []*Var              // list of implicitly declared lhs variables
		seen := make(map[Type]ast.Expr) // map of seen types to positions
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CaseClause)
			if clause == nil {
				check.invalidAST(s, "incorrect type switch case")
				continue
			}
			// Check each type in this type switch case.
			T := check.caseTypes(sx, clause.List, seen)
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
				check.softErrorf(lhs, _UnusedVar, "%s declared but not used", lhs.Name)
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
				check.error(clause.Comm, _InvalidSelectCase, "select case must be send or receive (possibly with assignment)")
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
			if x.mode != invalid && !allBoolean(x.typ) {
				check.error(s.Cond, _InvalidCond, "non-boolean condition in for statement")
			}
		}
		check.simpleStmt(s.Post)
		// spec: "The init statement may be a short variable
		// declaration, but the post statement must not."
		if s, _ := s.Post.(*ast.AssignStmt); s != nil && s.Tok == token.DEFINE {
			check.softErrorf(s, _InvalidPostDecl, "cannot declare in post statement")
			// Don't call useLHS here because we want to use the lhs in
			// this erroneous statement so that we don't get errors about
			// these lhs variables being declared but not used.
			check.use(s.Lhs...) // avoid follow-up errors
		}
		check.stmt(inner, s.Body)

	case *ast.RangeStmt:
		inner |= breakOk | continueOk

		// check expression to iterate over
		var x operand
		check.expr(&x, s.X)

		// determine key/value types
		var key, val Type
		if x.mode != invalid {
			// Ranging over a type parameter is permitted if it has a core type.
			var cause string
			u := coreType(x.typ)
			switch t := u.(type) {
			case nil:
				cause = check.sprintf("%s has no core type", x.typ)
			case *Chan:
				if s.Value != nil {
					check.softErrorf(s.Value, _InvalidIterVar, "range over %s permits only one iteration variable", &x)
					// ok to continue
				}
				if t.dir == SendOnly {
					cause = "receive from send-only channel"
				}
			}
			key, val = rangeKeyVal(u)
			if key == nil || cause != "" {
				if cause == "" {
					check.softErrorf(&x, _InvalidRangeExpr, "cannot range over %s", &x)
				} else {
					check.softErrorf(&x, _InvalidRangeExpr, "cannot range over %s (%s)", &x, cause)
				}
				// ok to continue
			}
		}

		// Open the for-statement block scope now, after the range clause.
		// Iteration variables declared with := need to go in this scope (was issue #51437).
		check.openScope(s, "range")
		defer check.closeScope()

		// check assignment to/declaration of iteration variables
		// (irregular assignment, cannot easily map to existing assignment checks)

		// lhs expressions and initialization value (rhs) types
		lhs := [2]ast.Expr{s.Key, s.Value}
		rhs := [2]Type{key, val} // key, val may be nil

		if s.Tok == token.DEFINE {
			// short variable declaration
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
					check.invalidAST(lhs, "cannot declare %s", lhs)
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
				scopePos := s.Body.Pos()
				for _, obj := range vars {
					check.declare(check.scope, nil /* recordDef already called */, obj, scopePos)
				}
			} else {
				check.error(inNode(s, s.TokPos), _NoNewVar, "no new variables on left side of :=")
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
		check.invalidAST(s, "invalid statement")
	}
}

// rangeKeyVal returns the key and value type produced by a range clause
// over an expression of type typ. If the range clause is not permitted
// the results are nil.
func rangeKeyVal(typ Type) (key, val Type) {
	switch typ := arrayPtrDeref(typ).(type) {
	case *Basic:
		if isString(typ) {
			return Typ[Int], universeRune // use 'rune' name
		}
	case *Array:
		return Typ[Int], typ.elem
	case *Slice:
		return Typ[Int], typ.elem
	case *Map:
		return typ.key, typ.elem
	case *Chan:
		return typ.elem, Typ[Invalid]
	}
	return
}
