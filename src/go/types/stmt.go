// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
	"internal/buildcfg"
	. "internal/types/errors"
	"sort"
)

func (check *Checker) funcBody(decl *declInfo, name string, sig *Signature, body *ast.BlockStmt, iota constant.Value) {
	if check.conf.IgnoreFuncBodies {
		panic("function body not ignored")
	}

	if check.conf._Trace {
		check.trace(body.Pos(), "-- %s: %s", name, sig)
	}

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
		check.error(atPos(body.Rbrace), MissingReturn, "missing return")
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
		return cmpPos(unused[i].pos, unused[j].pos) < 0
	})
	for _, v := range unused {
		check.softErrorf(v, UnusedVar, "declared and not used: %s", v.name)
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
	inTypeSwitch
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
			check.error(s, InvalidSyntaxTree, "case/communication clause expected")
		}
		if d != nil {
			if first != nil {
				check.errorf(d, DuplicateDefault, "multiple defaults (first at %s)", check.fset.Position(first.Pos()))
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
	var code Code
	switch check.rawExpr(nil, &x, call, nil, false) {
	case conversion:
		msg = "requires function call, not conversion"
		code = InvalidDefer
		if keyword == "go" {
			code = InvalidGo
		}
	case expression:
		msg = "discards result of"
		code = UnusedResults
	case statement:
		return
	default:
		panic("unreachable")
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
		check.expr(nil, &v, e)
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
					err := check.newError(DuplicateCase)
					err.addf(&v, "duplicate case %s in expression switch", &v)
					err.addf(atPos(vt.pos), "previous case")
					err.report()
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
	if name, _ := ast.Unparen(e).(*ast.Ident); name != nil {
		_, ok := check.lookup(name.Name).(*Nil)
		return ok
	}
	return false
}

// caseTypes typechecks the type expressions of a type case, checks for duplicate types
// using the seen map, and verifies that each type is valid with respect to the type of
// the operand x corresponding to the type switch expression. If that expression is not
// valid, x must be nil.
//
//	switch <x>.(type) {
//	case <types>: ...
//	...
//	}
//
// caseTypes returns the case-specific type for a variable v introduced through a short
// variable declaration by the type switch:
//
//	switch v := <x>.(type) {
//	case <types>: // T is the type of <v> in this case
//	...
//	}
//
// If there is exactly one type expression, T is the type of that expression. If there
// are multiple type expressions, or if predeclared nil is among the types, the result
// is the type of x. If x is invalid (nil), the result is the invalid type.
func (check *Checker) caseTypes(x *operand, types []ast.Expr, seen map[Type]ast.Expr) Type {
	var T Type
	var dummy operand
L:
	for _, e := range types {
		// The spec allows the value nil instead of a type.
		if check.isNil(e) {
			T = nil
			check.expr(nil, &dummy, e) // run e through expr so we get the usual Info recordings
		} else {
			T = check.varType(e)
			if !isValid(T) {
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
				err := check.newError(DuplicateCase)
				err.addf(e, "duplicate case %s in type switch", Ts)
				err.addf(other, "previous case")
				err.report()
				continue L
			}
		}
		seen[T] = e
		if x != nil && T != nil {
			check.typeAssertion(e, x, T, true)
		}
	}

	// spec: "In clauses with a case listing exactly one type, the variable has that type;
	// otherwise, the variable has the type of the expression in the TypeSwitchGuard.
	if len(types) != 1 || T == nil {
		T = Typ[Invalid]
		if x != nil {
			T = x.typ
		}
	}

	assert(T != nil)
	return T
}

// TODO(gri) Once we are certain that typeHash is correct in all situations, use this version of caseTypes instead.
// (Currently it may be possible that different types have identical names and import paths due to ImporterFrom.)
func (check *Checker) caseTypes_currently_unused(x *operand, xtyp *Interface, types []ast.Expr, seen map[string]ast.Expr) Type {
	var T Type
	var dummy operand
L:
	for _, e := range types {
		// The spec allows the value nil instead of a type.
		var hash string
		if check.isNil(e) {
			check.expr(nil, &dummy, e) // run e through expr so we get the usual Info recordings
			T = nil
			hash = "<nil>" // avoid collision with a type named nil
		} else {
			T = check.varType(e)
			if !isValid(T) {
				continue L
			}
			panic("enable typeHash(T, nil)")
			// hash = typeHash(T, nil)
		}
		// look for duplicate types
		if other := seen[hash]; other != nil {
			// talk about "case" rather than "type" because of nil case
			Ts := "nil"
			if T != nil {
				Ts = TypeString(T, check.qualifier)
			}
			err := check.newError(DuplicateCase)
			err.addf(e, "duplicate case %s in type switch", Ts)
			err.addf(other, "previous case")
			err.report()
			continue L
		}
		seen[hash] = e
		if T != nil {
			check.typeAssertion(e, x, T, true)
		}
	}

	// spec: "In clauses with a case listing exactly one type, the variable has that type;
	// otherwise, the variable has the type of the expression in the TypeSwitchGuard.
	if len(types) != 1 || T == nil {
		T = Typ[Invalid]
		if x != nil {
			T = x.typ
		}
	}

	assert(T != nil)
	return T
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

	// reset context for statements of inner blocks
	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)

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
		kind := check.rawExpr(nil, &x, s.X, nil, false)
		var msg string
		var code Code
		switch x.mode {
		default:
			if kind == statement {
				return
			}
			msg = "is not used"
			code = UnusedExpr
		case builtin:
			msg = "must be called"
			code = UncalledBuiltin
		case typexpr:
			msg = "is not an expression"
			code = NotAnExpr
		}
		check.errorf(&x, code, "%s %s", &x, msg)

	case *ast.SendStmt:
		var ch, val operand
		check.expr(nil, &ch, s.Chan)
		check.expr(nil, &val, s.Value)
		if ch.mode == invalid || val.mode == invalid {
			return
		}
		u := coreType(ch.typ)
		if u == nil {
			check.errorf(inNode(s, s.Arrow), InvalidSend, invalidOp+"cannot send to %s: no core type", &ch)
			return
		}
		uch, _ := u.(*Chan)
		if uch == nil {
			check.errorf(inNode(s, s.Arrow), InvalidSend, invalidOp+"cannot send to non-channel %s", &ch)
			return
		}
		if uch.dir == RecvOnly {
			check.errorf(inNode(s, s.Arrow), InvalidSend, invalidOp+"cannot send to receive-only channel %s", &ch)
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
			check.errorf(inNode(s, s.TokPos), InvalidSyntaxTree, "unknown inc/dec operation %s", s.Tok)
			return
		}

		var x operand
		check.expr(nil, &x, s.X)
		if x.mode == invalid {
			return
		}
		if !allNumeric(x.typ) {
			check.errorf(s.X, NonNumericIncDec, invalidOp+"%s%s (non-numeric type %s)", s.X, s.Tok, x.typ)
			return
		}

		Y := &ast.BasicLit{ValuePos: s.X.Pos(), Kind: token.INT, Value: "1"} // use x's position
		check.binary(&x, nil, s.X, Y, op, s.TokPos)
		if x.mode == invalid {
			return
		}
		check.assignVar(s.X, nil, &x, "assignment")

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			if len(s.Lhs) == 0 {
				check.error(s, InvalidSyntaxTree, "missing lhs in assignment")
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
				check.errorf(inNode(s, s.TokPos), MultiValAssignOp, "assignment operation %s requires single-valued expressions", s.Tok)
				return
			}
			op := assignOp(s.Tok)
			if op == token.ILLEGAL {
				check.errorf(atPos(s.TokPos), InvalidSyntaxTree, "unknown assignment operation %s", s.Tok)
				return
			}
			var x operand
			check.binary(&x, nil, s.Lhs[0], s.Rhs[0], op, s.TokPos)
			if x.mode == invalid {
				return
			}
			check.assignVar(s.Lhs[0], nil, &x, "assignment")
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
					err := check.newError(OutOfScopeResult)
					err.addf(s, "result parameter %s not in scope at return", obj.name)
					err.addf(alt, "inner declaration of %s", obj)
					err.report()
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
				check.error(s, MisplacedBreak, "break not in for, switch, or select statement")
			}
		case token.CONTINUE:
			if ctxt&continueOk == 0 {
				check.error(s, MisplacedContinue, "continue not in for statement")
			}
		case token.FALLTHROUGH:
			if ctxt&fallthroughOk == 0 {
				var msg string
				switch {
				case ctxt&finalSwitchCase != 0:
					msg = "cannot fallthrough final case in switch"
				case ctxt&inTypeSwitch != 0:
					msg = "cannot fallthrough in type switch"
				default:
					msg = "fallthrough statement out of place"
				}
				check.error(s, MisplacedFallthrough, msg)
			}
		default:
			check.errorf(s, InvalidSyntaxTree, "branch statement: %s", s.Tok)
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
		check.expr(nil, &x, s.Cond)
		if x.mode != invalid && !allBoolean(x.typ) {
			check.error(s.Cond, InvalidCond, "non-boolean condition in if statement")
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
			check.error(s.Else, InvalidSyntaxTree, "invalid else branch in if statement")
		}

	case *ast.SwitchStmt:
		inner |= breakOk
		check.openScope(s, "switch")
		defer check.closeScope()

		check.simpleStmt(s.Init)
		var x operand
		if s.Tag != nil {
			check.expr(nil, &x, s.Tag)
			// By checking assignment of x to an invisible temporary
			// (as a compiler would), we get all the relevant checks.
			check.assignment(&x, nil, "switch expression")
			if x.mode != invalid && !Comparable(x.typ) && !hasNil(x.typ) {
				check.errorf(&x, InvalidExprSwitch, "cannot switch on %s (%s is not comparable)", &x, x.typ)
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
				check.error(c, InvalidSyntaxTree, "incorrect expression switch case")
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
		inner |= breakOk | inTypeSwitch
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
				check.error(s, InvalidSyntaxTree, "incorrect form of type switch guard")
				return
			}

			lhs, _ = guard.Lhs[0].(*ast.Ident)
			if lhs == nil {
				check.error(s, InvalidSyntaxTree, "incorrect form of type switch guard")
				return
			}

			if lhs.Name == "_" {
				// _ := x.(type) is an invalid short variable declaration
				check.softErrorf(lhs, NoNewVar, "no new variable on left side of :=")
				lhs = nil // avoid declared and not used error below
			} else {
				check.recordDef(lhs, nil) // lhs variable is implicitly declared in each cause clause
			}

			rhs = guard.Rhs[0]

		default:
			check.error(s, InvalidSyntaxTree, "incorrect form of type switch guard")
			return
		}

		// rhs must be of the form: expr.(type) and expr must be an ordinary interface
		expr, _ := rhs.(*ast.TypeAssertExpr)
		if expr == nil || expr.Type != nil {
			check.error(s, InvalidSyntaxTree, "incorrect form of type switch guard")
			return
		}

		var sx *operand // switch expression against which cases are compared against; nil if invalid
		{
			var x operand
			check.expr(nil, &x, expr.X)
			if x.mode != invalid {
				if isTypeParam(x.typ) {
					check.errorf(&x, InvalidTypeSwitch, "cannot use type switch on type parameter value %s", &x)
				} else if IsInterface(x.typ) {
					sx = &x
				} else {
					check.errorf(&x, InvalidTypeSwitch, "%s is not an interface", &x)
				}
			}
		}

		check.multipleDefaults(s.Body.List)

		var lhsVars []*Var              // list of implicitly declared lhs variables
		seen := make(map[Type]ast.Expr) // map of seen types to positions
		for _, s := range s.Body.List {
			clause, _ := s.(*ast.CaseClause)
			if clause == nil {
				check.error(s, InvalidSyntaxTree, "incorrect type switch case")
				continue
			}
			// Check each type in this type switch case.
			T := check.caseTypes(sx, clause.List, seen)
			check.openScope(clause, "case")
			// If lhs exists, declare a corresponding variable in the case-local scope.
			if lhs != nil {
				obj := NewVar(lhs.Pos(), check.pkg, lhs.Name, T)
				// TODO(mdempsky): Just use clause.Colon? Why did I even suggest
				// "at the end of the TypeSwitchCase" in go.dev/issue/16794 instead?
				scopePos := clause.Pos() + token.Pos(len("default")) // for default clause (len(List) == 0)
				if n := len(clause.List); n > 0 {
					scopePos = clause.List[n-1].End()
				}
				check.declare(check.scope, nil, obj, scopePos)
				check.recordImplicit(clause, obj)
				// For the "declared and not used" error, all lhs variables act as
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
				check.softErrorf(lhs, UnusedVar, "%s declared and not used", lhs.Name)
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
				if x, _ := ast.Unparen(rhs).(*ast.UnaryExpr); x != nil && x.Op == token.ARROW {
					valid = true
				}
			}

			if !valid {
				check.error(clause.Comm, InvalidSelectCase, "select case must be send or receive (possibly with assignment)")
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
			check.expr(nil, &x, s.Cond)
			if x.mode != invalid && !allBoolean(x.typ) {
				check.error(s.Cond, InvalidCond, "non-boolean condition in for statement")
			}
		}
		check.simpleStmt(s.Post)
		// spec: "The init statement may be a short variable
		// declaration, but the post statement must not."
		if s, _ := s.Post.(*ast.AssignStmt); s != nil && s.Tok == token.DEFINE {
			check.softErrorf(s, InvalidPostDecl, "cannot declare in post statement")
			// Don't call useLHS here because we want to use the lhs in
			// this erroneous statement so that we don't get errors about
			// these lhs variables being declared and not used.
			check.use(s.Lhs...) // avoid follow-up errors
		}
		check.stmt(inner, s.Body)

	case *ast.RangeStmt:
		inner |= breakOk | continueOk
		check.rangeStmt(inner, s)

	default:
		check.error(s, InvalidSyntaxTree, "invalid statement")
	}
}

func (check *Checker) rangeStmt(inner stmtContext, s *ast.RangeStmt) {
	// Convert go/ast form to local variables.
	type Expr = ast.Expr
	type identType = ast.Ident
	identName := func(n *identType) string { return n.Name }
	sKey, sValue := s.Key, s.Value
	var sExtra ast.Expr = nil // (used only in types2 fork)
	isDef := s.Tok == token.DEFINE
	rangeVar := s.X
	noNewVarPos := inNode(s, s.TokPos)

	// Everything from here on is shared between cmd/compile/internal/types2 and go/types.

	// check expression to iterate over
	var x operand
	check.expr(nil, &x, rangeVar)

	// determine key/value types
	var key, val Type
	if x.mode != invalid {
		// Ranging over a type parameter is permitted if it has a core type.
		k, v, cause, ok := rangeKeyVal(x.typ, func(v goVersion) bool {
			return check.allowVersion(x.expr, v)
		})
		switch {
		case !ok && cause != "":
			check.softErrorf(&x, InvalidRangeExpr, "cannot range over %s: %s", &x, cause)
		case !ok:
			check.softErrorf(&x, InvalidRangeExpr, "cannot range over %s", &x)
		case k == nil && sKey != nil:
			check.softErrorf(sKey, InvalidIterVar, "range over %s permits no iteration variables", &x)
		case v == nil && sValue != nil:
			check.softErrorf(sValue, InvalidIterVar, "range over %s permits only one iteration variable", &x)
		case sExtra != nil:
			check.softErrorf(sExtra, InvalidIterVar, "range clause permits at most two iteration variables")
		}
		key, val = k, v
	}

	// Open the for-statement block scope now, after the range clause.
	// Iteration variables declared with := need to go in this scope (was go.dev/issue/51437).
	check.openScope(s, "range")
	defer check.closeScope()

	// check assignment to/declaration of iteration variables
	// (irregular assignment, cannot easily map to existing assignment checks)

	// lhs expressions and initialization value (rhs) types
	lhs := [2]Expr{sKey, sValue} // sKey, sValue may be nil
	rhs := [2]Type{key, val}     // key, val may be nil

	rangeOverInt := isInteger(x.typ)

	if isDef {
		// short variable declaration
		var vars []*Var
		for i, lhs := range lhs {
			if lhs == nil {
				continue
			}

			// determine lhs variable
			var obj *Var
			if ident, _ := lhs.(*identType); ident != nil {
				// declare new variable
				name := identName(ident)
				obj = NewVar(ident.Pos(), check.pkg, name, nil)
				check.recordDef(ident, obj)
				// _ variables don't count as new variables
				if name != "_" {
					vars = append(vars, obj)
				}
			} else {
				check.errorf(lhs, InvalidSyntaxTree, "cannot declare %s", lhs)
				obj = NewVar(lhs.Pos(), check.pkg, "_", nil) // dummy variable
			}
			assert(obj.typ == nil)

			// initialize lhs iteration variable, if any
			typ := rhs[i]
			if typ == nil || typ == Typ[Invalid] {
				// typ == Typ[Invalid] can happen if allowVersion fails.
				obj.typ = Typ[Invalid]
				obj.used = true // don't complain about unused variable
				continue
			}

			if rangeOverInt {
				assert(i == 0) // at most one iteration variable (rhs[1] == nil or Typ[Invalid] for rangeOverInt)
				check.initVar(obj, &x, "range clause")
			} else {
				var y operand
				y.mode = value
				y.expr = lhs // we don't have a better rhs expression to use here
				y.typ = typ
				check.initVar(obj, &y, "assignment") // error is on variable, use "assignment" not "range clause"
			}
			assert(obj.typ != nil)
		}

		// declare variables
		if len(vars) > 0 {
			scopePos := s.Body.Pos()
			for _, obj := range vars {
				check.declare(check.scope, nil /* recordDef already called */, obj, scopePos)
			}
		} else {
			check.error(noNewVarPos, NoNewVar, "no new variables on left side of :=")
		}
	} else if sKey != nil /* lhs[0] != nil */ {
		// ordinary assignment
		for i, lhs := range lhs {
			if lhs == nil {
				continue
			}

			// assign to lhs iteration variable, if any
			typ := rhs[i]
			if typ == nil || typ == Typ[Invalid] {
				continue
			}

			if rangeOverInt {
				assert(i == 0) // at most one iteration variable (rhs[1] == nil or Typ[Invalid] for rangeOverInt)
				check.assignVar(lhs, nil, &x, "range clause")
				// If the assignment succeeded, if x was untyped before, it now
				// has a type inferred via the assignment. It must be an integer.
				// (go.dev/issues/67027)
				if x.mode != invalid && !isInteger(x.typ) {
					check.softErrorf(lhs, InvalidRangeExpr, "cannot use iteration variable of type %s", x.typ)
				}
			} else {
				var y operand
				y.mode = value
				y.expr = lhs // we don't have a better rhs expression to use here
				y.typ = typ
				check.assignVar(lhs, nil, &y, "assignment") // error is on variable, use "assignment" not "range clause"
			}
		}
	} else if rangeOverInt {
		// If we don't have any iteration variables, we still need to
		// check that a (possibly untyped) integer range expression x
		// is valid.
		// We do this by checking the assignment _ = x. This ensures
		// that an untyped x can be converted to a value of its default
		// type (rune or int).
		check.assignment(&x, nil, "range clause")
	}

	check.stmt(inner, s.Body)
}

// rangeKeyVal returns the key and value type produced by a range clause
// over an expression of type typ.
// If allowVersion != nil, it is used to check the required language version.
// If the range clause is not permitted, rangeKeyVal returns ok = false.
// When ok = false, rangeKeyVal may also return a reason in cause.
func rangeKeyVal(typ Type, allowVersion func(goVersion) bool) (key, val Type, cause string, ok bool) {
	bad := func(cause string) (Type, Type, string, bool) {
		return Typ[Invalid], Typ[Invalid], cause, false
	}
	toSig := func(t Type) *Signature {
		sig, _ := coreType(t).(*Signature)
		return sig
	}

	orig := typ
	switch typ := arrayPtrDeref(coreType(typ)).(type) {
	case nil:
		return bad("no core type")
	case *Basic:
		if isString(typ) {
			return Typ[Int], universeRune, "", true // use 'rune' name
		}
		if isInteger(typ) {
			if allowVersion != nil && !allowVersion(go1_22) {
				return bad("requires go1.22 or later")
			}
			return orig, nil, "", true
		}
	case *Array:
		return Typ[Int], typ.elem, "", true
	case *Slice:
		return Typ[Int], typ.elem, "", true
	case *Map:
		return typ.key, typ.elem, "", true
	case *Chan:
		if typ.dir == SendOnly {
			return bad("receive from send-only channel")
		}
		return typ.elem, nil, "", true
	case *Signature:
		if !buildcfg.Experiment.RangeFunc && allowVersion != nil && !allowVersion(go1_23) {
			return bad("requires go1.23 or later")
		}
		assert(typ.Recv() == nil)
		switch {
		case typ.Params().Len() != 1:
			return bad("func must be func(yield func(...) bool): wrong argument count")
		case toSig(typ.Params().At(0).Type()) == nil:
			return bad("func must be func(yield func(...) bool): argument is not func")
		case typ.Results().Len() != 0:
			return bad("func must be func(yield func(...) bool): unexpected results")
		}
		cb := toSig(typ.Params().At(0).Type())
		assert(cb.Recv() == nil)
		switch {
		case cb.Params().Len() > 2:
			return bad("func must be func(yield func(...) bool): yield func has too many parameters")
		case cb.Results().Len() != 1 || !isBoolean(cb.Results().At(0).Type()):
			return bad("func must be func(yield func(...) bool): yield func does not return bool")
		}
		if cb.Params().Len() >= 1 {
			key = cb.Params().At(0).Type()
		}
		if cb.Params().Len() >= 2 {
			val = cb.Params().At(1).Type()
		}
		return key, val, "", true
	}
	return
}
