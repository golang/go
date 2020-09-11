// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of statements.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	"sort"
)

func (check *Checker) funcBody(decl *declInfo, name string, sig *Signature, body *syntax.BlockStmt, iota constant.Value) {
	if check.conf.Trace {
		check.trace(body.Pos(), "--- %s: %s", name, sig)
		defer func() {
			check.trace(endPos("body.End()"), "--- <end>")
		}()
	}

	// set function scope extent
	sig.scope.pos = body.Pos()
	sig.scope.end = endPos("body.End()")

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
		check.error(body, "missing return")
	}

	// TODO(gri) Should we make it an error to declare generic functions
	//           where the type parameters are not used?
	// 12/19/2018: Probably not - it can make sense to have an API with
	//           all functions uniformly sharing the same type parameters.

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
		return cmpPos(unused[i].pos, unused[j].pos) < 0
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

func (check *Checker) simpleStmt(s syntax.Stmt) {
	if s != nil {
		check.stmt(0, s)
	}
}

func trimTrailingEmptyStmts(list []syntax.Stmt) []syntax.Stmt {
	for i := len(list); i > 0; i-- {
		if _, ok := list[i-1].(*syntax.EmptyStmt); !ok {
			return list[:i]
		}
	}
	return nil
}

func (check *Checker) stmtList(ctxt stmtContext, list []syntax.Stmt) {
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

func (check *Checker) multipleSwitchDefaults(list []*syntax.CaseClause) {
	var first *syntax.CaseClause
	for _, c := range list {
		if c.Cases == nil {
			if first != nil {
				check.errorf(c, "multiple defaults (first at %s)", first.Pos())
				// TODO(gri) probably ok to bail out after first error (and simplify this code)
			} else {
				first = c
			}
		}
	}
}

func (check *Checker) multipleSelectDefaults(list []*syntax.CommClause) {
	var first *syntax.CommClause
	for _, c := range list {
		if c.Comm == nil {
			if first != nil {
				check.errorf(c, "multiple defaults (first at %s)", first.Pos())
				// TODO(gri) probably ok to bail out after first error (and simplify this code)
			} else {
				first = c
			}
		}
	}
}

func (check *Checker) openScope(node syntax.Node, comment string) {
	scope := NewScope(check.scope, node.Pos(), endPos("node.End()"), comment)
	check.recordScope(node, scope)
	check.scope = scope
}

func (check *Checker) closeScope() {
	check.scope = check.scope.Parent()
}

func (check *Checker) suspendedCall(keyword string, call *syntax.CallExpr) {
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
	check.errorf(&x, "%s %s %s", keyword, msg, &x)
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
		pos syntax.Pos
		typ Type
	}
)

func (check *Checker) caseValues(x *operand, values []syntax.Expr, seen valueMap) {
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
		check.comparison(&res, x, syntax.Eql)
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
				if check.identical(v.typ, vt.typ) {
					check.errorf(&v, "duplicate case %s in expression switch", &v)
					check.error(vt.pos, "\tprevious case") // secondary error, \t indented
					continue L
				}
			}
			seen[val] = append(seen[val], valueType{v.pos(), v.typ})
		}
	}
}

func (check *Checker) caseTypes(x *operand, xtyp *Interface, types []syntax.Expr, seen map[Type]syntax.Pos, strict bool) (T Type) {
L:
	for _, e := range types {
		T = check.typOrNil(e)
		if T == Typ[Invalid] {
			continue L
		}
		if T != nil {
			check.ordinaryType(e.Pos(), T)
		}
		// look for duplicate types
		// (quadratic algorithm, but type switches tend to be reasonably small)
		for t, pos := range seen {
			if T == nil && t == nil || T != nil && t != nil && check.identical(T, t) {
				// talk about "case" rather than "type" because of nil case
				Ts := "nil"
				if T != nil {
					Ts = T.String()
				}
				check.errorf(e, "duplicate case %s in type switch", Ts)
				check.error(pos, "\tprevious case") // secondary error, \t indented
				continue L
			}
		}
		seen[T] = e.Pos()
		if T != nil {
			check.typeAssertion(e.Pos(), x, xtyp, T, strict)
		}
	}
	return
}

// stmt typechecks statement s.
func (check *Checker) stmt(ctxt stmtContext, s syntax.Stmt) {
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
	case *syntax.EmptyStmt:
		// ignore

	case *syntax.DeclStmt:
		check.declStmt(s.DeclList)

	case *syntax.LabeledStmt:
		check.hasLabel = true
		check.stmt(ctxt, s.Stmt)

	case *syntax.ExprStmt:
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
		check.errorf(&x, "%s %s", &x, msg)

	case *syntax.SendStmt:
		var ch, x operand
		check.expr(&ch, s.Chan)
		check.expr(&x, s.Value)
		if ch.mode == invalid || x.mode == invalid {
			return
		}

		tch := ch.typ.Chan()
		if tch == nil {
			check.invalidOpf(s, "cannot send to non-chan type %s", ch.typ)
			return
		}

		if tch.dir == RecvOnly {
			check.invalidOpf(s, "cannot send to receive-only type %s", tch)
			return
		}

		check.assignment(&x, tch.elem, "send")

	case *syntax.AssignStmt:
		lhs := unpackExpr(s.Lhs)
		rhs := unpackExpr(s.Rhs)
		if s.Op == 0 || s.Op == syntax.Def {
			// regular assignment or short variable declaration
			if len(lhs) == 0 {
				check.invalidASTf(s, "missing lhs in assignment")
				return
			}
			if s.Op == syntax.Def {
				check.shortVarDecl(s.Pos(), lhs, rhs)
			} else {
				// regular assignment
				check.assignVars(lhs, rhs)
			}
		} else {
			// assignment operations
			if len(lhs) != 1 || len(rhs) != 1 {
				check.errorf(s, "assignment operation %s requires single-valued expressions", s.Op)
				return
			}
			var x operand
			check.binary(&x, nil, lhs[0], rhs[0], s.Op)
			if x.mode == invalid {
				return
			}
			check.assignVar(lhs[0], &x)
		}

	// case *syntax.GoStmt:
	// 	check.suspendedCall("go", s.Call)

	// case *syntax.DeferStmt:
	// 	check.suspendedCall("defer", s.Call)
	case *syntax.CallStmt:
		// TODO(gri) get rid of this conversion to string
		kind := "go"
		if s.Tok == syntax.Defer {
			kind = "defer"
		}
		check.suspendedCall(kind, s.Call)

	case *syntax.ReturnStmt:
		res := check.sig.results
		results := unpackExpr(s.Results)
		if res.Len() > 0 {
			// function returns results
			// (if one, say the first, result parameter is named, all of them are named)
			if len(results) == 0 && res.vars[0].name != "" {
				// spec: "Implementation restriction: A compiler may disallow an empty expression
				// list in a "return" statement if a different entity (constant, type, or variable)
				// with the same name as a result parameter is in scope at the place of the return."
				for _, obj := range res.vars {
					if alt := check.lookup(obj.name); alt != nil && alt != obj {
						check.errorf(s, "result parameter %s not in scope at return", obj.name)
						check.errorf(alt, "\tinner declaration of %s", obj)
						// ok to continue
					}
				}
			} else {
				// return has results or result parameters are unnamed
				check.initVars(res.vars, results, s.Pos())
			}
		} else if len(results) > 0 {
			check.error(results[0], "no result values expected")
			check.use(results...)
		}

	case *syntax.BranchStmt:
		if s.Label != nil {
			check.hasLabel = true
			return // checked in 2nd pass (check.labels)
		}
		switch s.Tok {
		case syntax.Break:
			if ctxt&breakOk == 0 {
				check.error(s, "break not in for, switch, or select statement")
			}
		case syntax.Continue:
			if ctxt&continueOk == 0 {
				check.error(s, "continue not in for statement")
			}
		case syntax.Fallthrough:
			if ctxt&fallthroughOk == 0 {
				msg := "fallthrough statement out of place"
				if ctxt&finalSwitchCase != 0 {
					msg = "cannot fallthrough final case in switch"
				}
				check.error(s, msg)
			}
		default:
			check.invalidASTf(s, "branch statement: %s", s.Tok)
		}

	case *syntax.BlockStmt:
		check.openScope(s, "block")
		defer check.closeScope()

		check.stmtList(inner, s.List)

	case *syntax.IfStmt:
		check.openScope(s, "if")
		defer check.closeScope()

		check.simpleStmt(s.Init)
		var x operand
		check.expr(&x, s.Cond)
		if x.mode != invalid && !isBoolean(x.typ) {
			check.error(s.Cond, "non-boolean condition in if statement")
		}
		check.stmt(inner, s.Then)
		// The parser produces a correct AST but if it was modified
		// elsewhere the else branch may be invalid. Check again.
		switch s.Else.(type) {
		case nil:
			// valid or error already reported
		case *syntax.IfStmt, *syntax.BlockStmt:
			check.stmt(inner, s.Else)
		default:
			check.error(s.Else, "invalid else branch in if statement")
		}

	case *syntax.SwitchStmt:
		inner |= breakOk
		check.openScope(s, "switch")
		defer check.closeScope()

		check.simpleStmt(s.Init)

		if g, _ := s.Tag.(*syntax.TypeSwitchGuard); g != nil {
			check.typeSwitchStmt(inner, s, g)
		} else {
			check.switchStmt(inner, s)
		}

	case *syntax.SelectStmt:
		inner |= breakOk

		check.multipleSelectDefaults(s.Body)

		for _, clause := range s.Body {
			if clause == nil {
				continue // error reported before
			}

			// clause.Comm must be a SendStmt, RecvStmt, or default case
			valid := false
			var rhs syntax.Expr // rhs of RecvStmt, or nil
			switch s := clause.Comm.(type) {
			case nil, *syntax.SendStmt:
				valid = true
			case *syntax.AssignStmt:
				if _, ok := s.Rhs.(*syntax.ListExpr); !ok {
					rhs = s.Rhs
				}
			case *syntax.ExprStmt:
				rhs = s.X
			}

			// if present, rhs must be a receive operation
			if rhs != nil {
				if x, _ := unparen(rhs).(*syntax.Operation); x != nil && x.Y == nil && x.Op == syntax.Recv {
					valid = true
				}
			}

			if !valid {
				check.error(clause.Comm, "select case must be send or receive (possibly with assignment)")
				continue
			}

			check.openScope(s, "case")
			if clause.Comm != nil {
				check.stmt(inner, clause.Comm)
			}
			check.stmtList(inner, clause.Body)
			check.closeScope()
		}

	case *syntax.ForStmt:
		inner |= breakOk | continueOk
		check.openScope(s, "for")
		defer check.closeScope()

		if rclause, _ := s.Init.(*syntax.RangeClause); rclause != nil {
			check.rangeStmt(inner, s, rclause)
			break
		}

		check.simpleStmt(s.Init)
		if s.Cond != nil {
			var x operand
			check.expr(&x, s.Cond)
			if x.mode != invalid && !isBoolean(x.typ) {
				check.error(s.Cond, "non-boolean condition in for statement")
			}
		}
		check.simpleStmt(s.Post)
		// spec: "The init statement may be a short variable
		// declaration, but the post statement must not."
		if s, _ := s.Post.(*syntax.AssignStmt); s != nil && s.Op == syntax.Def {
			check.softErrorf(s, "cannot declare in post statement")
			// Don't call useLHS here because we want to use the lhs in
			// this erroneous statement so that we don't get errors about
			// these lhs variables being declared but not used.
			check.use(s.Lhs) // avoid follow-up errors
		}
		check.stmt(inner, s.Body)

	default:
		check.error(s, "invalid statement")
	}
}

func newName(pos syntax.Pos, value string) *syntax.Name {
	n := new(syntax.Name)
	// TODO(gri) why does this not work?
	//n.pos = pos
	n.Value = value
	return n
}

func (check *Checker) switchStmt(inner stmtContext, s *syntax.SwitchStmt) {
	// init statement already handled

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
		// TODO(gri) should have a better position here
		pos := s.Rbrace
		if len(s.Body) > 0 {
			pos = s.Body[0].Pos()
		}
		x.expr = newName(pos, "true")
	}

	check.multipleSwitchDefaults(s.Body)

	seen := make(valueMap) // map of seen case values to positions and types
	for i, clause := range s.Body {
		if clause == nil {
			check.invalidASTf(clause, "incorrect expression switch case")
			continue
		}
		check.caseValues(&x, unpackExpr(clause.Cases), seen)
		check.openScope(clause, "case")
		inner := inner
		if i+1 < len(s.Body) {
			inner |= fallthroughOk
		} else {
			inner |= finalSwitchCase
		}
		check.stmtList(inner, clause.Body)
		check.closeScope()
	}
}

func (check *Checker) typeSwitchStmt(inner stmtContext, s *syntax.SwitchStmt, guard *syntax.TypeSwitchGuard) {
	// init statement already handled

	// A type switch guard must be of the form:
	//
	//     TypeSwitchGuard = [ identifier ":=" ] PrimaryExpr "." "(" "type" ")" .
	//                          \__lhs__/        \___rhs___/

	// check lhs, if any
	lhs := guard.Lhs
	if lhs != nil {
		if lhs.Value == "_" {
			// _ := x.(type) is an invalid short variable declaration
			check.softErrorf(lhs, "no new variable on left side of :=")
			lhs = nil // avoid declared but not used error below
		} else {
			check.recordDef(lhs, nil) // lhs variable is implicitly declared in each cause clause
		}
	}

	// check rhs
	var x operand
	check.expr(&x, guard.X)
	if x.mode == invalid {
		return
	}
	var xtyp *Interface
	var strict bool
	switch t := x.typ.Under().(type) {
	case *Interface:
		xtyp = t
	case *TypeParam:
		xtyp = t.Bound()
		strict = true
	default:
		check.errorf(&x, "%s is not an interface or generic type", &x)
		return
	}

	check.multipleSwitchDefaults(s.Body)

	var lhsVars []*Var                // list of implicitly declared lhs variables
	seen := make(map[Type]syntax.Pos) // map of seen types to positions
	for _, clause := range s.Body {
		if clause == nil {
			check.invalidASTf(s, "incorrect type switch case")
			continue
		}
		// Check each type in this type switch case.
		cases := unpackExpr(clause.Cases)
		T := check.caseTypes(&x, xtyp, cases, seen, strict)
		check.openScope(clause, "case")
		// If lhs exists, declare a corresponding variable in the case-local scope.
		if lhs != nil {
			// spec: "The TypeSwitchGuard may include a short variable declaration.
			// When that form is used, the variable is declared at the beginning of
			// the implicit block in each clause. In clauses with a case listing
			// exactly one type, the variable has that type; otherwise, the variable
			// has the type of the expression in the TypeSwitchGuard."
			if len(cases) != 1 || T == nil {
				T = x.typ
			}
			obj := NewVar(lhs.Pos(), check.pkg, lhs.Value, T)
			scopePos := clause.Pos() /* + syntax.Pos(len("default")) */ // for default clause (len(List) == 0)
			if n := len(cases); n > 0 {
				scopePos = cases[n-1].Pos() // TODO(gri) this should really be the end pos
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
			check.softErrorf(lhs, "%s declared but not used", lhs.Value)
		}
	}
}

func (check *Checker) rangeStmt(inner stmtContext, s *syntax.ForStmt, rclause *syntax.RangeClause) {
	// scope already opened

	// check expression to iterate over
	var x operand
	check.expr(&x, rclause.X)

	// determine lhs, if any
	sKey := rclause.Lhs // possibly nil
	var sValue syntax.Expr
	if p, _ := sKey.(*syntax.ListExpr); p != nil {
		if len(p.ElemList) != 2 {
			check.invalidASTf(s, "invalid lhs in range clause")
			return
		}
		sKey = p.ElemList[0]
		sValue = p.ElemList[1]
	}

	// determine key/value types
	var key, val Type
	if x.mode != invalid {
		typ := optype(x.typ.Under())
		if _, ok := typ.(*Chan); ok && sValue != nil {
			// TODO(gri) this also needs to happen for channels in generic variables
			check.softErrorf(sValue, "range over %s permits only one iteration variable", &x)
			// ok to continue
		}
		var msg string
		key, val, msg = rangeKeyVal(typ, isVarName(sKey), isVarName(sValue))
		if key == nil || msg != "" {
			if msg != "" {
				msg = ": " + msg
			}
			check.softErrorf(&x, "cannot range over %s%s", &x, msg)
			// ok to continue
		}
	}

	// check assignment to/declaration of iteration variables
	// (irregular assignment, cannot easily map to existing assignment checks)

	// lhs expressions and initialization value (rhs) types
	lhs := [2]syntax.Expr{sKey, sValue}
	rhs := [2]Type{key, val} // key, val may be nil

	if rclause.Def {
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
			if ident, _ := lhs.(*syntax.Name); ident != nil {
				// declare new variable
				name := ident.Value
				obj = NewVar(ident.Pos(), check.pkg, name, nil)
				check.recordDef(ident, obj)
				// _ variables don't count as new variables
				if name != "_" {
					vars = append(vars, obj)
				}
			} else {
				check.errorf(lhs, "cannot declare %s", lhs)
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
				// spec: "The scope of a constant or variable identifier declared inside
				// a function begins at the end of the ConstSpec or VarSpec (ShortVarDecl
				// for short variable declarations) and ends at the end of the innermost
				// containing block."
				check.declare(check.scope, nil /* recordDef already called */, obj, scopePos)
			}
		} else {
			check.error(s, "no new variables on left side of :=")
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
}

// isVarName reports whether x is a non-nil, non-blank (_) expression.
func isVarName(x syntax.Expr) bool {
	if x == nil {
		return false
	}
	ident, _ := unparen(x).(*syntax.Name)
	return ident == nil || ident.Value != "_"
}

// rangeKeyVal returns the key and value type produced by a range clause
// over an expression of type typ, and possibly an error message. If the
// range clause is not permitted the returned key is nil or msg is not
// empty (in that case we still may have a non-nil key type which can be
// used to reduce the chance for follow-on errors).
// The wantKey, wantVal, and hasVal flags indicate which of the iteration
// variables are used or present; this matters if we range over a generic
// type where not all keys or values are of the same type.
func rangeKeyVal(typ Type, wantKey, wantVal bool) (Type, Type, string) {
	switch typ := typ.(type) {
	case *Basic:
		if isString(typ) {
			return Typ[Int], universeRune, "" // use 'rune' name
		}
	case *Array:
		return Typ[Int], typ.elem, ""
	case *Slice:
		return Typ[Int], typ.elem, ""
	case *Pointer:
		if typ := typ.base.Array(); typ != nil {
			return Typ[Int], typ.elem, ""
		}
	case *Map:
		return typ.key, typ.elem, ""
	case *Chan:
		var msg string
		if typ.dir == SendOnly {
			msg = "send-only channel"
		}
		return typ.elem, Typ[Invalid], msg
	case *Sum:
		first := true
		var key, val Type
		var msg string
		typ.is(func(t Type) bool {
			k, v, m := rangeKeyVal(t.Under(), wantKey, wantVal)
			if k == nil || m != "" {
				key, val, msg = k, v, m
				return false
			}
			if first {
				key, val, msg = k, v, m
				first = false
				return true
			}
			if wantKey && !Identical(key, k) {
				key, val, msg = nil, nil, "all possible values must have the same key type"
				return false
			}
			if wantVal && !Identical(val, v) {
				key, val, msg = nil, nil, "all possible values must have the same element type"
				return false
			}
			return true
		})
		return key, val, msg
	}
	return nil, nil, ""
}
