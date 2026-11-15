// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

// LoGo/s M7-S2/M7-S3 comptime call positions (SPEC Sec 2.5, T-205).
// callExpr calls in through one hook line. Values of the virtual
// package exist only at compile time: Fields serves const-for ranges
// (checked by the const-for checker, never here), IsStruct serves
// `const if` conditions (folded here), and every other call position
// reports [comptime::runtime_use], except a live comptime.Error, which
// fails compilation with its message.

// logosPredState is the `const if` condition context: depth counts the
// condition expressions being checked, and calls holds the predicate
// calls whose answer waits for an instantiation.
type logosPredState struct {
	depth int
	calls map[*syntax.CallExpr]bool
}

// logosComptimeCall vets one call for compile-time-only placement. It
// reports whether it answered the call itself: an IsStruct predicate in
// a `const if` condition becomes a constant (or a deferred value inside
// a comptime generic). Every other call keeps the normal check, so the
// arguments and results are still typed; Error messages go out through a
// deferred check that reads the checked constant, keeping error matching
// per-line. ViewAs keeps the normal check under its own M7-S5
// validation, and unknown names keep the upstream undefined-selector
// diagnostic.
func (check *Checker) logosComptimeCall(x *operand, call *syntax.CallExpr) bool {
	sel, fname, ok := check.logosComptimeQualifier(call)
	if !ok {
		return false
	}
	if check.logosComptimeCallee == nil {
		check.logosComptimeCallee = make(map[*syntax.SelectorExpr]bool)
	}
	check.logosComptimeCallee[sel] = true
	check.recordUse(sel.Sel, check.logosComptimeObj(fname))
	if name, ok := sel.X.(*syntax.Name); ok {
		if _, obj := check.lookupScope(name.Value); obj != nil {
			check.recordUse(name, obj)
			if pn, ok := obj.(*PkgName); ok {
				check.usedPkgNames[pn] = true
			}
		}
	}
	switch {
	case fname == "IsStruct" && check.logosPred.depth > 0 && check.logosIsStruct(x, call):
		return true
	case isLogosLayoutName(fname) && check.logosLayoutCall(x, call, fname):
		return true
	case fname == "TypeOf" && check.logosTypeOfCall(x, call):
		return true
	case fname == "Fields", fname == "IsStruct", fname == "TypeOf",
		fname == "SizeOf", fname == "AlignOf", fname == "HasPointers":
		check.errorf(syntax.StartPos(call), InvalidPkgUse, "[comptime::runtime_use] comptime.%s is compile-time only", fname)
	// Invariant (deferred error): Error messages go out through a
	// deferred check reading the checked constant, keeping per-line matching.
	case fname == "Error":
		check.logosComptimeError(call)
	}
	return false
}

func (check *Checker) logosComptimeValueUse(e *syntax.SelectorExpr, obj Object) {
	fname, ok := logosComptimeFunc(obj)
	if !ok || fname == "ViewAs" || check.logosComptimeCallee[e] {
		return
	}
	check.errorf(syntax.StartPos(e), InvalidPkgUse, "[comptime::runtime_use] comptime.%s is compile-time only", fname)
}

// logosIsStruct answers comptime.IsStruct[X]() in a `const if`
// condition. A concrete X gives an untyped boolean constant, true when
// its underlying type is a struct. A type parameter X, and the field
// type of a const for over one, give a boolean value that is not a
// constant: the answer is per instantiation, so the condition is
// deferred to the copies. It reports false for any other call shape,
// which the normal check then rejects.
func (check *Checker) logosIsStruct(x *operand, call *syntax.CallExpr) bool {
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	targs := syntax.UnpackListExpr(idx.Index)
	if len(targs) != 1 {
		return false
	}
	targ := check.varType(targs[0])
	x.expr = call
	switch {
	case !isValid(targ):
		x.invalidate()
	case isTypeParam(targ) || check.logosGenericFieldType(targs[0]):
		x.mode_ = value
		x.typ_ = Typ[Bool]
		if check.logosPred.calls == nil {
			check.logosPred.calls = make(map[*syntax.CallExpr]bool)
		}
		check.logosPred.calls[call] = true
	default:
		_, isStruct := targ.Underlying().(*Struct)
		x.mode_ = constant_
		x.typ_ = Typ[UntypedBool]
		x.val = constant.MakeBool(isStruct)
	}
	return true
}
func (check *Checker) logosTypeOfCall(x *operand, call *syntax.CallExpr) bool {
	if !check.logosComptimeInGeneric() {
		return false
	}
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	targs := syntax.UnpackListExpr(idx.Index)
	if len(targs) != 1 {
		return false
	}
	targ := check.varType(targs[0])
	x.expr = call
	if !isValid(targ) {
		x.invalidate()
		return true
	}
	obj := check.logosComptimeObj("Type")
	tn, ok := obj.(*TypeName)
	if !ok || tn.Type() == nil {
		return false
	}
	x.mode_ = value
	x.typ_ = tn.Type()
	return true
}

// logosPredOnly reports whether every non-constant leaf of the condition
// e is one of the deferred predicate calls. The connectives `!`, `&&`,
// `||`, `==`, `!=` and parentheses are looked through; a constant leaf
// is a typed constant recorded in Info.Types or an untyped constant
// still waiting in the untyped table. Short-circuit constants do not
// fold here: `true || p` stays deferred and the copies fold it per
// instantiation, so no constant branch is ever proven from a deferred
// predicate.
func (check *Checker) logosPredOnly(e syntax.Expr, calls map[*syntax.CallExpr]bool) bool {
	switch e := e.(type) {
	case *syntax.ParenExpr:
		return check.logosPredOnly(e.X, calls)
	case *syntax.Operation:
		switch e.Op {
		case syntax.Not, syntax.AndAnd, syntax.OrOr, syntax.Eql, syntax.Neq:
			return check.logosPredOnly(e.X, calls) && (e.Y == nil || check.logosPredOnly(e.Y, calls))
		}
	case *syntax.CallExpr:
		if calls[e] {
			return true
		}
	}
	return check.logosConstLeaf(e)
}

// logosChecked returns the type and constant value recorded for e: the
// Info.Types entry when that map exists, else the syntax-node store the
// compiler fills (Info.TypeOf reads the same two places). ok is false
// when nothing is recorded.
func (check *Checker) logosChecked(e syntax.Expr) (Type, constant.Value, bool) {
	if m := check.Info.Types; m != nil {
		tv, ok := m[e]
		return tv.Type, tv.Value, ok
	}
	if check.Info.StoreTypesInSyntax {
		if tv := e.GetTypeInfo(); tv.Type != nil {
			return tv.Type, tv.Value, true
		}
	}
	return nil, nil, false
}

// logosConstLeaf reports whether e was checked to a constant.
func (check *Checker) logosConstLeaf(e syntax.Expr) bool {
	if _, val, ok := check.logosChecked(e); ok && val != nil {
		return true
	}
	info, ok := check.untyped[e]
	return ok && info.val != nil
}

// logosGenericFieldType reports whether e is the f.Type of a const for
// field variable whose loop is not being checked per concrete field: its
// type is then only approximated, so a predicate on it cannot answer.
func (check *Checker) logosGenericFieldType(e syntax.Expr) bool {
	sel, ok := e.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Type" {
		return false
	}
	name, ok := sel.X.(*syntax.Name)
	if !ok {
		return false
	}
	_, obj := check.lookupScope(name.Value)
	v, ok := obj.(*Var)
	if !ok || check.logosFieldVars[v] == nil {
		return false
	}
	b := check.logosFieldBind
	return b == nil || b.info != check.logosFieldVars[v]
}

// logosComptimeError defers a comptime.Error call: the message is read
// from the checked argument. A call inside a generic function is left
// for the per-instantiation expansion (M7-S3 lowering reports it); a
// call in a dead const-if branch (M7-S1) is silent; any other live call
// with a constant message fails with that message, and a
// non-constant message fails as not constant.
// logosComptimeInGeneric reports whether the call sits inside a generic
// function declaration: only declarations with a function type-parameter
// list get per-instantiation copies, so only those defer and the copies
// answer. A method of a generic type with no function type parameters
// is never copied, so a call there answers eagerly at the first check
// and the diagnostic cannot vanish. It reads the declaration stack, not
// a global, so lowered rechecks (fresh checkers sharing Info) decide
// from their own concrete copies.
func (check *Checker) logosComptimeInGeneric() bool {
	// Invariant (innermost decl): the current signature's own function
	// type parameters defer the call to per-instantiation copies.
	if check.sig != nil && check.sig.TypeParams().Len() > 0 {
		return true
	}
	for _, sig := range check.logosFuncSigs {
		// Invariant (enclosing decl): an outer generic function
		// declaration defers the call too; its copy answers per
		// instantiation.
		if sig != nil && sig.TypeParams().Len() > 0 {
			return true
		}
	}
	// Invariant (concrete): no enclosing function type parameters means
	// the first check answers directly.
	return false
}

func (check *Checker) logosComptimeError(call *syntax.CallExpr) {
	// Invariant (deferral): generic instantiations answer per copy.
	generic := check.logosComptimeInGeneric()
	// Invariant (dead silence): M7-S1 dead-branch membership is the
	// first-check Info proof, shared with lowered rechecks.
	dead := check.Info.LogosConstIfDead[check.logosCurStmt]
	var arg syntax.Expr
	if len(call.ArgList) == 1 {
		arg = call.ArgList[0]
	}
	check.later(func() {
		// Invariant (no double report): malformed, generic-deferred,
		// or dead-branch calls stay silent here.
		if arg == nil || generic || dead {
			return
		}
		typ, val, ok := check.logosChecked(arg)
		// Invariant (typed-mirror): untyped or non-string arguments
		// keep the normal check's diagnostic, never a second one.
		if !ok || !isValid(typ) || !isString(typ) {
			return
		}
		// Invariant (constant obligation): a live non-constant message
		// fails as not constant without weakening the text.
		if val == nil || val.Kind() != constant.String {
			check.error(arg.Pos(), InvalidConstInit, "[comptime::error_not_constant] comptime.Error needs a constant message")
			return
		}
		// Invariant (live error): a live constant message fails with its text.
		check.errorf(arg.Pos(), InvalidConstInit, "[comptime::error] %s", constant.StringVal(val))
	}).describef(call, "comptime.Error %s", call.Pos())
}
