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
// per-line.
func (check *Checker) logosComptimeCall(x *operand, call *syntax.CallExpr) bool {
	sel, fname, ok := check.logosComptimeQualifier(call)
	if !ok {
		return false
	}
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
	case fname == "Fields", fname == "IsStruct", fname == "TypeOf":
		check.errorf(syntax.StartPos(call), InvalidPkgUse, "[comptime::runtime_use] comptime.%s is compile-time only", fname)
	case fname == "Error":
		check.logosComptimeError(call)
	}
	return false
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

// logosPredOnly reports whether every non-constant leaf of the condition
// e is one of the deferred predicate calls. The connectives `!`, `&&`,
// `||`, `==`, `!=` and parentheses are looked through; a constant leaf
// is a typed constant recorded in Info.Types or an untyped constant
// still waiting in the untyped table.
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

// logosConstLeaf reports whether e was checked to a constant.
func (check *Checker) logosConstLeaf(e syntax.Expr) bool {
	if tv, ok := check.Info.Types[e]; ok && tv.Value != nil {
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
func (check *Checker) logosComptimeError(call *syntax.CallExpr) {
	generic := check.sig != nil && check.sig.TypeParams().Len() > 0
	dead := check.Info.LogosConstIfDead[check.logosCurStmt]
	var arg syntax.Expr
	if len(call.ArgList) == 1 {
		arg = call.ArgList[0]
	}
	check.later(func() {
		if arg == nil || generic || dead {
			return
		}
		tv, ok := check.Info.Types[arg]
		if !ok || !isValid(tv.Type) || !isString(tv.Type) {
			return
		}
		if tv.Value == nil || tv.Value.Kind() != constant.String {
			check.error(arg.Pos(), InvalidConstInit, "[comptime::error_not_constant] comptime.Error needs a constant message")
			return
		}
		check.errorf(arg.Pos(), InvalidConstInit, "[comptime::error] %s", constant.StringVal(tv.Value))
	}).describef(call, "comptime.Error %s", call.Pos())
}
