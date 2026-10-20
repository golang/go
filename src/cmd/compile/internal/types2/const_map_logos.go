// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// This file gives the M5-S4 const map view its standard-library rule (SPEC
// T-201, T-Sec9): a const map passes to a standard-library parameter only
// when that parameter is read-only (fact ParamMutated=0).
//
// Facts travel in __.LOGOS, and no standard-library package carries that
// member: the compiler writes facts only for loGo/s packages (noder
// logosGoCallerHook never lowers Go packages), so the fact reader has
// nothing to read for "maps". This table is the stand-in until std facts
// exist. Each entry lists the parameters the function mutates as a bit
// mask (bit i set means parameter i is mutated, the ParamMutated encoding).
// A function missing from the table fails closed, as a const view
// argument always did.

// logosStdMutatedParams maps "importpath.Func" to its ParamMutated mask.
var logosStdMutatedParams = map[string]uint64{
	"maps.All":        0, // m: read
	"maps.Clone":      0, // m: read
	"maps.Collect":    0, // no map parameter
	"maps.Copy":       1, // dst: written; src: read
	"maps.DeleteFunc": 1, // m: deleted from
	"maps.Equal":      0, // m1, m2: read
	"maps.EqualFunc":  0, // m1, m2: read
	"maps.Insert":     1, // m: written
	"maps.Keys":       0, // m: read
	"maps.Values":     0, // m: read
}

// logosImportedCallMask resolves the callee of call when it names a function
// of an imported package (pkg.F). It returns the callee's ParamMutated mask
// from the table, whether the callee is imported at all, and whether the table
// knows it.
func logosImportedCallMask(check *Checker, call *syntax.CallExpr) (mask uint64, imported, known bool) {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok {
		return 0, false, false
	}
	name, ok := sel.X.(*syntax.Name)
	if !ok {
		return 0, false, false
	}
	pkg, ok := check.lookup(name.Value).(*PkgName)
	if !ok {
		return 0, false, false
	}
	mask, known = logosStdMutatedParams[pkg.Imported().Path()+"."+sel.Sel.Value]
	return mask, true, known
}

// logosConstMapKeyOK reports whether a const map key of type t has a base
// hash the lowering can compute (SPEC T-Sec9): strings, numbers, and bools,
// and structs and arrays of them. Interfaces, pointers, channels, and complex
// numbers have none.
func logosConstMapKeyOK(t Type) bool {
	switch u := t.Underlying().(type) {
	case *Basic:
		return u.info&(IsString|IsBoolean|IsInteger|IsFloat) != 0
	case *Struct:
		for i := range u.NumFields() {
			if f := u.Field(i); f.Name() != "_" && !logosConstMapKeyOK(f.Type()) {
				return false
			}
		}
		return true
	case *Array:
		return logosConstMapKeyOK(u.Elem())
	}
	return false
}

// logosConstMapReadOnly reports whether argument i of call may take a const
// map's materialized copy. An imported callee decides by its table mask. A
// function of this package takes it only when the callee is a const func:
// const funcs run at compile time, and the evaluator copies each argument at
// binding (ctfe copyArg, evalcall.go): a write through a parameter stays in
// the callee and never reaches the caller's recorded const value. The purity
// check covers package variables only, so it is not the guard here. Every
// other .lgo parameter fails closed, as T-201 requires for non-const
// parameters of .lgo functions.
func (check *Checker) logosConstMapReadOnly(call *syntax.CallExpr, imported, known bool, mask uint64, i int) bool {
	if imported {
		return known && i < 64 && mask&(1<<uint(i)) == 0
	}
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	fn, ok := check.lookup(name.Value).(*Func)
	if !ok || fn.Pkg() != check.pkg {
		return false
	}
	bodies, consts := check.logosConstFuncBodies()
	fd, ok := bodies[name.Value]
	return ok && consts[fd]
}

// logosConstMapArgs decides the const map arguments of a call before type
// inference (SPEC T-201, T-Sec9). A const map reaches a const parameter as its
// view. Every other parameter takes the materialized map (the element type),
// which is what inference must see for a ~map[K]V constraint, unless the
// callee mutates it: a standard-library function the table marks mutating
// (maps.DeleteFunc), an imported function the table does not know, and a
// non-const function of this package all fail with const_view_mutation. A
// const func of this package takes the materialized copy.
func (check *Checker) logosConstMapArgs(call *syntax.CallExpr, sigParams *Tuple, args []*operand) {
	if !check.conf.Logos || sigParams == nil {
		return
	}
	mask, imported, known := logosImportedCallMask(check, call)
	for i, a := range args {
		if !a.isValid() || i >= sigParams.Len() {
			continue
		}
		c := logosAsConstView(a.typ())
		if c == nil || !c.ConstMap {
			continue
		}
		if logosAsConstView(sigParams.At(i).Type()) != nil {
			continue
		}
		readOnly := check.logosConstMapReadOnly(call, imported, known, mask, i)
		if !readOnly {
			check.errorf(a, IncompatibleAssign, "%s cannot pass argument through const view %s in argument to %s", logosConstMutation, ExprString(a.expr), ExprString(call.Fun))
			a.invalidate()
			continue
		}
		a.typ_ = c.Elem
		check.updateExprType(a.expr, c.Elem, false)
	}
}
