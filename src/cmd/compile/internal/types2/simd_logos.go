// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosSimdTooLargeRule = "[correctness::simd_array_too_large]"
const logosSimdShapeRule = "[correctness::simd_array_shape]"

const logosSimdMaxBytes = 64

var logosSimdMethods = map[string]bool{
	"sum": true,
	"min": true,
	"max": true,
	"dot": true,
}

func LogosSimdMethod(name string) bool {
	return logosSimdMethods[name]
}

type logosSimdShape struct {
	elem *Basic
	typ  Type
	len  int64
}

func logosSimdShapeOf(typ Type) (logosSimdShape, bool) {
	at, ok := typ.Underlying().(*Array)
	if !ok {
		return logosSimdShape{}, false
	}
	b, ok := at.Elem().Underlying().(*Basic)
	if !ok {
		return logosSimdShape{}, false
	}
	if b.Info()&(IsInteger|IsFloat) == 0 {
		return logosSimdShape{}, false
	}
	return logosSimdShape{elem: b, typ: at.Elem(), len: at.Len()}, true
}

func (check *Checker) logosSimdSize(sh logosSimdShape) int64 {
	var width int64
	switch sh.elem.Kind() {
	case Int, Uint, Uintptr:
		width = 8
		if check.conf.Sizes != nil {
			if w := check.conf.Sizes.Sizeof(Typ[Int]); w == 4 || w == 8 {
				width = w
			}
		}
	case Int8, Uint8:
		width = 1
	case Int16, Uint16:
		width = 2
	case Int32, Uint32, Float32:
		width = 4
	case Int64, Uint64, Float64:
		width = 8
	default:
		return logosSimdMaxBytes + 1
	}
	return sh.len * width
}

func logosSimdOpPair(op syntax.Operator, k BasicKind) bool {
	info := Typ[k].Info()
	switch op {
	case syntax.Add, syntax.Sub, syntax.Mul, syntax.Div, syntax.And, syntax.Or, syntax.Xor:
		return info&(IsInteger|IsFloat) != 0
	}
	return false
}

func logosSimdPow2(n int64) bool {
	return n >= 1 && n&(n-1) == 0
}

func (check *Checker) logosSimdBinary(x *operand, e syntax.Expr, op syntax.Operator) bool {
	if e == nil || !check.conf.Logos {
		return false
	}
	sh, ok := logosSimdShapeOf(x.typ())
	if !ok {
		return false
	}
	if !logosSimdOpPair(op, sh.elem.Kind()) {
		return false
	}
	if size := check.logosSimdSize(sh); size > logosSimdMaxBytes {
		check.errorf(x, MismatchedTypes, "%s %s is %d bytes; element-wise operators need at most 64", logosSimdTooLargeRule, x.typ(), size)
		x.invalidate()
		return true
	}
	if !logosSimdPow2(sh.len) {
		check.errorf(x, MismatchedTypes, "%s %s has length %d; element-wise operators need a power-of-two length", logosSimdShapeRule, x.typ(), sh.len)
		x.invalidate()
		return true
	}
	x.mode_ = value
	return true
}

func (check *Checker) logosSimdCall(x *operand, call *syntax.CallExpr) bool {
	if !check.conf.Logos {
		return false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	method := sel.Sel.Value
	if !logosSimdMethods[method] {
		return false
	}
	var rx operand
	check.exprOrType(&rx, sel.X, false)
	if !rx.isValid() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
		return true
	}
	if rx.mode() == typexpr {
		return false
	}
	if obj, _, _ := lookupFieldOrMethod(rx.typ(), rx.mode() == variable, check.pkg, method, false); obj != nil {
		return false
	}
	rt, _, _ := logosReadonlyBase(rx.typ())
	sh, ok := logosSimdShapeOf(rt)
	if !ok {
		return false
	}
	fail := func() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
	}
	args := call.ArgList
	if hasDots(call) {
		check.errorf(sel.Sel, InvalidDotDotDot, "invalid use of ... in simd reduction %s", method)
		fail()
		return true
	}
	want := 0
	if method == "dot" {
		want = 1
	}
	if len(args) != want {
		check.errorf(sel.Sel, WrongArgCount, "wrong number of arguments in simd reduction %s: have %d, want %d", method, len(args), want)
		fail()
		return true
	}
	if size := check.logosSimdSize(sh); size > logosSimdMaxBytes {
		check.errorf(sel.Sel, MismatchedTypes, "%s %s is %d bytes; reductions need at most 64", logosSimdTooLargeRule, rx.typ(), size)
		fail()
		return true
	}
	if !logosSimdPow2(sh.len) {
		check.errorf(sel.Sel, MismatchedTypes, "%s %s has length %d; reductions need a power-of-two length", logosSimdShapeRule, rx.typ(), sh.len)
		fail()
		return true
	}
	if method == "dot" {
		var av operand
		check.expr(nil, &av, args[0])
		if !av.isValid() {
			fail()
			return true
		}
		if at, _, _ := logosReadonlyBase(av.typ()); !Identical(at, rt) {
			check.errorf(&av, IncompatibleAssign, "cannot use %s as %s value in argument to %s", &av, rt, call.Fun)
			fail()
			return true
		}
	}
	x.mode_ = value
	x.typ_ = sh.typ
	x.expr = call
	return true
}
