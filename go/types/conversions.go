// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of conversions.

package types

import (
	"go/ast"

	"code.google.com/p/go.tools/go/exact"
)

// conversion typechecks the type conversion conv to type typ.
// The result of the conversion is returned via x.
// If the conversion has type errors, the returned
// x is marked as invalid (x.mode == invalid).
//
func (check *checker) conversion(x *operand, conv *ast.CallExpr, typ Type) {
	var final Type // declare before gotos

	// all conversions have one argument
	if len(conv.Args) != 1 {
		check.invalidOp(conv.Pos(), "%s conversion requires exactly one argument", conv)
		goto Error
	}

	// evaluate argument
	check.expr(x, conv.Args[0])
	if x.mode == invalid {
		goto Error
	}

	if x.mode == constant && isConstType(typ) {
		// constant conversion
		typ := typ.Underlying().(*Basic)
		// For now just implement string(x) where x is an integer,
		// as a temporary work-around for issue 4982, which is a
		// common issue.
		if typ.kind == String {
			switch {
			case x.isInteger():
				codepoint := int64(-1)
				if i, ok := exact.Int64Val(x.val); ok {
					codepoint = i
				}
				// If codepoint < 0 the absolute value is too large (or unknown) for
				// conversion. This is the same as converting any other out-of-range
				// value - let string(codepoint) do the work.
				x.val = exact.MakeString(string(codepoint))
			case isString(x.typ):
				// nothing to do
			default:
				goto ErrorMsg
			}
		}
		// TODO(gri) verify the remaining conversions.
	} else {
		// non-constant conversion
		if !x.isConvertible(check.ctxt, typ) {
			goto ErrorMsg
		}
		x.mode = value
	}

	// The conversion argument types are final. For untyped values the
	// conversion provides the type, per the spec: "A constant may be
	// given a type explicitly by a constant declaration or conversion,...".
	final = x.typ
	if isUntyped(final) {
		final = typ
		// For conversions to interfaces, use the argument type's
		// default type instead. Keep untyped nil for untyped nil
		// arguments.
		if _, ok := typ.Underlying().(*Interface); ok {
			final = defaultType(x.typ)
		}
	}
	check.updateExprType(x.expr, final, true)

	check.conversions[conv] = true // for cap/len checking
	x.expr = conv
	x.typ = typ
	return

ErrorMsg:
	check.invalidOp(conv.Pos(), "cannot convert %s to %s", x, typ)
Error:
	x.mode = invalid
	x.expr = conv
}

func (x *operand) isConvertible(ctxt *Context, T Type) bool {
	// "x is assignable to T"
	if x.isAssignable(ctxt, T) {
		return true
	}

	// "x's type and T have identical underlying types"
	V := x.typ
	Vu := V.Underlying()
	Tu := T.Underlying()
	if IsIdentical(Vu, Tu) {
		return true
	}

	// "x's type and T are unnamed pointer types and their pointer base types have identical underlying types"
	if V, ok := V.(*Pointer); ok {
		if T, ok := T.(*Pointer); ok {
			if IsIdentical(V.base.Underlying(), T.base.Underlying()) {
				return true
			}
		}
	}

	// "x's type and T are both integer or floating point types"
	if (isInteger(V) || isFloat(V)) && (isInteger(T) || isFloat(T)) {
		return true
	}

	// "x's type and T are both complex types"
	if isComplex(V) && isComplex(T) {
		return true
	}

	// "x is an integer or a slice of bytes or runes and T is a string type"
	if (isInteger(V) || isBytesOrRunes(Vu)) && isString(T) {
		return true
	}

	// "x is a string and T is a slice of bytes or runes"
	if isString(V) && isBytesOrRunes(Tu) {
		return true
	}

	// package unsafe:
	// "any pointer or value of underlying type uintptr can be converted into a unsafe.Pointer"
	if (isPointer(Vu) || isUintptr(Vu)) && isUnsafePointer(T) {
		return true
	}
	// "and vice versa"
	if isUnsafePointer(V) && (isPointer(Tu) || isUintptr(Tu)) {
		return true
	}

	return false
}

func isUintptr(typ Type) bool {
	t, ok := typ.(*Basic)
	return ok && t.kind == Uintptr
}

func isUnsafePointer(typ Type) bool {
	t, ok := typ.(*Basic)
	return ok && t.kind == UnsafePointer
}

func isPointer(typ Type) bool {
	_, ok := typ.(*Pointer)
	return ok
}

func isBytesOrRunes(typ Type) bool {
	if s, ok := typ.(*Slice); ok {
		t, ok := s.elt.Underlying().(*Basic)
		return ok && (t.kind == Byte || t.kind == Rune)
	}
	return false
}
