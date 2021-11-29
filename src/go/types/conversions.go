// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of conversions.

package types

import (
	"go/constant"
	"unicode"
)

// Conversion type-checks the conversion T(x).
// The result is in x.
func (check *Checker) conversion(x *operand, T Type) {
	constArg := x.mode == constant_

	constConvertibleTo := func(T Type, val *constant.Value) bool {
		switch t, _ := under(T).(*Basic); {
		case t == nil:
			// nothing to do
		case representableConst(x.val, check, t, val):
			return true
		case isInteger(x.typ) && isString(t):
			codepoint := unicode.ReplacementChar
			if i, ok := constant.Uint64Val(x.val); ok && i <= unicode.MaxRune {
				codepoint = rune(i)
			}
			if val != nil {
				*val = constant.MakeString(string(codepoint))
			}
			return true
		}
		return false
	}

	var ok bool
	var cause string
	switch {
	case constArg && isConstType(T):
		// constant conversion
		ok = constConvertibleTo(T, &x.val)
	case constArg && isTypeParam(T):
		// x is convertible to T if it is convertible
		// to each specific type in the type set of T.
		// If T's type set is empty, or if it doesn't
		// have specific types, constant x cannot be
		// converted.
		ok = T.(*TypeParam).underIs(func(u Type) bool {
			// t is nil if there are no specific type terms
			if u == nil {
				cause = check.sprintf("%s does not contain specific types", T)
				return false
			}
			if !constConvertibleTo(u, nil) {
				cause = check.sprintf("cannot convert %s to %s (in %s)", x, u, T)
				return false
			}
			return true
		})
		x.mode = value // type parameters are not constants
	case x.convertibleTo(check, T, &cause):
		// non-constant conversion
		ok = true
		x.mode = value
	}

	if !ok {
		// TODO(rfindley): use types2-style error reporting here.
		if compilerErrorMessages {
			if cause != "" {
				// Add colon at end of line if we have a following cause.
				check.errorf(x, _InvalidConversion, "cannot convert %s to type %s:\n\t%s", x, T, cause)
			} else {
				check.errorf(x, _InvalidConversion, "cannot convert %s to type %s", x, T)
			}
		} else {
			if cause != "" {
				check.errorf(x, _InvalidConversion, "cannot convert %s to %s (%s)", x, T, cause)
			} else {
				check.errorf(x, _InvalidConversion, "cannot convert %s to %s", x, T)
			}
		}
		x.mode = invalid
		return
	}

	// The conversion argument types are final. For untyped values the
	// conversion provides the type, per the spec: "A constant may be
	// given a type explicitly by a constant declaration or conversion,...".
	if isUntyped(x.typ) {
		final := T
		// - For conversions to interfaces, use the argument's default type.
		// - For conversions of untyped constants to non-constant types, also
		//   use the default type (e.g., []byte("foo") should report string
		//   not []byte as type for the constant "foo").
		// - Keep untyped nil for untyped nil arguments.
		// - For integer to string conversions, keep the argument type.
		//   (See also the TODO below.)
		if IsInterface(T) && !isTypeParam(T) || constArg && !isConstType(T) || x.isNil() {
			final = Default(x.typ) // default type of untyped nil is untyped nil
		} else if isInteger(x.typ) && allString(T) {
			final = x.typ
		}
		check.updateExprType(x.expr, final, true)
	}

	x.typ = T
}

// TODO(gri) convertibleTo checks if T(x) is valid. It assumes that the type
// of x is fully known, but that's not the case for say string(1<<s + 1.0):
// Here, the type of 1<<s + 1.0 will be UntypedFloat which will lead to the
// (correct!) refusal of the conversion. But the reported error is essentially
// "cannot convert untyped float value to string", yet the correct error (per
// the spec) is that we cannot shift a floating-point value: 1 in 1<<s should
// be converted to UntypedFloat because of the addition of 1.0. Fixing this
// is tricky because we'd have to run updateExprType on the argument first.
// (Issue #21982.)

// convertibleTo reports whether T(x) is valid. In the failure case, *cause
// may be set to the cause for the failure.
// The check parameter may be nil if convertibleTo is invoked through an
// exported API call, i.e., when all methods have been type-checked.
func (x *operand) convertibleTo(check *Checker, T Type, cause *string) bool {
	// "x is assignable to T"
	if ok, _ := x.assignableTo(check, T, cause); ok {
		return true
	}

	// "V and T have identical underlying types if tags are ignored
	// and V and T are not type parameters"
	V := x.typ
	Vu := under(V)
	Tu := under(T)
	Vp, _ := V.(*TypeParam)
	Tp, _ := T.(*TypeParam)
	if IdenticalIgnoreTags(Vu, Tu) && Vp == nil && Tp == nil {
		return true
	}

	// "V and T are unnamed pointer types and their pointer base types
	// have identical underlying types if tags are ignored
	// and their pointer base types are not type parameters"
	if V, ok := V.(*Pointer); ok {
		if T, ok := T.(*Pointer); ok {
			if IdenticalIgnoreTags(under(V.base), under(T.base)) && !isTypeParam(V.base) && !isTypeParam(T.base) {
				return true
			}
		}
	}

	// "V and T are both integer or floating point types"
	if isIntegerOrFloat(Vu) && isIntegerOrFloat(Tu) {
		return true
	}

	// "V and T are both complex types"
	if isComplex(Vu) && isComplex(Tu) {
		return true
	}

	// "V is an integer or a slice of bytes or runes and T is a string type"
	if (isInteger(Vu) || isBytesOrRunes(Vu)) && isString(Tu) {
		return true
	}

	// "V is a string and T is a slice of bytes or runes"
	if isString(Vu) && isBytesOrRunes(Tu) {
		return true
	}

	// package unsafe:
	// "any pointer or value of underlying type uintptr can be converted into a unsafe.Pointer"
	if (isPointer(Vu) || isUintptr(Vu)) && isUnsafePointer(Tu) {
		return true
	}
	// "and vice versa"
	if isUnsafePointer(Vu) && (isPointer(Tu) || isUintptr(Tu)) {
		return true
	}

	// "V a slice, T is a pointer-to-array type,
	// and the slice and array types have identical element types."
	if s, _ := Vu.(*Slice); s != nil {
		if p, _ := Tu.(*Pointer); p != nil {
			if a, _ := under(p.Elem()).(*Array); a != nil {
				if Identical(s.Elem(), a.Elem()) {
					if check == nil || check.allowVersion(check.pkg, 1, 17) {
						return true
					}
					if cause != nil {
						*cause = "conversion of slices to array pointers requires go1.17 or later"
					}
				}
			}
		}
	}

	// optimization: if we don't have type parameters, we're done
	if Vp == nil && Tp == nil {
		return false
	}

	errorf := func(format string, args ...interface{}) {
		if check != nil && cause != nil {
			msg := check.sprintf(format, args...)
			if *cause != "" {
				msg += "\n\t" + *cause
			}
			*cause = msg
		}
	}

	// generic cases with specific type terms
	// (generic operands cannot be constants, so we can ignore x.val)
	switch {
	case Vp != nil && Tp != nil:
		x := *x // don't clobber outer x
		return Vp.is(func(V *term) bool {
			if V == nil {
				return false // no specific types
			}
			x.typ = V.typ
			return Tp.is(func(T *term) bool {
				if T == nil {
					return false // no specific types
				}
				if !x.convertibleTo(check, T.typ, cause) {
					errorf("cannot convert %s (in %s) to %s (in %s)", V.typ, Vp, T.typ, Tp)
					return false
				}
				return true
			})
		})
	case Vp != nil:
		x := *x // don't clobber outer x
		return Vp.is(func(V *term) bool {
			if V == nil {
				return false // no specific types
			}
			x.typ = V.typ
			if !x.convertibleTo(check, T, cause) {
				errorf("cannot convert %s (in %s) to %s", V.typ, Vp, T)
				return false
			}
			return true
		})
	case Tp != nil:
		return Tp.is(func(T *term) bool {
			if T == nil {
				return false // no specific types
			}
			if !x.convertibleTo(check, T.typ, cause) {
				errorf("cannot convert %s to %s (in %s)", x.typ, T.typ, Tp)
				return false
			}
			return true
		})
	}

	return false
}

func isUintptr(typ Type) bool {
	t, _ := under(typ).(*Basic)
	return t != nil && t.kind == Uintptr
}

func isUnsafePointer(typ Type) bool {
	t, _ := under(typ).(*Basic)
	return t != nil && t.kind == UnsafePointer
}

func isPointer(typ Type) bool {
	_, ok := under(typ).(*Pointer)
	return ok
}

func isBytesOrRunes(typ Type) bool {
	if s, _ := under(typ).(*Slice); s != nil {
		t, _ := under(s.elem).(*Basic)
		return t != nil && (t.kind == Byte || t.kind == Rune)
	}
	return false
}
