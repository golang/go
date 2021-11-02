// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of conversions.

package types2

import (
	"go/constant"
	"unicode"
)

// Conversion type-checks the conversion T(x).
// The result is in x.
func (check *Checker) conversion(x *operand, T Type) {
	constArg := x.mode == constant_

	constConvertibleTo := func(T Type, val *constant.Value) bool {
		switch t := asBasic(T); {
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
		ok = under(T).(*TypeParam).underIs(func(u Type) bool {
			// t is nil if there are no specific type terms
			// TODO(gri) add a cause in case of failure
			return u != nil && constConvertibleTo(u, nil)
		})
		x.mode = value // type parameters are not constants
	case x.convertibleTo(check, T, &cause):
		// non-constant conversion
		ok = true
		x.mode = value
	}

	if !ok {
		var err error_
		err.errorf(x, "cannot convert %s to %s", x, T)
		if cause != "" {
			err.errorf(nopos, cause)
		}
		check.report(&err)
		x.mode = invalid
		return
	}

	// The conversion argument types are final. For untyped values the
	// conversion provides the type, per the spec: "A constant may be
	// given a type explicitly by a constant declaration or conversion,...".
	if isUntyped(x.typ) {
		final := T
		// - For conversions to interfaces, except for untyped nil arguments,
		//   use the argument's default type.
		// - For conversions of untyped constants to non-constant types, also
		//   use the default type (e.g., []byte("foo") should report string
		//   not []byte as type for the constant "foo").
		// - For integer to string conversions, keep the argument type.
		//   (See also the TODO below.)
		if x.typ == Typ[UntypedNil] {
			// ok
		} else if IsInterface(T) || constArg && !isConstType(T) {
			final = Default(x.typ)
		} else if isInteger(x.typ) && isString(T) {
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

	// determine type parameter operands with specific type terms
	Vp, _ := under(x.typ).(*TypeParam)
	Tp, _ := under(T).(*TypeParam)
	if Vp != nil && !Vp.hasTerms() {
		Vp = nil
	}
	if Tp != nil && !Tp.hasTerms() {
		Tp = nil
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
		return Vp.is(func(V *term) bool {
			return Tp.is(func(T *term) bool {
				if !convertibleToImpl(check, V.typ, T.typ, cause) {
					errorf("cannot convert %s (in %s) to %s (in %s)", V.typ, Vp, T.typ, Tp)
					return false
				}
				return true
			})
		})
	case Vp != nil:
		return Vp.is(func(V *term) bool {
			if !convertibleToImpl(check, V.typ, T, cause) {
				errorf("cannot convert %s (in %s) to %s", V.typ, Vp, T)
				return false
			}
			return true
		})
	case Tp != nil:
		return Tp.is(func(T *term) bool {
			if !convertibleToImpl(check, x.typ, T.typ, cause) {
				errorf("cannot convert %s to %s (in %s)", x.typ, T.typ, Tp)
				return false
			}
			return true
		})
	}

	// non-generic case
	return convertibleToImpl(check, x.typ, T, cause)
}

// convertibleToImpl should only be called by convertibleTo
func convertibleToImpl(check *Checker, V, T Type, cause *string) bool {
	// "V and T have identical underlying types if tags are ignored"
	Vu := under(V)
	Tu := under(T)
	if IdenticalIgnoreTags(Vu, Tu) {
		return true
	}

	// "V and T are unnamed pointer types and their pointer base types
	// have identical underlying types if tags are ignored"
	if V, ok := V.(*Pointer); ok {
		if T, ok := T.(*Pointer); ok {
			if IdenticalIgnoreTags(under(V.base), under(T.base)) {
				return true
			}
		}
	}

	// "V and T are both integer or floating point types"
	if isIntegerOrFloat(V) && isIntegerOrFloat(T) {
		return true
	}

	// "V and T are both complex types"
	if isComplex(V) && isComplex(T) {
		return true
	}

	// "V is an integer or a slice of bytes or runes and T is a string type"
	if (isInteger(V) || isBytesOrRunes(Vu)) && isString(T) {
		return true
	}

	// "V is a string and T is a slice of bytes or runes"
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

	// "V a slice, T is a pointer-to-array type,
	// and the slice and array types have identical element types."
	if s := asSlice(V); s != nil {
		if p := asPointer(T); p != nil {
			if a := asArray(p.Elem()); a != nil {
				if Identical(s.Elem(), a.Elem()) {
					if check == nil || check.allowVersion(check.pkg, 1, 17) {
						return true
					}
					// check != nil
					if cause != nil {
						if check.conf.CompilerErrorMessages {
							// compiler error message assumes a -lang flag
							*cause = "conversion of slices to array pointers only supported as of -lang=go1.17"
						} else {
							*cause = "conversion of slices to array pointers requires go1.17 or later"
						}
					}
					return false
				}
			}
		}
	}

	return false
}

// Helper predicates for convertibleToImpl. The types provided to convertibleToImpl
// may be type parameters but they won't have specific type terms. Thus it is ok to
// use the toT convenience converters in the predicates below.

func isUintptr(typ Type) bool {
	t := asBasic(typ)
	return t != nil && t.kind == Uintptr
}

func isUnsafePointer(typ Type) bool {
	// TODO(gri): Is this asBasic(typ) instead of typ.(*Basic) correct?
	//            (The former calls under(), while the latter doesn't.)
	//            The spec does not say so, but gc claims it is. See also
	//            issue 6326.
	t := asBasic(typ)
	return t != nil && t.kind == UnsafePointer
}

func isPointer(typ Type) bool {
	return asPointer(typ) != nil
}

func isBytesOrRunes(typ Type) bool {
	if s := asSlice(typ); s != nil {
		t := asBasic(s.elem)
		return t != nil && (t.kind == Byte || t.kind == Rune)
	}
	return false
}
