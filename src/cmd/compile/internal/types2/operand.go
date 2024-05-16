// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines operands and associated operations.

package types2

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	. "internal/types/errors"
)

// An operandMode specifies the (addressing) mode of an operand.
type operandMode byte

const (
	invalid   operandMode = iota // operand is invalid
	novalue                      // operand represents no value (result of a function call w/o result)
	builtin                      // operand is a built-in function
	typexpr                      // operand is a type
	constant_                    // operand is a constant; the operand's typ is a Basic type
	variable                     // operand is an addressable variable
	mapindex                     // operand is a map index expression (acts like a variable on lhs, commaok on rhs of an assignment)
	value                        // operand is a computed value
	nilvalue                     // operand is the nil value - only used by types2
	commaok                      // like value, but operand may be used in a comma,ok expression
	commaerr                     // like commaok, but second value is error, not boolean
	cgofunc                      // operand is a cgo function
)

var operandModeString = [...]string{
	invalid:   "invalid operand",
	novalue:   "no value",
	builtin:   "built-in",
	typexpr:   "type",
	constant_: "constant",
	variable:  "variable",
	mapindex:  "map index expression",
	value:     "value",
	nilvalue:  "nil", // only used by types2
	commaok:   "comma, ok expression",
	commaerr:  "comma, error expression",
	cgofunc:   "cgo function",
}

// An operand represents an intermediate value during type checking.
// Operands have an (addressing) mode, the expression evaluating to
// the operand, the operand's type, a value for constants, and an id
// for built-in functions.
// The zero value of operand is a ready to use invalid operand.
type operand struct {
	mode operandMode
	expr syntax.Expr
	typ  Type
	val  constant.Value
	id   builtinId
}

// Pos returns the position of the expression corresponding to x.
// If x is invalid the position is nopos.
func (x *operand) Pos() syntax.Pos {
	// x.expr may not be set if x is invalid
	if x.expr == nil {
		return nopos
	}
	return x.expr.Pos()
}

// Operand string formats
// (not all "untyped" cases can appear due to the type system,
// but they fall out naturally here)
//
// mode       format
//
// invalid    <expr> (               <mode>                    )
// novalue    <expr> (               <mode>                    )
// builtin    <expr> (               <mode>                    )
// typexpr    <expr> (               <mode>                    )
//
// constant   <expr> (<untyped kind> <mode>                    )
// constant   <expr> (               <mode>       of type <typ>)
// constant   <expr> (<untyped kind> <mode> <val>              )
// constant   <expr> (               <mode> <val> of type <typ>)
//
// variable   <expr> (<untyped kind> <mode>                    )
// variable   <expr> (               <mode>       of type <typ>)
//
// mapindex   <expr> (<untyped kind> <mode>                    )
// mapindex   <expr> (               <mode>       of type <typ>)
//
// value      <expr> (<untyped kind> <mode>                    )
// value      <expr> (               <mode>       of type <typ>)
//
// nilvalue   untyped nil
// nilvalue   nil    (                            of type <typ>)
//
// commaok    <expr> (<untyped kind> <mode>                    )
// commaok    <expr> (               <mode>       of type <typ>)
//
// commaerr   <expr> (<untyped kind> <mode>                    )
// commaerr   <expr> (               <mode>       of type <typ>)
//
// cgofunc    <expr> (<untyped kind> <mode>                    )
// cgofunc    <expr> (               <mode>       of type <typ>)
func operandString(x *operand, qf Qualifier) string {
	// special-case nil
	if isTypes2 {
		if x.mode == nilvalue {
			switch x.typ {
			case nil, Typ[Invalid]:
				return "nil (with invalid type)"
			case Typ[UntypedNil]:
				return "nil"
			default:
				return fmt.Sprintf("nil (of type %s)", TypeString(x.typ, qf))
			}
		}
	} else { // go/types
		if x.mode == value && x.typ == Typ[UntypedNil] {
			return "nil"
		}
	}

	var buf bytes.Buffer

	var expr string
	if x.expr != nil {
		expr = ExprString(x.expr)
	} else {
		switch x.mode {
		case builtin:
			expr = predeclaredFuncs[x.id].name
		case typexpr:
			expr = TypeString(x.typ, qf)
		case constant_:
			expr = x.val.String()
		}
	}

	// <expr> (
	if expr != "" {
		buf.WriteString(expr)
		buf.WriteString(" (")
	}

	// <untyped kind>
	hasType := false
	switch x.mode {
	case invalid, novalue, builtin, typexpr:
		// no type
	default:
		// should have a type, but be cautious (don't crash during printing)
		if x.typ != nil {
			if isUntyped(x.typ) {
				buf.WriteString(x.typ.(*Basic).name)
				buf.WriteByte(' ')
				break
			}
			hasType = true
		}
	}

	// <mode>
	buf.WriteString(operandModeString[x.mode])

	// <val>
	if x.mode == constant_ {
		if s := x.val.String(); s != expr {
			buf.WriteByte(' ')
			buf.WriteString(s)
		}
	}

	// <typ>
	if hasType {
		if isValid(x.typ) {
			var intro string
			if isGeneric(x.typ) {
				intro = " of generic type "
			} else {
				intro = " of type "
			}
			buf.WriteString(intro)
			WriteType(&buf, x.typ, qf)
			if tpar, _ := x.typ.(*TypeParam); tpar != nil {
				buf.WriteString(" constrained by ")
				WriteType(&buf, tpar.bound, qf) // do not compute interface type sets here
				// If we have the type set and it's empty, say so for better error messages.
				if hasEmptyTypeset(tpar) {
					buf.WriteString(" with empty type set")
				}
			}
		} else {
			buf.WriteString(" with invalid type")
		}
	}

	// )
	if expr != "" {
		buf.WriteByte(')')
	}

	return buf.String()
}

func (x *operand) String() string {
	return operandString(x, nil)
}

// setConst sets x to the untyped constant for literal lit.
func (x *operand) setConst(k syntax.LitKind, lit string) {
	var kind BasicKind
	switch k {
	case syntax.IntLit:
		kind = UntypedInt
	case syntax.FloatLit:
		kind = UntypedFloat
	case syntax.ImagLit:
		kind = UntypedComplex
	case syntax.RuneLit:
		kind = UntypedRune
	case syntax.StringLit:
		kind = UntypedString
	default:
		panic("unreachable")
	}

	val := makeFromLiteral(lit, k)
	if val.Kind() == constant.Unknown {
		x.mode = invalid
		x.typ = Typ[Invalid]
		return
	}
	x.mode = constant_
	x.typ = Typ[kind]
	x.val = val
}

// isNil reports whether x is the (untyped) nil value.
func (x *operand) isNil() bool {
	if isTypes2 {
		return x.mode == nilvalue
	} else { // go/types
		return x.mode == value && x.typ == Typ[UntypedNil]
	}
}

// assignableTo reports whether x is assignable to a variable of type T. If the
// result is false and a non-nil cause is provided, it may be set to a more
// detailed explanation of the failure (result != ""). The returned error code
// is only valid if the (first) result is false. The check parameter may be nil
// if assignableTo is invoked through an exported API call, i.e., when all
// methods have been type-checked.
func (x *operand) assignableTo(check *Checker, T Type, cause *string) (bool, Code) {
	if x.mode == invalid || !isValid(T) {
		return true, 0 // avoid spurious errors
	}

	V := x.typ

	// x's type is identical to T
	if Identical(V, T) {
		return true, 0
	}

	Vu := under(V)
	Tu := under(T)
	Vp, _ := V.(*TypeParam)
	Tp, _ := T.(*TypeParam)

	// x is an untyped value representable by a value of type T.
	if isUntyped(Vu) {
		assert(Vp == nil)
		if Tp != nil {
			// T is a type parameter: x is assignable to T if it is
			// representable by each specific type in the type set of T.
			return Tp.is(func(t *term) bool {
				if t == nil {
					return false
				}
				// A term may be a tilde term but the underlying
				// type of an untyped value doesn't change so we
				// don't need to do anything special.
				newType, _, _ := check.implicitTypeAndValue(x, t.typ)
				return newType != nil
			}), IncompatibleAssign
		}
		newType, _, _ := check.implicitTypeAndValue(x, T)
		return newType != nil, IncompatibleAssign
	}
	// Vu is typed

	// x's type V and T have identical underlying types
	// and at least one of V or T is not a named type
	// and neither V nor T is a type parameter.
	if Identical(Vu, Tu) && (!hasName(V) || !hasName(T)) && Vp == nil && Tp == nil {
		return true, 0
	}

	// T is an interface type, but not a type parameter, and V implements T.
	// Also handle the case where T is a pointer to an interface so that we get
	// the Checker.implements error cause.
	if _, ok := Tu.(*Interface); ok && Tp == nil || isInterfacePtr(Tu) {
		if check.implements(x.Pos(), V, T, false, cause) {
			return true, 0
		}
		// V doesn't implement T but V may still be assignable to T if V
		// is a type parameter; do not report an error in that case yet.
		if Vp == nil {
			return false, InvalidIfaceAssign
		}
		if cause != nil {
			*cause = ""
		}
	}

	// If V is an interface, check if a missing type assertion is the problem.
	if Vi, _ := Vu.(*Interface); Vi != nil && Vp == nil {
		if check.implements(x.Pos(), T, V, false, nil) {
			// T implements V, so give hint about type assertion.
			if cause != nil {
				*cause = "need type assertion"
			}
			return false, IncompatibleAssign
		}
	}

	// x is a bidirectional channel value, T is a channel
	// type, x's type V and T have identical element types,
	// and at least one of V or T is not a named type.
	if Vc, ok := Vu.(*Chan); ok && Vc.dir == SendRecv {
		if Tc, ok := Tu.(*Chan); ok && Identical(Vc.elem, Tc.elem) {
			return !hasName(V) || !hasName(T), InvalidChanAssign
		}
	}

	// optimization: if we don't have type parameters, we're done
	if Vp == nil && Tp == nil {
		return false, IncompatibleAssign
	}

	errorf := func(format string, args ...any) {
		if check != nil && cause != nil {
			msg := check.sprintf(format, args...)
			if *cause != "" {
				msg += "\n\t" + *cause
			}
			*cause = msg
		}
	}

	// x's type V is not a named type and T is a type parameter, and
	// x is assignable to each specific type in T's type set.
	if !hasName(V) && Tp != nil {
		ok := false
		code := IncompatibleAssign
		Tp.is(func(T *term) bool {
			if T == nil {
				return false // no specific types
			}
			ok, code = x.assignableTo(check, T.typ, cause)
			if !ok {
				errorf("cannot assign %s to %s (in %s)", x.typ, T.typ, Tp)
				return false
			}
			return true
		})
		return ok, code
	}

	// x's type V is a type parameter and T is not a named type,
	// and values x' of each specific type in V's type set are
	// assignable to T.
	if Vp != nil && !hasName(T) {
		x := *x // don't clobber outer x
		ok := false
		code := IncompatibleAssign
		Vp.is(func(V *term) bool {
			if V == nil {
				return false // no specific types
			}
			x.typ = V.typ
			ok, code = x.assignableTo(check, T, cause)
			if !ok {
				errorf("cannot assign %s (in %s) to %s", V.typ, Vp, T)
				return false
			}
			return true
		})
		return ok, code
	}

	return false, IncompatibleAssign
}
