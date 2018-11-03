// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines operands and associated operations.

package types

import (
	"bytes"
	"go/ast"
	"go/constant"
	"go/token"
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
	commaok                      // like value, but operand may be used in a comma,ok expression
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
	commaok:   "comma, ok expression",
}

// An operand represents an intermediate value during type checking.
// Operands have an (addressing) mode, the expression evaluating to
// the operand, the operand's type, a value for constants, and an id
// for built-in functions.
// The zero value of operand is a ready to use invalid operand.
//
type operand struct {
	mode operandMode
	expr ast.Expr
	typ  Type
	val  constant.Value
	id   builtinId
}

// pos returns the position of the expression corresponding to x.
// If x is invalid the position is token.NoPos.
//
func (x *operand) pos() token.Pos {
	// x.expr may not be set if x is invalid
	if x.expr == nil {
		return token.NoPos
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
// commaok    <expr> (<untyped kind> <mode>                    )
// commaok    <expr> (               <mode>       of type <typ>)
//
func operandString(x *operand, qf Qualifier) string {
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
		if x.typ != Typ[Invalid] {
			buf.WriteString(" of type ")
			WriteType(&buf, x.typ, qf)
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
func (x *operand) setConst(tok token.Token, lit string) {
	var kind BasicKind
	switch tok {
	case token.INT:
		kind = UntypedInt
	case token.FLOAT:
		kind = UntypedFloat
	case token.IMAG:
		kind = UntypedComplex
	case token.CHAR:
		kind = UntypedRune
	case token.STRING:
		kind = UntypedString
	default:
		unreachable()
	}

	x.mode = constant_
	x.typ = Typ[kind]
	x.val = constant.MakeFromLiteral(lit, tok, 0)
}

// isNil reports whether x is the nil value.
func (x *operand) isNil() bool {
	return x.mode == value && x.typ == Typ[UntypedNil]
}

// TODO(gri) The functions operand.assignableTo, checker.convertUntyped,
//           checker.representable, and checker.assignment are
//           overlapping in functionality. Need to simplify and clean up.

// assignableTo reports whether x is assignable to a variable of type T.
// If the result is false and a non-nil reason is provided, it may be set
// to a more detailed explanation of the failure (result != "").
// The check parameter may be nil if assignableTo is invoked through
// an exported API call, i.e., when all methods have been type-checked.
func (x *operand) assignableTo(check *Checker, T Type, reason *string) bool {
	if x.mode == invalid || T == Typ[Invalid] {
		return true // avoid spurious errors
	}

	V := x.typ

	// x's type is identical to T
	if Identical(V, T) {
		return true
	}

	Vu := V.Underlying()
	Tu := T.Underlying()

	// x is an untyped value representable by a value of type T
	// TODO(gri) This is borrowing from checker.convertUntyped and
	//           checker.representable. Need to clean up.
	if isUntyped(Vu) {
		switch t := Tu.(type) {
		case *Basic:
			if x.isNil() && t.kind == UnsafePointer {
				return true
			}
			if x.mode == constant_ {
				return representableConst(x.val, check, t, nil)
			}
			// The result of a comparison is an untyped boolean,
			// but may not be a constant.
			if Vb, _ := Vu.(*Basic); Vb != nil {
				return Vb.kind == UntypedBool && isBoolean(Tu)
			}
		case *Interface:
			return x.isNil() || t.Empty()
		case *Pointer, *Signature, *Slice, *Map, *Chan:
			return x.isNil()
		}
	}
	// Vu is typed

	// x's type V and T have identical underlying types
	// and at least one of V or T is not a named type
	if Identical(Vu, Tu) && (!isNamed(V) || !isNamed(T)) {
		return true
	}

	// T is an interface type and x implements T
	if Ti, ok := Tu.(*Interface); ok {
		if m, wrongType := check.missingMethod(x.typ, Ti, true); m != nil /* Implements(x.typ, Ti) */ {
			if reason != nil {
				if wrongType {
					*reason = "wrong type for method " + m.Name()
				} else {
					*reason = "missing method " + m.Name()
				}
			}
			return false
		}
		return true
	}

	// x is a bidirectional channel value, T is a channel
	// type, x's type V and T have identical element types,
	// and at least one of V or T is not a named type
	if Vc, ok := Vu.(*Chan); ok && Vc.dir == SendRecv {
		if Tc, ok := Tu.(*Chan); ok && Identical(Vc.elem, Tc.elem) {
			return !isNamed(V) || !isNamed(T)
		}
	}

	return false
}
