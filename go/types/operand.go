// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines operands and associated operations.

package types

import (
	"bytes"
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// An operandMode specifies the (addressing) mode of an operand.
type operandMode int

const (
	invalid  operandMode = iota // operand is invalid
	novalue                     // operand represents no value (result of a function call w/o result)
	builtin                     // operand is a built-in function
	typexpr                     // operand is a type
	constant                    // operand is a constant; the operand's typ is a Basic type
	variable                    // operand is an addressable variable
	mapindex                    // operand is a map index expression (acts like a variable on lhs, commaok on rhs of an assignment)
	value                       // operand is a computed value
	commaok                     // like value, but operand may be used in a comma,ok expression
)

var operandModeString = [...]string{
	invalid:  "invalid operand",
	novalue:  "no value",
	builtin:  "built-in",
	typexpr:  "type",
	constant: "constant",
	variable: "variable",
	mapindex: "map index expression",
	value:    "value",
	commaok:  "comma, ok expression",
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
	val  exact.Value
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
func (x *operand) String() string {
	var buf bytes.Buffer

	var expr string
	if x.expr != nil {
		expr = ExprString(x.expr)
	} else {
		switch x.mode {
		case builtin:
			expr = predeclaredFuncs[x.id].name
		case typexpr:
			expr = TypeString(nil, x.typ)
		case constant:
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
		// has type
		if isUntyped(x.typ) {
			buf.WriteString(x.typ.(*Basic).name)
			buf.WriteByte(' ')
			break
		}
		hasType = true
	}

	// <mode>
	buf.WriteString(operandModeString[x.mode])

	// <val>
	if x.mode == constant {
		if s := x.val.String(); s != expr {
			buf.WriteByte(' ')
			buf.WriteString(s)
		}
	}

	// <typ>
	if hasType {
		if x.typ != Typ[Invalid] {
			buf.WriteString(" of type ")
			WriteType(&buf, nil, x.typ)
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

// setConst sets x to the untyped constant for literal lit.
func (x *operand) setConst(tok token.Token, lit string) {
	val := exact.MakeFromLiteral(lit, tok)
	if val == nil {
		// TODO(gri) Should we make it an unknown constant instead?
		x.mode = invalid
		return
	}

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
	}

	x.mode = constant
	x.typ = Typ[kind]
	x.val = val
}

// isNil reports whether x is the nil value.
func (x *operand) isNil() bool {
	return x.mode == value && x.typ == Typ[UntypedNil]
}

// TODO(gri) The functions operand.isAssignableTo, checker.convertUntyped,
//           checker.isRepresentable, and checker.assignment are
//           overlapping in functionality. Need to simplify and clean up.

// isAssignableTo reports whether x is assignable to a variable of type T.
func (x *operand) isAssignableTo(conf *Config, T Type) bool {
	if x.mode == invalid || T == Typ[Invalid] {
		return true // avoid spurious errors
	}

	V := x.typ

	// x's type is identical to T
	if IsIdentical(V, T) {
		return true
	}

	Vu := V.Underlying()
	Tu := T.Underlying()

	// T is an interface type and x implements T
	// (Do this check first as it might succeed early.)
	if Ti, ok := Tu.(*Interface); ok {
		if m, _ := MissingMethod(x.typ, Ti, true); m == nil {
			return true
		}
	}

	// x's type V and T have identical underlying types
	// and at least one of V or T is not a named type
	if IsIdentical(Vu, Tu) && (!isNamed(V) || !isNamed(T)) {
		return true
	}

	// x is a bidirectional channel value, T is a channel
	// type, x's type V and T have identical element types,
	// and at least one of V or T is not a named type
	if Vc, ok := Vu.(*Chan); ok && Vc.dir == SendRecv {
		if Tc, ok := Tu.(*Chan); ok && IsIdentical(Vc.elem, Tc.elem) {
			return !isNamed(V) || !isNamed(T)
		}
	}

	// x is the predeclared identifier nil and T is a pointer,
	// function, slice, map, channel, or interface type
	if x.isNil() {
		switch t := Tu.(type) {
		case *Basic:
			if t.kind == UnsafePointer {
				return true
			}
		case *Pointer, *Signature, *Slice, *Map, *Chan, *Interface:
			return true
		}
		return false
	}

	// x is an untyped constant representable by a value of type T
	// TODO(gri) This is borrowing from checker.convertUntyped and
	//           checker.isRepresentable. Need to clean up.
	if isUntyped(Vu) {
		switch t := Tu.(type) {
		case *Basic:
			if x.mode == constant {
				return isRepresentableConst(x.val, conf, t.kind, nil)
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

	return false
}

// isInteger reports whether x is a (typed or untyped) integer value.
func (x *operand) isInteger() bool {
	return x.mode == invalid ||
		isInteger(x.typ) ||
		x.mode == constant && isRepresentableConst(x.val, nil, UntypedInt, nil) // no *Config required for UntypedInt
}
