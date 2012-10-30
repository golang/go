// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines operands and associated operations.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
)

// An operandMode specifies the (addressing) mode of an operand.
type operandMode int

const (
	invalid  operandMode = iota // operand is invalid (due to an earlier error) - ignore
	novalue                     // operand represents no value (result of a function call w/o result)
	typexpr                     // operand is a type
	constant                    // operand is a constant; the operand's typ is a Basic type
	variable                    // operand is an addressable variable
	value                       // operand is a computed value
	valueok                     // like mode == value, but operand may be used in a comma,ok expression
)

var operandModeString = [...]string{
	invalid:  "invalid",
	novalue:  "no value",
	typexpr:  "type",
	constant: "constant",
	variable: "variable",
	value:    "value",
	valueok:  "value,ok",
}

// An operand represents an intermediate value during type checking.
// Operands have an (addressing) mode, the expression evaluating to
// the operand, the operand's type, and for constants a constant value.
//
type operand struct {
	mode operandMode
	expr ast.Expr
	typ  Type
	val  interface{}
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

func (x *operand) String() string {
	if x.mode == invalid {
		return "invalid operand"
	}
	var buf bytes.Buffer
	if x.expr != nil {
		buf.WriteString(exprString(x.expr))
		buf.WriteString(" (")
	}
	buf.WriteString(operandModeString[x.mode])
	if x.mode == constant {
		fmt.Fprintf(&buf, " %v", x.val)
	}
	if x.mode != novalue && (x.mode != constant || !isUntyped(x.typ)) {
		fmt.Fprintf(&buf, " of type %s", typeString(x.typ))
	}
	if x.expr != nil {
		buf.WriteByte(')')
	}
	return buf.String()
}

// setConst sets x to the untyped constant for literal lit.
func (x *operand) setConst(tok token.Token, lit string) {
	x.mode = invalid

	var kind BasicKind
	var val interface{}
	switch tok {
	case token.INT:
		kind = UntypedInt
		val = makeIntConst(lit)

	case token.FLOAT:
		kind = UntypedFloat
		val = makeFloatConst(lit)

	case token.IMAG:
		kind = UntypedComplex
		val = makeComplexConst(lit)

	case token.CHAR:
		kind = UntypedRune
		val = makeRuneConst(lit)

	case token.STRING:
		kind = UntypedString
		val = makeStringConst(lit)
	}

	if val != nil {
		x.mode = constant
		x.typ = Typ[kind]
		x.val = val
	}
}

// implements reports whether x implements interface T.
func (x *operand) implements(T *Interface) bool {
	if x.mode == invalid {
		return true // avoid spurious errors
	}

	unimplemented()
	return true
}

// isAssignable reports whether x is assignable to a variable of type T.
func (x *operand) isAssignable(T Type) bool {
	if x.mode == invalid || T == Typ[Invalid] {
		return true // avoid spurious errors
	}

	V := x.typ

	// x's type is identical to T
	if isIdentical(V, T) {
		return true
	}

	Vu := underlying(V)
	Tu := underlying(T)

	// x's type V and T have identical underlying types
	// and at least one of V or T is not a named type
	if isIdentical(Vu, Tu) {
		return !isNamed(V) || !isNamed(T)
	}

	// T is an interface type and x implements T
	if Ti, ok := Tu.(*Interface); ok && x.implements(Ti) {
		return true
	}

	// x is a bidirectional channel value, T is a channel
	// type, x's type V and T have identical element types,
	// and at least one of V or T is not a named type
	if Vc, ok := Vu.(*Chan); ok && Vc.Dir == ast.SEND|ast.RECV {
		if Tc, ok := Tu.(*Chan); ok && isIdentical(Vc.Elt, Tc.Elt) {
			return !isNamed(V) || !isNamed(T)
		}
	}

	// x is the predeclared identifier nil and T is a pointer,
	// function, slice, map, channel, or interface type
	if x.typ == Typ[UntypedNil] {
		switch Tu.(type) {
		case *Pointer, *Signature, *Slice, *Map, *Chan, *Interface:
			return true
		}
		return false
	}

	// x is an untyped constant representable by a value of type T
	// - this is taken care of in the assignment check
	// TODO(gri) double-check - isAssignable is used elsewhere

	return false
}

// isInteger reports whether x is a (typed or untyped) integer value.
func (x *operand) isInteger() bool {
	return x.mode == invalid ||
		isInteger(x.typ) ||
		x.mode == constant && isRepresentableConst(x.val, UntypedInt)
}

// lookupField returns the struct field with the given name in typ.
// If no such field exists, the result is nil.
// TODO(gri) should this be a method of Struct?
//
func lookupField(typ *Struct, name string) *StructField {
	// TODO(gri) deal with embedding and conflicts - this is
	//           a very basic version to get going for now.
	for _, f := range typ.Fields {
		if f.Name == name {
			return f
		}
	}
	return nil
}
