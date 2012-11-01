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
		format := " %v"
		if isString(x.typ) {
			format = " %q"
		}
		fmt.Fprintf(&buf, format, x.val)
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

// isNil reports whether x is the predeclared nil constant.
func (x *operand) isNil() bool {
	return x.mode == constant && x.val == nilConst
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
	if x.isNil() {
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

type lookupResult struct {
	mode operandMode
	typ  Type
}

// lookupFieldRecursive is similar to FieldByNameFunc in reflect/type.go
// TODO(gri): FieldByNameFunc seems more complex - what are we missing?
func lookupFieldRecursive(list []*NamedType, name string) (res lookupResult) {
	// visited records the types that have been searched already
	visited := make(map[Type]bool)

	// embedded types of the next lower level
	var next []*NamedType

	potentialMatch := func(mode operandMode, typ Type) bool {
		if res.mode != invalid {
			// name appeared multiple times at this level - annihilate
			res.mode = invalid
			return false
		}
		res.mode = mode
		res.typ = typ
		return true
	}

	// look for name in all types of this level
	for len(list) > 0 {
		assert(res.mode == invalid)
		for _, typ := range list {
			if visited[typ] {
				// We have seen this type before, at a higher level.
				// That higher level shadows the lower level we are
				// at now, and either we would have found or not
				// found the field before. Ignore this type now.
				continue
			}
			visited[typ] = true

			// look for a matching attached method
			if data := typ.Obj.Data; data != nil {
				if obj := data.(*ast.Scope).Lookup(name); obj != nil {
					assert(obj.Type != nil)
					if !potentialMatch(value, obj.Type.(Type)) {
						return // name collision
					}
				}
			}

			switch typ := underlying(typ).(type) {
			case *Struct:
				// look for a matching fieldm and collect embedded types
				for _, f := range typ.Fields {
					if f.Name == name {
						assert(f.Type != nil)
						if !potentialMatch(variable, f.Type) {
							return // name collision
						}
						continue
					}
					// Collect embedded struct fields for searching the next
					// lower level, but only if we have not seen a match yet.
					// Embedded fields are always of the form T or *T where
					// T is a named type.
					if f.IsAnonymous && res.mode == invalid {
						next = append(next, deref(f.Type).(*NamedType))
					}
				}

			case *Interface:
				// look for a matching method
				for _, obj := range typ.Methods {
					if obj.Name == name {
						assert(obj.Type != nil)
						if !potentialMatch(value, obj.Type.(Type)) {
							return // name collision
						}
					}
				}
			}
		}

		if res.mode != invalid {
			// we found a match on this level
			return
		}

		// search the next level
		list = append(list[:0], next...) // don't waste underlying arrays
		next = next[:0]
	}
	return
}

func lookupField(typ Type, name string) (operandMode, Type) {
	typ = deref(typ)

	if typ, ok := typ.(*NamedType); ok {
		if data := typ.Obj.Data; data != nil {
			if obj := data.(*ast.Scope).Lookup(name); obj != nil {
				assert(obj.Type != nil)
				return value, obj.Type.(Type)
			}
		}
	}

	switch typ := underlying(typ).(type) {
	case *Struct:
		var list []*NamedType
		for _, f := range typ.Fields {
			if f.Name == name {
				return variable, f.Type
			}
			if f.IsAnonymous {
				list = append(list, deref(f.Type).(*NamedType))
			}
		}
		if len(list) > 0 {
			res := lookupFieldRecursive(list, name)
			return res.mode, res.typ
		}

	case *Interface:
		for _, obj := range typ.Methods {
			if obj.Name == name {
				return value, obj.Type.(Type)
			}
		}
	}

	// not found
	return invalid, nil
}
