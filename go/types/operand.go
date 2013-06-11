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

	"code.google.com/p/go.tools/go/exact"
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
	val  exact.Value
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

// isNil reports whether x is the predeclared nil constant.
func (x *operand) isNil() bool {
	return x.mode == constant && x.val.Kind() == exact.Nil
}

// TODO(gri) The functions operand.isAssignable, checker.convertUntyped,
//           checker.isRepresentable, and checker.assignOperand are
//           overlapping in functionality. Need to simplify and clean up.

// isAssignable reports whether x is assignable to a variable of type T.
func (x *operand) isAssignable(ctxt *Context, T Type) bool {
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
		if m, _ := missingMethod(x.typ, Ti); m == nil {
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
	if Vc, ok := Vu.(*Chan); ok && Vc.dir == ast.SEND|ast.RECV {
		if Tc, ok := Tu.(*Chan); ok && IsIdentical(Vc.elt, Tc.elt) {
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
				return isRepresentableConst(x.val, ctxt, t.kind)
			}
			// The result of a comparison is an untyped boolean,
			// but may not be a constant.
			if Vb, _ := Vu.(*Basic); Vb != nil {
				return Vb.kind == UntypedBool && isBoolean(Tu)
			}
		case *Interface:
			return x.isNil() || t.IsEmpty()
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
		x.mode == constant && isRepresentableConst(x.val, nil, UntypedInt) // no context required for UntypedInt
}

// lookupResult represents the result of a struct field/method lookup.
type lookupResult struct {
	mode  operandMode
	obj   Object // *Field or *Func; valid if mode != invalid
	index []int  // field index sequence; nil for methods
}

type embeddedType struct {
	typ       *Named
	index     []int // field index sequence
	multiples bool  // if set, typ is embedded multiple times at the same level
}

// lookupFieldBreadthFirst searches all types in list for a single entry (field
// or method) of the given name from the given package. If such a field is found,
// the result describes the field mode and type; otherwise the result mode is invalid.
// (This function is similar in structure to FieldByNameFunc in reflect/type.go)
//
func lookupFieldBreadthFirst(list []embeddedType, pkg *Package, name string) (res lookupResult) {
	// visited records the types that have been searched already.
	visited := make(map[*Named]bool)

	// embedded types of the next lower level
	var next []embeddedType

	// potentialMatch is invoked every time a match is found.
	potentialMatch := func(multiples bool, mode operandMode, obj Object) bool {
		if multiples || res.mode != invalid {
			// name appeared already at this level - annihilate
			res.mode = invalid
			return false
		}
		// first appearance of name
		res.mode = mode
		res.obj = obj
		res.index = nil
		return true
	}

	// Search the current level if there is any work to do and collect
	// embedded types of the next lower level in the next list.
	for len(list) > 0 {
		// The res.mode indicates whether we have found a match already
		// on this level (mode != invalid), or not (mode == invalid).
		assert(res.mode == invalid)

		// start with empty next list (don't waste underlying array)
		next = next[:0]

		// look for name in all types at this level
		for _, e := range list {
			typ := e.typ
			if visited[typ] {
				continue
			}
			visited[typ] = true

			// look for a matching attached method
			if obj := typ.methods.Lookup(pkg, name); obj != nil {
				m := obj.(*Func)
				assert(m.typ != nil)
				if !potentialMatch(e.multiples, value, m) {
					return // name collision
				}
			}

			switch t := typ.underlying.(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				if t.fields == nil {
					break
				}
				for i, obj := range t.fields.entries {
					f := obj.(*Field)
					if f.isMatch(pkg, name) {
						assert(f.typ != nil)
						if !potentialMatch(e.multiples, variable, f) {
							return // name collision
						}
						var index []int
						index = append(index, e.index...) // copy e.index
						index = append(index, i)
						res.index = index
						continue
					}
					// Collect embedded struct fields for searching the next
					// lower level, but only if we have not seen a match yet
					// (if we have a match it is either the desired field or
					// we have a name collision on the same level; in either
					// case we don't need to look further).
					// Embedded fields are always of the form T or *T where
					// T is a named type. If typ appeared multiple times at
					// this level, f.Type appears multiple times at the next
					// level.
					if f.anonymous && res.mode == invalid {
						// Ignore embedded basic types - only user-defined
						// named types can have methods or have struct fields.
						if t, _ := f.typ.Deref().(*Named); t != nil {
							var index []int
							index = append(index, e.index...) // copy e.index
							index = append(index, i)
							next = append(next, embeddedType{t, index, e.multiples})
						}
					}
				}

			case *Interface:
				// look for a matching method
				if obj := t.methods.Lookup(pkg, name); obj != nil {
					m := obj.(*Func)
					assert(m.typ != nil)
					if !potentialMatch(e.multiples, value, m) {
						return // name collision
					}
				}
			}
		}

		if res.mode != invalid {
			// we found a single match on this level
			return
		}

		// No match and no collision so far.
		// Compute the list to search for the next level.
		list = list[:0] // don't waste underlying array
		for _, e := range next {
			// Instead of adding the same type multiple times, look for
			// it in the list and mark it as multiple if it was added
			// before.
			// We use a sequential search (instead of a map for next)
			// because the lists tend to be small, can easily be reused,
			// and explicit search appears to be faster in this case.
			if alt := findType(list, e.typ); alt != nil {
				alt.multiples = true
			} else {
				list = append(list, e)
			}
		}

	}

	return
}

func findType(list []embeddedType, typ *Named) *embeddedType {
	for i := range list {
		if p := &list[i]; p.typ == typ {
			return p
		}
	}
	return nil
}

func lookupField(typ Type, pkg *Package, name string) lookupResult {
	typ = typ.Deref()

	if t, ok := typ.(*Named); ok {
		if obj := t.methods.Lookup(pkg, name); obj != nil {
			m := obj.(*Func)
			assert(m.typ != nil)
			return lookupResult{value, m, nil}
		}
		typ = t.underlying
	}

	switch t := typ.(type) {
	case *Struct:
		if t.fields == nil {
			break
		}
		var next []embeddedType
		for i, obj := range t.fields.entries {
			f := obj.(*Field)
			if f.isMatch(pkg, name) {
				return lookupResult{variable, f, []int{i}}
			}
			if f.anonymous {
				// Possible optimization: If the embedded type
				// is a pointer to the current type we could
				// ignore it.
				// Ignore embedded basic types - only user-defined
				// named types can have methods or have struct fields.
				if t, _ := f.typ.Deref().(*Named); t != nil {
					next = append(next, embeddedType{t, []int{i}, false})
				}
			}
		}
		if len(next) > 0 {
			return lookupFieldBreadthFirst(next, pkg, name)
		}

	case *Interface:
		if obj := t.methods.Lookup(pkg, name); obj != nil {
			m := obj.(*Func)
			assert(m.typ != nil)
			return lookupResult{value, m, nil}
		}
	}

	// not found
	return lookupResult{mode: invalid}
}
