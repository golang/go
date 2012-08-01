// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types declares the types used to represent Go types
// (UNDER CONSTRUCTION). ANY AND ALL PARTS MAY CHANGE.
//
package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"sort"
)

// All types implement the Type interface.
type Type interface {
	isType()
	String() string
}

// All concrete types embed implementsType which
// ensures that all types implement the Type interface.
type implementsType struct{}

func (t *implementsType) isType() {}

// A Bad type is a non-nil placeholder type when we don't know a type.
type Bad struct {
	implementsType
	Msg string // for better error reporting/debugging
}

func (t *Bad) String() string {
	return fmt.Sprintf("badType(%s)", t.Msg)
}

// A Basic represents a (unnamed) basic type.
type Basic struct {
	implementsType
	// TODO(gri) need a field specifying the exact basic type
}

func (t *Basic) String() string {
	// TODO(gri) print actual type information
	return "basicType"
}

// An Array represents an array type [Len]Elt.
type Array struct {
	implementsType
	Len uint64
	Elt Type
}

func (t *Array) String() string {
	return fmt.Sprintf("[%d]%s", t.Len, t.Elt)
}

// A Slice represents a slice type []Elt.
type Slice struct {
	implementsType
	Elt Type
}

func (t *Slice) String() string {
	return "[]" + t.Elt.String()
}

// A Struct represents a struct type struct{...}.
// Anonymous fields are represented by objects with empty names.
type Struct struct {
	implementsType
	Fields ObjList  // struct fields; or nil
	Tags   []string // corresponding tags; or nil
	// TODO(gri) This type needs some rethinking:
	// - at the moment anonymous fields are marked with "" object names,
	//   and their names have to be reconstructed
	// - there is no scope for fast lookup (but the parser creates one)
}

func (t *Struct) String() string {
	buf := bytes.NewBufferString("struct{")
	for i, fld := range t.Fields {
		if i > 0 {
			buf.WriteString("; ")
		}
		if fld.Name != "" {
			buf.WriteString(fld.Name)
			buf.WriteByte(' ')
		}
		buf.WriteString(fld.Type.(Type).String())
		if i < len(t.Tags) && t.Tags[i] != "" {
			fmt.Fprintf(buf, " %q", t.Tags[i])
		}
	}
	buf.WriteByte('}')
	return buf.String()
}

// A Pointer represents a pointer type *Base.
type Pointer struct {
	implementsType
	Base Type
}

func (t *Pointer) String() string {
	return "*" + t.Base.String()
}

// A Func represents a function type func(...) (...).
// Unnamed parameters are represented by objects with empty names.
type Func struct {
	implementsType
	Recv       *ast.Object // nil if not a method
	Params     ObjList     // (incoming) parameters from left to right; or nil
	Results    ObjList     // (outgoing) results from left to right; or nil
	IsVariadic bool        // true if the last parameter's type is of the form ...T
}

func writeParams(buf *bytes.Buffer, params ObjList, isVariadic bool) {
	buf.WriteByte('(')
	for i, par := range params {
		if i > 0 {
			buf.WriteString(", ")
		}
		if par.Name != "" {
			buf.WriteString(par.Name)
			buf.WriteByte(' ')
		}
		if isVariadic && i == len(params)-1 {
			buf.WriteString("...")
		}
		buf.WriteString(par.Type.(Type).String())
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, t *Func) {
	writeParams(buf, t.Params, t.IsVariadic)
	if len(t.Results) == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if len(t.Results) == 1 && t.Results[0].Name == "" {
		// single unnamed result
		buf.WriteString(t.Results[0].Type.(Type).String())
		return
	}

	// multiple or named result(s)
	writeParams(buf, t.Results, false)
}

func (t *Func) String() string {
	buf := bytes.NewBufferString("func")
	writeSignature(buf, t)
	return buf.String()
}

// An Interface represents an interface type interface{...}.
type Interface struct {
	implementsType
	Methods ObjList // interface methods sorted by name; or nil
}

func (t *Interface) String() string {
	buf := bytes.NewBufferString("interface{")
	for i, m := range t.Methods {
		if i > 0 {
			buf.WriteString("; ")
		}
		buf.WriteString(m.Name)
		writeSignature(buf, m.Type.(*Func))
	}
	buf.WriteByte('}')
	return buf.String()
}

// A Map represents a map type map[Key]Elt.
type Map struct {
	implementsType
	Key, Elt Type
}

func (t *Map) String() string {
	return fmt.Sprintf("map[%s]%s", t.Key, t.Elt)
}

// A Chan represents a channel type chan Elt, <-chan Elt, or chan<-Elt.
type Chan struct {
	implementsType
	Dir ast.ChanDir
	Elt Type
}

func (t *Chan) String() string {
	var s string
	switch t.Dir {
	case ast.SEND:
		s = "chan<- "
	case ast.RECV:
		s = "<-chan "
	default:
		s = "chan "
	}
	return s + t.Elt.String()
}

// A Name represents a named type as declared in a type declaration.
type Name struct {
	implementsType
	Underlying Type        // nil if not fully declared
	Obj        *ast.Object // corresponding declared object
	// TODO(gri) need to remember fields and methods.
}

func (t *Name) String() string {
	return t.Obj.Name
}

// If typ is a pointer type, Deref returns the pointer's base type;
// otherwise it returns typ.
func Deref(typ Type) Type {
	if typ, ok := typ.(*Pointer); ok {
		return typ.Base
	}
	return typ
}

// Underlying returns the underlying type of a type.
func Underlying(typ Type) Type {
	if typ, ok := typ.(*Name); ok {
		utyp := typ.Underlying
		if _, ok := utyp.(*Basic); !ok {
			return utyp
		}
		// the underlying type of a type name referring
		// to an (untyped) basic type is the basic type
		// name
	}
	return typ
}

// An ObjList represents an ordered (in some fashion) list of objects.
type ObjList []*ast.Object

// ObjList implements sort.Interface.
func (list ObjList) Len() int           { return len(list) }
func (list ObjList) Less(i, j int) bool { return list[i].Name < list[j].Name }
func (list ObjList) Swap(i, j int)      { list[i], list[j] = list[j], list[i] }

// Sort sorts an object list by object name.
func (list ObjList) Sort() { sort.Sort(list) }

// identicalTypes returns true if both lists a and b have the
// same length and corresponding objects have identical types.
func identicalTypes(a, b ObjList) bool {
	if len(a) == len(b) {
		for i, x := range a {
			y := b[i]
			if !Identical(x.Type.(Type), y.Type.(Type)) {
				return false
			}
		}
		return true
	}
	return false
}

// Identical returns true if two types are identical.
func Identical(x, y Type) bool {
	if x == y {
		return true
	}

	switch x := x.(type) {
	case *Bad:
		// A Bad type is always identical to any other type
		// (to avoid spurious follow-up errors).
		return true

	case *Basic:
		if y, ok := y.(*Basic); ok {
			panic("unimplemented")
			_ = y
		}

	case *Array:
		// Two array types are identical if they have identical element types
		// and the same array length.
		if y, ok := y.(*Array); ok {
			return x.Len == y.Len && Identical(x.Elt, y.Elt)
		}

	case *Slice:
		// Two slice types are identical if they have identical element types.
		if y, ok := y.(*Slice); ok {
			return Identical(x.Elt, y.Elt)
		}

	case *Struct:
		// Two struct types are identical if they have the same sequence of fields,
		// and if corresponding fields have the same names, and identical types,
		// and identical tags. Two anonymous fields are considered to have the same
		// name. Lower-case field names from different packages are always different.
		if y, ok := y.(*Struct); ok {
			// TODO(gri) handle structs from different packages
			if identicalTypes(x.Fields, y.Fields) {
				for i, f := range x.Fields {
					g := y.Fields[i]
					if f.Name != g.Name || x.Tags[i] != y.Tags[i] {
						return false
					}
				}
				return true
			}
		}

	case *Pointer:
		// Two pointer types are identical if they have identical base types.
		if y, ok := y.(*Pointer); ok {
			return Identical(x.Base, y.Base)
		}

	case *Func:
		// Two function types are identical if they have the same number of parameters
		// and result values, corresponding parameter and result types are identical,
		// and either both functions are variadic or neither is. Parameter and result
		// names are not required to match.
		if y, ok := y.(*Func); ok {
			return identicalTypes(x.Params, y.Params) &&
				identicalTypes(x.Results, y.Results) &&
				x.IsVariadic == y.IsVariadic
		}

	case *Interface:
		// Two interface types are identical if they have the same set of methods with
		// the same names and identical function types. Lower-case method names from
		// different packages are always different. The order of the methods is irrelevant.
		if y, ok := y.(*Interface); ok {
			return identicalTypes(x.Methods, y.Methods) // methods are sorted
		}

	case *Map:
		// Two map types are identical if they have identical key and value types.
		if y, ok := y.(*Map); ok {
			return Identical(x.Key, y.Key) && Identical(x.Elt, y.Elt)
		}

	case *Chan:
		// Two channel types are identical if they have identical value types
		// and the same direction.
		if y, ok := y.(*Chan); ok {
			return x.Dir == y.Dir && Identical(x.Elt, y.Elt)
		}

	case *Name:
		// Two named types are identical if their type names originate
		// in the same type declaration.
		if y, ok := y.(*Name); ok {
			return x.Obj == y.Obj ||
				// permit bad objects to be equal to avoid
				// follow up errors
				x.Obj != nil && x.Obj.Kind == ast.Bad ||
				y.Obj != nil && y.Obj.Kind == ast.Bad
		}
	}

	return false
}
