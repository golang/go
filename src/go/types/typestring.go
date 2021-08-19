// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of types.

package types

import (
	"bytes"
	"fmt"
	"go/token"
	"unicode/utf8"
)

// A Qualifier controls how named package-level objects are printed in
// calls to TypeString, ObjectString, and SelectionString.
//
// These three formatting routines call the Qualifier for each
// package-level object O, and if the Qualifier returns a non-empty
// string p, the object is printed in the form p.O.
// If it returns an empty string, only the object name O is printed.
//
// Using a nil Qualifier is equivalent to using (*Package).Path: the
// object is qualified by the import path, e.g., "encoding/json.Marshal".
//
type Qualifier func(*Package) string

// RelativeTo returns a Qualifier that fully qualifies members of
// all packages other than pkg.
func RelativeTo(pkg *Package) Qualifier {
	if pkg == nil {
		return nil
	}
	return func(other *Package) string {
		if pkg == other {
			return "" // same package; unqualified
		}
		return other.Path()
	}
}

// TypeString returns the string representation of typ.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func TypeString(typ Type, qf Qualifier) string {
	var buf bytes.Buffer
	WriteType(&buf, typ, qf)
	return buf.String()
}

// WriteType writes the string representation of typ to buf.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func WriteType(buf *bytes.Buffer, typ Type, qf Qualifier) {
	writeType(buf, typ, qf, make([]Type, 0, 8))
}

// instanceMarker is the prefix for an instantiated type
// in "non-evaluated" instance form.
const instanceMarker = '#'

func writeType(buf *bytes.Buffer, typ Type, qf Qualifier, visited []Type) {
	// Theoretically, this is a quadratic lookup algorithm, but in
	// practice deeply nested composite types with unnamed component
	// types are uncommon. This code is likely more efficient than
	// using a map.
	for _, t := range visited {
		if t == typ {
			fmt.Fprintf(buf, "○%T", goTypeName(typ)) // cycle to typ
			return
		}
	}
	visited = append(visited, typ)

	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		// exported basic types go into package unsafe
		// (currently this is just unsafe.Pointer)
		if token.IsExported(t.name) {
			if obj, _ := Unsafe.scope.Lookup(t.name).(*TypeName); obj != nil {
				writeTypeName(buf, obj, qf)
				break
			}
		}
		buf.WriteString(t.name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.len)
		writeType(buf, t.elem, qf, visited)

	case *Slice:
		buf.WriteString("[]")
		writeType(buf, t.elem, qf, visited)

	case *Struct:
		buf.WriteString("struct{")
		for i, f := range t.fields {
			if i > 0 {
				buf.WriteString("; ")
			}
			// This doesn't do the right thing for embedded type
			// aliases where we should print the alias name, not
			// the aliased type (see issue #44410).
			if !f.embedded {
				buf.WriteString(f.name)
				buf.WriteByte(' ')
			}
			writeType(buf, f.typ, qf, visited)
			if tag := t.Tag(i); tag != "" {
				fmt.Fprintf(buf, " %q", tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		writeType(buf, t.base, qf, visited)

	case *Tuple:
		writeTuple(buf, t, false, qf, visited)

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, t, qf, visited)

	case *Union:
		// Unions only appear as (syntactic) embedded elements
		// in interfaces and syntactically cannot be empty.
		if t.Len() == 0 {
			panic("empty union")
		}
		for i, t := range t.terms {
			if i > 0 {
				buf.WriteByte('|')
			}
			if t.tilde {
				buf.WriteByte('~')
			}
			writeType(buf, t.typ, qf, visited)
		}

	case *Interface:
		buf.WriteString("interface{")
		first := true
		// print explicit interface methods and embedded types
		for _, m := range t.methods {
			if !first {
				buf.WriteString("; ")
			}
			first = false
			buf.WriteString(m.name)
			writeSignature(buf, m.typ.(*Signature), qf, visited)
		}
		for _, typ := range t.embeddeds {
			if !first {
				buf.WriteString("; ")
			}
			first = false
			writeType(buf, typ, qf, visited)
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		writeType(buf, t.key, qf, visited)
		buf.WriteByte(']')
		writeType(buf, t.elem, qf, visited)

	case *Chan:
		var s string
		var parens bool
		switch t.dir {
		case SendRecv:
			s = "chan "
			// chan (<-chan T) requires parentheses
			if c, _ := t.elem.(*Chan); c != nil && c.dir == RecvOnly {
				parens = true
			}
		case SendOnly:
			s = "chan<- "
		case RecvOnly:
			s = "<-chan "
		default:
			unreachable()
		}
		buf.WriteString(s)
		if parens {
			buf.WriteByte('(')
		}
		writeType(buf, t.elem, qf, visited)
		if parens {
			buf.WriteByte(')')
		}

	case *Named:
		if t.instance != nil {
			buf.WriteByte(instanceMarker)
		}
		writeTypeName(buf, t.obj, qf)
		if t.targs != nil {
			// instantiated type
			buf.WriteByte('[')
			writeTypeList(buf, t.targs.list(), qf, visited)
			buf.WriteByte(']')
		} else if t.TParams().Len() != 0 {
			// parameterized type
			writeTParamList(buf, t.TParams().list(), qf, visited)
		}

	case *TypeParam:
		s := "?"
		if t.obj != nil {
			// Optionally write out package for typeparams (like Named).
			// TODO(rfindley): this is required for import/export, so
			// we maybe need a separate function that won't be changed
			// for debugging purposes.
			if t.obj.pkg != nil {
				writePackage(buf, t.obj.pkg, qf)
			}
			s = t.obj.name
		}
		buf.WriteString(s + subscript(t.id))

	case *top:
		buf.WriteString("⊤")

	default:
		// For externally defined implementations of Type.
		// Note: In this case cycles won't be caught.
		buf.WriteString(t.String())
	}
}

func writeTypeList(buf *bytes.Buffer, list []Type, qf Qualifier, visited []Type) {
	for i, typ := range list {
		if i > 0 {
			buf.WriteString(", ")
		}
		writeType(buf, typ, qf, visited)
	}
}

func writeTParamList(buf *bytes.Buffer, list []*TypeParam, qf Qualifier, visited []Type) {
	// TODO(rFindley) compare this with the corresponding implementation in types2
	buf.WriteString("[")
	var prev Type
	for i, tpar := range list {
		// Determine the type parameter and its constraint.
		// list is expected to hold type parameter names,
		// but don't crash if that's not the case.
		var bound Type
		if tpar != nil {
			bound = tpar.bound // should not be nil but we want to see it if it is
		}

		if i > 0 {
			if bound != prev {
				// bound changed - write previous one before advancing
				buf.WriteByte(' ')
				writeType(buf, prev, qf, visited)
			}
			buf.WriteString(", ")
		}
		prev = bound

		if tpar != nil {
			writeType(buf, tpar, qf, visited)
		} else {
			buf.WriteString(tpar.obj.name)
		}
	}
	if prev != nil {
		buf.WriteByte(' ')
		writeType(buf, prev, qf, visited)
	}
	buf.WriteByte(']')
}

func writeTypeName(buf *bytes.Buffer, obj *TypeName, qf Qualifier) {
	if obj == nil {
		buf.WriteString("<Named w/o object>")
		return
	}
	if obj.pkg != nil {
		writePackage(buf, obj.pkg, qf)
	}
	buf.WriteString(obj.name)

	if instanceHashing != 0 {
		// For local defined types, use the (original!) TypeName's scope
		// numbers to disambiguate.
		typ := obj.typ.(*Named)
		// TODO(gri) Figure out why typ.orig != typ.orig.orig sometimes
		//           and whether the loop can iterate more than twice.
		//           (It seems somehow connected to instance types.)
		for typ.orig != typ {
			typ = typ.orig
		}
		writeScopeNumbers(buf, typ.obj.parent)
	}
}

// writeScopeNumbers writes the number sequence for this scope to buf
// in the form ".i.j.k" where i, j, k, etc. stand for scope numbers.
// If a scope is nil or has no parent (such as a package scope), nothing
// is written.
func writeScopeNumbers(buf *bytes.Buffer, s *Scope) {
	if s != nil && s.number > 0 {
		writeScopeNumbers(buf, s.parent)
		fmt.Fprintf(buf, ".%d", s.number)
	}
}

func writeTuple(buf *bytes.Buffer, tup *Tuple, variadic bool, qf Qualifier, visited []Type) {
	buf.WriteByte('(')
	if tup != nil {
		for i, v := range tup.vars {
			if i > 0 {
				buf.WriteString(", ")
			}
			if v.name != "" {
				buf.WriteString(v.name)
				buf.WriteByte(' ')
			}
			typ := v.typ
			if variadic && i == len(tup.vars)-1 {
				if s, ok := typ.(*Slice); ok {
					buf.WriteString("...")
					typ = s.elem
				} else {
					// special case:
					// append(s, "foo"...) leads to signature func([]byte, string...)
					if t := asBasic(typ); t == nil || t.kind != String {
						panic("expected string type")
					}
					writeType(buf, typ, qf, visited)
					buf.WriteString("...")
					continue
				}
			}
			writeType(buf, typ, qf, visited)
		}
	}
	buf.WriteByte(')')
}

// WriteSignature writes the representation of the signature sig to buf,
// without a leading "func" keyword.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func WriteSignature(buf *bytes.Buffer, sig *Signature, qf Qualifier) {
	writeSignature(buf, sig, qf, make([]Type, 0, 8))
}

func writeSignature(buf *bytes.Buffer, sig *Signature, qf Qualifier, visited []Type) {
	if sig.TParams().Len() != 0 {
		writeTParamList(buf, sig.TParams().list(), qf, visited)
	}

	writeTuple(buf, sig.params, sig.variadic, qf, visited)

	n := sig.results.Len()
	if n == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if n == 1 && sig.results.vars[0].name == "" {
		// single unnamed result
		writeType(buf, sig.results.vars[0].typ, qf, visited)
		return
	}

	// multiple or named result(s)
	writeTuple(buf, sig.results, false, qf, visited)
}

// subscript returns the decimal (utf8) representation of x using subscript digits.
func subscript(x uint64) string {
	const w = len("₀") // all digits 0...9 have the same utf8 width
	var buf [32 * w]byte
	i := len(buf)
	for {
		i -= w
		utf8.EncodeRune(buf[i:], '₀'+rune(x%10)) // '₀' == U+2080
		x /= 10
		if x == 0 {
			break
		}
	}
	return string(buf[i:])
}
