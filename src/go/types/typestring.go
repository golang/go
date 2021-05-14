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

// If gcCompatibilityMode is set, printing of types is modified
// to match the representation of some types in the gc compiler:
//
//	- byte and rune lose their alias name and simply stand for
//	  uint8 and int32 respectively
//	- embedded interfaces get flattened (the embedding info is lost,
//	  and certain recursive interface types cannot be printed anymore)
//
// This makes it easier to compare packages computed with the type-
// checker vs packages imported from gc export data.
//
// Caution: This flag affects all uses of WriteType, globally.
// It is only provided for testing in conjunction with
// gc-generated data.
//
// This flag is exported in the x/tools/go/types package. We don't
// need it at the moment in the std repo and so we don't export it
// anymore. We should eventually try to remove it altogether.
// TODO(gri) remove this
var gcCompatibilityMode bool

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

		if gcCompatibilityMode {
			// forget the alias names
			switch t.kind {
			case Byte:
				t = Typ[Uint8]
			case Rune:
				t = Typ[Int32]
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

	case *_Sum:
		for i, t := range t.types {
			if i > 0 {
				buf.WriteString(", ")
			}
			writeType(buf, t, qf, visited)
		}

	case *Interface:
		// We write the source-level methods and embedded types rather
		// than the actual method set since resolved method signatures
		// may have non-printable cycles if parameters have embedded
		// interface types that (directly or indirectly) embed the
		// current interface. For instance, consider the result type
		// of m:
		//
		//     type T interface{
		//         m() interface{ T }
		//     }
		//
		buf.WriteString("interface{")
		empty := true
		if gcCompatibilityMode {
			// print flattened interface
			// (useful to compare against gc-generated interfaces)
			for i, m := range t.allMethods {
				if i > 0 {
					buf.WriteString("; ")
				}
				buf.WriteString(m.name)
				writeSignature(buf, m.typ.(*Signature), qf, visited)
				empty = false
			}
			if !empty && t.allTypes != nil {
				buf.WriteString("; ")
			}
			if t.allTypes != nil {
				buf.WriteString("type ")
				writeType(buf, t.allTypes, qf, visited)
			}
		} else {
			// print explicit interface methods and embedded types
			for i, m := range t.methods {
				if i > 0 {
					buf.WriteString("; ")
				}
				buf.WriteString(m.name)
				writeSignature(buf, m.typ.(*Signature), qf, visited)
				empty = false
			}
			if !empty && t.types != nil {
				buf.WriteString("; ")
			}
			if t.types != nil {
				buf.WriteString("type ")
				writeType(buf, t.types, qf, visited)
				empty = false
			}
			if !empty && len(t.embeddeds) > 0 {
				buf.WriteString("; ")
			}
			for i, typ := range t.embeddeds {
				if i > 0 {
					buf.WriteString("; ")
				}
				writeType(buf, typ, qf, visited)
				empty = false
			}
		}
		if t.allMethods == nil || len(t.methods) > len(t.allMethods) {
			if !empty {
				buf.WriteByte(' ')
			}
			buf.WriteString("/* incomplete */")
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
			panic("unreachable")
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
		writeTypeName(buf, t.obj, qf)
		if t.targs != nil {
			// instantiated type
			buf.WriteByte('[')
			writeTypeList(buf, t.targs, qf, visited)
			buf.WriteByte(']')
		} else if t.tparams != nil {
			// parameterized type
			writeTParamList(buf, t.tparams, qf, visited)
		}

	case *_TypeParam:
		s := "?"
		if t.obj != nil {
			s = t.obj.name
		}
		buf.WriteString(s + subscript(t.id))

	case *instance:
		buf.WriteByte(instanceMarker) // indicate "non-evaluated" syntactic instance
		writeTypeName(buf, t.base.obj, qf)
		buf.WriteByte('[')
		writeTypeList(buf, t.targs, qf, visited)
		buf.WriteByte(']')

	case *bottom:
		buf.WriteString("⊥")

	case *top:
		buf.WriteString("⊤")

	default:
		// For externally defined implementations of Type.
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

func writeTParamList(buf *bytes.Buffer, list []*TypeName, qf Qualifier, visited []Type) {
	// TODO(rFindley) compare this with the corresponding implementation in types2
	buf.WriteString("[")
	var prev Type
	for i, p := range list {
		// TODO(rFindley) support 'any' sugar here.
		var b Type = &emptyInterface
		if t, _ := p.typ.(*_TypeParam); t != nil && t.bound != nil {
			b = t.bound
		}
		if i > 0 {
			if b != prev {
				// type bound changed - write previous one before advancing
				buf.WriteByte(' ')
				writeType(buf, prev, qf, visited)
			}
			buf.WriteString(", ")
		}
		prev = b

		if t, _ := p.typ.(*_TypeParam); t != nil {
			writeType(buf, t, qf, visited)
		} else {
			buf.WriteString(p.name)
		}
	}
	if prev != nil {
		buf.WriteByte(' ')
		writeType(buf, prev, qf, visited)
	}
	buf.WriteByte(']')
}

func writeTypeName(buf *bytes.Buffer, obj *TypeName, qf Qualifier) {
	s := "<Named w/o object>"
	if obj != nil {
		if obj.pkg != nil {
			writePackage(buf, obj.pkg, qf)
		}
		// TODO(gri): function-local named types should be displayed
		// differently from named types at package level to avoid
		// ambiguity.
		s = obj.name
	}
	buf.WriteString(s)
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
						panic("internal error: string type expected")
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
	if sig.tparams != nil {
		writeTParamList(buf, sig.tparams, qf, visited)
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
