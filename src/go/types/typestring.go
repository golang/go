// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of types.

package types

import (
	"bytes"
	"fmt"
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

// RelativeTo(pkg) returns a Qualifier that fully qualifies members of
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

func writeType(buf *bytes.Buffer, typ Type, qf Qualifier, visited []Type) {
	// Theoretically, this is a quadratic lookup algorithm, but in
	// practice deeply nested composite types with unnamed component
	// types are uncommon. This code is likely more efficient than
	// using a map.
	for _, t := range visited {
		if t == typ {
			fmt.Fprintf(buf, "○%T", typ) // cycle to typ
			return
		}
	}
	visited = append(visited, typ)

	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		if t.kind == UnsafePointer {
			buf.WriteString("unsafe.")
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
			for i, typ := range t.embeddeds {
				if i > 0 || len(t.methods) > 0 {
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
		s := "<Named w/o object>"
		if obj := t.obj; obj != nil {
			if obj.pkg != nil {
				writePackage(buf, obj.pkg, qf)
			}
			// TODO(gri): function-local named types should be displayed
			// differently from named types at package level to avoid
			// ambiguity.
			s = obj.name
		}
		buf.WriteString(s)

	default:
		// For externally defined implementations of Type.
		buf.WriteString(t.String())
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
					if t, ok := typ.Underlying().(*Basic); !ok || t.kind != String {
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
