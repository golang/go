// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of types.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
)

// TypeString returns the string representation of typ.
// Named types are printed package-qualified if they
// do not belong to this package.
func TypeString(this *Package, typ Type) string {
	var buf bytes.Buffer
	WriteType(&buf, this, typ)
	return buf.String()
}

// WriteType writes the string representation of typ to buf.
// Named types are printed package-qualified if they
// do not belong to this package.
func WriteType(buf *bytes.Buffer, this *Package, typ Type) {
	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		if t.kind == UnsafePointer {
			buf.WriteString("unsafe.")
		}
		buf.WriteString(t.name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.len)
		WriteType(buf, this, t.elem)

	case *Slice:
		buf.WriteString("[]")
		WriteType(buf, this, t.elem)

	case *Struct:
		buf.WriteString("struct{")
		for i, f := range t.fields {
			if i > 0 {
				buf.WriteString("; ")
			}
			if !f.anonymous {
				buf.WriteString(f.name)
				buf.WriteByte(' ')
			}
			WriteType(buf, this, f.typ)
			if tag := t.Tag(i); tag != "" {
				fmt.Fprintf(buf, " %q", tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		WriteType(buf, this, t.base)

	case *Tuple:
		writeTuple(buf, this, t, false)

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, this, t)

	case *Interface:
		// We write the source-level methods and embedded types rather
		// than the actual method set since resolved method signatures
		// may have non-printable cycles if parameters have anonymous
		// interface types that (directly or indirectly) embed the
		// current interface. For instance, consider the result type
		// of m:
		//
		//     type T interface{
		//         m() interface{ T }
		//     }
		//
		buf.WriteString("interface{")
		for i, m := range t.methods {
			if i > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(m.name)
			writeSignature(buf, this, m.typ.(*Signature))
		}
		for i, typ := range t.types {
			if i > 0 || len(t.methods) > 0 {
				buf.WriteString("; ")
			}
			WriteType(buf, this, typ)
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		WriteType(buf, this, t.key)
		buf.WriteByte(']')
		WriteType(buf, this, t.elem)

	case *Chan:
		var s string
		var parens bool
		switch t.dir {
		case ast.SEND:
			s = "chan<- "
		case ast.RECV:
			s = "<-chan "
		default:
			s = "chan "
			// chan (<-chan T) requires parentheses
			if c, _ := t.elem.(*Chan); c != nil && c.dir == ast.RECV {
				parens = true
			}
		}
		buf.WriteString(s)
		if parens {
			buf.WriteByte('(')
		}
		WriteType(buf, this, t.elem)
		if parens {
			buf.WriteByte(')')
		}

	case *Named:
		s := "<Named w/o object>"
		if obj := t.obj; obj != nil {
			if obj.pkg != nil {
				if obj.pkg != this {
					buf.WriteString(obj.pkg.path)
					buf.WriteByte('.')
				}
				// TODO(gri): function-local named types should be displayed
				// differently from named types at package level to avoid
				// ambiguity.
			}
			s = t.obj.name
		}
		buf.WriteString(s)

	default:
		// For externally defined implementations of Type.
		buf.WriteString(t.String())
	}
}

func writeTuple(buf *bytes.Buffer, this *Package, tup *Tuple, isVariadic bool) {
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
			if isVariadic && i == len(tup.vars)-1 {
				buf.WriteString("...")
				typ = typ.(*Slice).elem
			}
			WriteType(buf, this, typ)
		}
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, this *Package, sig *Signature) {
	writeTuple(buf, this, sig.params, sig.isVariadic)

	n := sig.results.Len()
	if n == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if n == 1 && sig.results.vars[0].name == "" {
		// single unnamed result
		WriteType(buf, this, sig.results.vars[0].typ)
		return
	}

	// multiple or named result(s)
	writeTuple(buf, this, sig.results, false)
}
