// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the TypeString function.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
)

// typeString returns a string representation for typ.
func typeString(typ Type) string {
	var buf bytes.Buffer
	writeType(&buf, typ)
	return buf.String()
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
		writeType(buf, par.Type.(Type))
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, sig *Signature) {
	writeParams(buf, sig.Params, sig.IsVariadic)
	if len(sig.Results) == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if len(sig.Results) == 1 && sig.Results[0].Name == "" {
		// single unnamed result
		writeType(buf, sig.Results[0].Type.(Type))
		return
	}

	// multiple or named result(s)
	writeParams(buf, sig.Results, false)
}

func writeType(buf *bytes.Buffer, typ Type) {
	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		buf.WriteString(t.Name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.Len)
		writeType(buf, t.Elt)

	case *Slice:
		buf.WriteString("[]")
		writeType(buf, t.Elt)

	case *Struct:
		buf.WriteString("struct{")
		for i, f := range t.Fields {
			if i > 0 {
				buf.WriteString("; ")
			}
			if !f.IsAnonymous {
				buf.WriteString(f.Name)
				buf.WriteByte(' ')
			}
			writeType(buf, f.Type)
			if f.Tag != "" {
				fmt.Fprintf(buf, " %q", f.Tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		writeType(buf, t.Base)

	case *tuple:
		buf.WriteByte('(')
		for i, typ := range t.list {
			if i > 0 {
				buf.WriteString("; ")
			}
			writeType(buf, typ)
		}
		buf.WriteByte(')')

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, t)

	case *builtin:
		fmt.Fprintf(buf, "<type of %s>", t.name)

	case *Interface:
		buf.WriteString("interface{")
		for i, m := range t.Methods {
			if i > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(m.Name)
			writeSignature(buf, m.Type.(*Signature))
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		writeType(buf, t.Key)
		buf.WriteByte(']')
		writeType(buf, t.Elt)

	case *Chan:
		var s string
		switch t.Dir {
		case ast.SEND:
			s = "chan<- "
		case ast.RECV:
			s = "<-chan "
		default:
			s = "chan "
		}
		buf.WriteString(s)
		writeType(buf, t.Elt)

	case *NamedType:
		buf.WriteString(t.Obj.Name)

	default:
		fmt.Fprintf(buf, "<type %T>", t)
	}
}
