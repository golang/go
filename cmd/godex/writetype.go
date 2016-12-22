// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements writing of types. The functionality is lifted
// directly from go/types, but now contains various modifications for
// nicer output.
//
// TODO(gri) back-port once we have a fixed interface and once the
// go/types API is not frozen anymore for the 1.3 release; and remove
// this implementation if possible.

package main

import "go/types"

func (p *printer) writeType(this *types.Package, typ types.Type) {
	p.writeTypeInternal(this, typ, make([]types.Type, 8))
}

// From go/types - leave for now to ease back-porting this code.
const GcCompatibilityMode = false

func (p *printer) writeTypeInternal(this *types.Package, typ types.Type, visited []types.Type) {
	// Theoretically, this is a quadratic lookup algorithm, but in
	// practice deeply nested composite types with unnamed component
	// types are uncommon. This code is likely more efficient than
	// using a map.
	for _, t := range visited {
		if t == typ {
			p.printf("○%T", typ) // cycle to typ
			return
		}
	}
	visited = append(visited, typ)

	switch t := typ.(type) {
	case nil:
		p.print("<nil>")

	case *types.Basic:
		if t.Kind() == types.UnsafePointer {
			p.print("unsafe.")
		}
		if GcCompatibilityMode {
			// forget the alias names
			switch t.Kind() {
			case types.Byte:
				t = types.Typ[types.Uint8]
			case types.Rune:
				t = types.Typ[types.Int32]
			}
		}
		p.print(t.Name())

	case *types.Array:
		p.printf("[%d]", t.Len())
		p.writeTypeInternal(this, t.Elem(), visited)

	case *types.Slice:
		p.print("[]")
		p.writeTypeInternal(this, t.Elem(), visited)

	case *types.Struct:
		n := t.NumFields()
		if n == 0 {
			p.print("struct{}")
			return
		}

		p.print("struct {\n")
		p.indent++
		for i := 0; i < n; i++ {
			f := t.Field(i)
			if !f.Anonymous() {
				p.printf("%s ", f.Name())
			}
			p.writeTypeInternal(this, f.Type(), visited)
			if tag := t.Tag(i); tag != "" {
				p.printf(" %q", tag)
			}
			p.print("\n")
		}
		p.indent--
		p.print("}")

	case *types.Pointer:
		p.print("*")
		p.writeTypeInternal(this, t.Elem(), visited)

	case *types.Tuple:
		p.writeTuple(this, t, false, visited)

	case *types.Signature:
		p.print("func")
		p.writeSignatureInternal(this, t, visited)

	case *types.Interface:
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
		n := t.NumMethods()
		if n == 0 {
			p.print("interface{}")
			return
		}

		p.print("interface {\n")
		p.indent++
		if GcCompatibilityMode {
			// print flattened interface
			// (useful to compare against gc-generated interfaces)
			for i := 0; i < n; i++ {
				m := t.Method(i)
				p.print(m.Name())
				p.writeSignatureInternal(this, m.Type().(*types.Signature), visited)
				p.print("\n")
			}
		} else {
			// print explicit interface methods and embedded types
			for i, n := 0, t.NumExplicitMethods(); i < n; i++ {
				m := t.ExplicitMethod(i)
				p.print(m.Name())
				p.writeSignatureInternal(this, m.Type().(*types.Signature), visited)
				p.print("\n")
			}
			for i, n := 0, t.NumEmbeddeds(); i < n; i++ {
				typ := t.Embedded(i)
				p.writeTypeInternal(this, typ, visited)
				p.print("\n")
			}
		}
		p.indent--
		p.print("}")

	case *types.Map:
		p.print("map[")
		p.writeTypeInternal(this, t.Key(), visited)
		p.print("]")
		p.writeTypeInternal(this, t.Elem(), visited)

	case *types.Chan:
		var s string
		var parens bool
		switch t.Dir() {
		case types.SendRecv:
			s = "chan "
			// chan (<-chan T) requires parentheses
			if c, _ := t.Elem().(*types.Chan); c != nil && c.Dir() == types.RecvOnly {
				parens = true
			}
		case types.SendOnly:
			s = "chan<- "
		case types.RecvOnly:
			s = "<-chan "
		default:
			panic("unreachable")
		}
		p.print(s)
		if parens {
			p.print("(")
		}
		p.writeTypeInternal(this, t.Elem(), visited)
		if parens {
			p.print(")")
		}

	case *types.Named:
		s := "<Named w/o object>"
		if obj := t.Obj(); obj != nil {
			if pkg := obj.Pkg(); pkg != nil {
				if pkg != this {
					p.print(pkg.Path())
					p.print(".")
				}
				// TODO(gri): function-local named types should be displayed
				// differently from named types at package level to avoid
				// ambiguity.
			}
			s = obj.Name()
		}
		p.print(s)

	default:
		// For externally defined implementations of Type.
		p.print(t.String())
	}
}

func (p *printer) writeTuple(this *types.Package, tup *types.Tuple, variadic bool, visited []types.Type) {
	p.print("(")
	for i, n := 0, tup.Len(); i < n; i++ {
		if i > 0 {
			p.print(", ")
		}
		v := tup.At(i)
		if name := v.Name(); name != "" {
			p.print(name)
			p.print(" ")
		}
		typ := v.Type()
		if variadic && i == n-1 {
			p.print("...")
			typ = typ.(*types.Slice).Elem()
		}
		p.writeTypeInternal(this, typ, visited)
	}
	p.print(")")
}

func (p *printer) writeSignature(this *types.Package, sig *types.Signature) {
	p.writeSignatureInternal(this, sig, make([]types.Type, 8))
}

func (p *printer) writeSignatureInternal(this *types.Package, sig *types.Signature, visited []types.Type) {
	p.writeTuple(this, sig.Params(), sig.Variadic(), visited)

	res := sig.Results()
	n := res.Len()
	if n == 0 {
		// no result
		return
	}

	p.print(" ")
	if n == 1 && res.At(0).Name() == "" {
		// single unnamed result
		p.writeTypeInternal(this, res.At(0).Type(), visited)
		return
	}

	// multiple or named result(s)
	p.writeTuple(this, res, false, visited)
}
