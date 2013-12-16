// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/ast"
	"strings"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// debugging support
const (
	debug = false // emit debugging data
	trace = false // print emitted data
)

const (
	magic   = "\n$$ exports $$\n"
	version = 0
)

// Object and type tags. Must be < 0.
const (
	// Objects
	_Package = -(iota + 1)
	_Const
	_Type
	_Var
	_Func

	// Types
	_Basic
	_Array
	_Slice
	_Struct
	_Pointer
	_Tuple
	_Signature
	_Interface
	_Map
	_Chan
	_Named
)

// ExportData serializes the interface (exported package objects)
// of package pkg and returns the corresponding data. The export
// format is described elsewhere (TODO).
func ExportData(pkg *types.Package) []byte {
	p := exporter{
		data: []byte(magic),
		// TODO(gri) If we can't have nil packages
		// or types, remove nil entries at index 0.
		pkgIndex: map[*types.Package]int{nil: 0},
		typIndex: map[types.Type]int{nil: 0},
	}

	// populate typIndex with predeclared types
	for _, t := range types.Typ[1:] {
		p.typIndex[t] = len(p.typIndex)
	}
	p.typIndex[types.Universe.Lookup("error").Type()] = len(p.typIndex)

	if trace {
		p.tracef("export %s\n", pkg.Name())
		defer p.tracef("\n")
	}

	p.int(version)

	p.pkg(pkg)

	// collect exported objects from package scope
	var list []types.Object
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if exported(name) {
			list = append(list, scope.Lookup(name))
		}
	}

	// write objects
	p.int(len(list))
	for _, obj := range list {
		p.obj(obj)
	}

	return p.data
}

type exporter struct {
	data     []byte
	pkgIndex map[*types.Package]int
	typIndex map[types.Type]int

	// tracing support
	indent string
}

func (p *exporter) pkg(pkg *types.Package) {
	if trace {
		p.tracef("package { ")
		defer p.tracef("} ")
	}

	// if the package was seen before, write its index (>= 0)
	if i, ok := p.pkgIndex[pkg]; ok {
		p.int(i)
		return
	}
	p.pkgIndex[pkg] = len(p.pkgIndex)

	// otherwise, write the package tag (< 0) and package data
	p.int(_Package)
	p.string(pkg.Name())
	p.string(pkg.Path())
}

func (p *exporter) obj(obj types.Object) {
	if trace {
		p.tracef("object %s {\n", obj.Name())
		defer p.tracef("}\n")
	}

	switch obj := obj.(type) {
	case *types.Const:
		p.int(_Const)
		p.string(obj.Name())
		p.typ(obj.Type())
		p.val(obj.Val())
	case *types.TypeName:
		p.int(_Type)
		// name is written by corresponding named type
		p.typ(obj.Type().(*types.Named))
	case *types.Var:
		p.int(_Var)
		p.string(obj.Name())
		p.typ(obj.Type())
	case *types.Func:
		p.int(_Func)
		p.string(obj.Name())
		p.signature(obj.Type().(*types.Signature))
	default:
		panic(fmt.Sprintf("unexpected object type %T", obj))
	}
}

func (p *exporter) val(x exact.Value) {
	if trace {
		p.tracef("value { ")
		defer p.tracef("} ")
	}

	kind := x.Kind()
	p.int(int(kind))
	switch kind {
	case exact.Bool:
		p.bool(exact.BoolVal(x))
	case exact.String:
		p.string(exact.StringVal(x))
	case exact.Int:
		p.intVal(x)
	case exact.Float:
		p.floatVal(x)
	case exact.Complex:
		p.floatVal(exact.Real(x))
		p.floatVal(exact.Imag(x))
	default:
		panic(fmt.Sprintf("unexpected value kind %d", kind))
	}
}

func (p *exporter) intVal(x exact.Value) {
	sign := exact.Sign(x)
	p.int(sign)
	if sign != 0 {
		p.bytes(exact.Bytes(x))
	}
}

func (p *exporter) floatVal(x exact.Value) {
	p.intVal(exact.Num(x))
	if exact.Sign(x) != 0 {
		// TODO(gri): For gc-generated constants, the denominator is
		// often a large power of two. Use a more compact representation.
		p.bytes(exact.Bytes(exact.Denom(x)))
	}
}

func (p *exporter) typ(typ types.Type) {
	if trace {
		p.tracef("type {\n")
		defer p.tracef("}\n")
	}

	// if the type was seen before, write its index (>= 0)
	if i, ok := p.typIndex[typ]; ok {
		p.int(i)
		return
	}
	p.typIndex[typ] = len(p.typIndex)

	// otherwise, write the type tag (< 0) and type data
	switch t := typ.(type) {
	case *types.Basic:
		// Basic types are pre-recorded and don't usually end up here.
		// However, the alias types byte and rune are not in the types.Typ
		// table and get emitted here (once per package, if they appear).
		// This permits faithful reconstruction of the alias type (i.e.,
		// keeping the name). If we decide to eliminate the distinction
		// between the alias types, this code can go.
		p.int(_Basic)
		p.string(t.Name())

	case *types.Array:
		p.int(_Array)
		p.typ(t.Elem())
		p.int64(t.Len())

	case *types.Slice:
		p.int(_Slice)
		p.typ(t.Elem())

	case *types.Struct:
		p.int(_Struct)
		n := t.NumFields()
		p.int(n)
		for i := 0; i < n; i++ {
			p.field(t.Field(i))
			p.string(t.Tag(i))
		}

	case *types.Pointer:
		p.int(_Pointer)
		p.typ(t.Elem())

	case *types.Signature:
		p.int(_Signature)
		p.signature(t)

	case *types.Interface:
		p.int(_Interface)
		n := t.NumMethods()
		p.int(n)
		for i := 0; i < n; i++ {
			m := t.Method(i)
			p.qualifiedName(m.Pkg(), m.Name())
			p.signature(m.Type().(*types.Signature))
		}

	case *types.Map:
		p.int(_Map)
		p.typ(t.Key())
		p.typ(t.Elem())

	case *types.Chan:
		p.int(_Chan)
		p.int(int(t.Dir()))
		p.typ(t.Elem())

	case *types.Named:
		p.int(_Named)

		// write type object
		obj := t.Obj()
		p.string(obj.Name())
		p.pkg(obj.Pkg())

		// write underlying type
		p.typ(t.Underlying())

		// write associated methods
		n := t.NumMethods()
		p.int(n)
		for i := 0; i < n; i++ {
			m := t.Method(i)
			p.string(m.Name())
			p.signature(m.Type().(*types.Signature))
		}

	default:
		panic("unreachable")
	}
}

func (p *exporter) field(f *types.Var) {
	// anonymous fields have "" name
	name := ""
	if !f.Anonymous() {
		name = f.Name()
	}

	p.qualifiedName(f.Pkg(), name)
	p.typ(f.Type())
}

func (p *exporter) qualifiedName(pkg *types.Package, name string) {
	p.string(name)
	// exported names don't write package
	if !exported(name) {
		if pkg == nil {
			panic(fmt.Sprintf("nil package for unexported qualified name %s", name))
		}
		p.pkg(pkg)
	}
}

func (p *exporter) signature(sig *types.Signature) {
	// TODO(gri) We only need to record the receiver type
	//           for interface methods if we flatten them
	//           out. If we track embedded types instead,
	//           the information is already present.
	if recv := sig.Recv(); recv != nil {
		p.bool(true)
		p.param(recv)
	} else {
		p.bool(false)
	}
	p.tuple(sig.Params())
	p.tuple(sig.Results())
	p.bool(sig.IsVariadic())
}

func (p *exporter) param(v *types.Var) {
	p.string(v.Name())
	p.typ(v.Type())
}

func (p *exporter) tuple(t *types.Tuple) {
	n := t.Len()
	p.int(n)
	for i := 0; i < n; i++ {
		p.param(t.At(i))
	}
}

// ----------------------------------------------------------------------------
// encoders

func (p *exporter) bool(b bool) {
	var x int64
	if b {
		x = 1
	}
	p.int64(x)
}

func (p *exporter) string(s string) {
	// TODO(gri) consider inlining this to avoid an extra allocation
	p.bytes([]byte(s))
}

func (p *exporter) int(x int) {
	p.int64(int64(x))
}

func (p *exporter) int64(x int64) {
	if debug {
		p.marker('i')
	}

	if trace {
		p.tracef("%d ", x)
	}

	p.rawInt64(x)
}

func (p *exporter) bytes(b []byte) {
	if debug {
		p.marker('b')
	}

	if trace {
		p.tracef("%q ", b)
	}

	p.rawInt64(int64(len(b)))
	if len(b) > 0 {
		p.data = append(p.data, b...)
	}
}

// marker emits a marker byte and position information which makes
// it easy for a reader to detect if it is "out of sync". Used in
// debug mode only.
func (p *exporter) marker(m byte) {
	if debug {
		p.data = append(p.data, m)
		p.rawInt64(int64(len(p.data)))
	}
}

// rawInt64 should only be used by low-level encoders
func (p *exporter) rawInt64(x int64) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], x)
	p.data = append(p.data, tmp[:n]...)
}

// utility functions

func (p *exporter) tracef(format string, args ...interface{}) {
	// rewrite format string to take care of indentation
	const indent = ".  "
	if strings.IndexAny(format, "{}\n") >= 0 {
		var buf bytes.Buffer
		for i := 0; i < len(format); i++ {
			// no need to deal with runes
			ch := format[i]
			switch ch {
			case '{':
				p.indent += indent
			case '}':
				p.indent = p.indent[:len(p.indent)-len(indent)]
				if i+1 < len(format) && format[i+1] == '\n' {
					buf.WriteByte('\n')
					buf.WriteString(p.indent)
					buf.WriteString("} ")
					i++
					continue
				}
			}
			buf.WriteByte(ch)
			if ch == '\n' {
				buf.WriteString(p.indent)
			}
		}
		format = buf.String()
	}
	fmt.Printf(format, args...)
}

func exported(name string) bool {
	return ast.IsExported(name)
}
