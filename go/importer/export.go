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
	version = "v0"
)

// Object and type tags. Must be < 0.
const (
	// Packages
	packageTag = -(iota + 1)

	// Objects
	constTag
	typeTag
	varTag
	funcTag

	// Types
	basicTag
	arrayTag
	sliceTag
	structTag
	pointerTag
	signatureTag
	interfaceTag
	mapTag
	chanTag
	namedTag

	// Values
	intTag
	floatTag
	fractionTag
	complexTag
	stringTag
	falseTag
	trueTag
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

	p.string(version)

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
	p.int(packageTag)
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
		p.int(constTag)
		p.string(obj.Name())
		p.typ(obj.Type())
		p.value(obj.Val())
	case *types.TypeName:
		p.int(typeTag)
		// name is written by corresponding named type
		p.typ(obj.Type().(*types.Named))
	case *types.Var:
		p.int(varTag)
		p.string(obj.Name())
		p.typ(obj.Type())
	case *types.Func:
		p.int(funcTag)
		p.string(obj.Name())
		p.typ(obj.Type())
	default:
		panic(fmt.Sprintf("unexpected object type %T", obj))
	}
}

func (p *exporter) value(x exact.Value) {
	if trace {
		p.tracef("value { ")
		defer p.tracef("} ")
	}

	switch kind := x.Kind(); kind {
	case exact.Bool:
		tag := falseTag
		if exact.BoolVal(x) {
			tag = trueTag
		}
		p.int(tag)
	case exact.String:
		p.int(stringTag)
		p.string(exact.StringVal(x))
	case exact.Int:
		if i, ok := exact.Int64Val(x); ok {
			p.int(intTag)
			p.int64(i)
			return
		}
		p.int(floatTag)
		p.float(x)
	case exact.Float:
		p.int(fractionTag)
		p.fraction(x)
	case exact.Complex:
		p.int(complexTag)
		p.fraction(exact.Real(x))
		p.fraction(exact.Imag(x))
	default:
		panic(fmt.Sprintf("unexpected value kind %d", kind))
	}
}

func (p *exporter) float(x exact.Value) {
	sign := exact.Sign(x)
	p.int(sign)
	if sign == 0 {
		return
	}

	p.ufloat(x)
}

func (p *exporter) fraction(x exact.Value) {
	sign := exact.Sign(x)
	p.int(sign)
	if sign == 0 {
		return
	}

	p.ufloat(exact.Num(x))
	p.ufloat(exact.Denom(x))
}

func (p *exporter) ufloat(x exact.Value) {
	mant := exact.Bytes(x)
	exp8 := -1
	for i, b := range mant {
		if b != 0 {
			exp8 = i
			break
		}
	}
	if exp8 < 0 {
		panic(fmt.Sprintf("%s has no mantissa", x))
	}
	p.int(exp8 * 8)
	p.bytes(mant[exp8:])
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
		p.int(basicTag)
		p.string(t.Name())

	case *types.Array:
		p.int(arrayTag)
		p.int64(t.Len())
		p.typ(t.Elem())

	case *types.Slice:
		p.int(sliceTag)
		p.typ(t.Elem())

	case *types.Struct:
		p.int(structTag)
		n := t.NumFields()
		p.int(n)
		for i := 0; i < n; i++ {
			p.field(t.Field(i))
			p.string(t.Tag(i))
		}

	case *types.Pointer:
		p.int(pointerTag)
		p.typ(t.Elem())

	case *types.Signature:
		p.int(signatureTag)
		p.signature(t)

	case *types.Interface:
		p.int(interfaceTag)

		// write methods
		n := t.NumExplicitMethods()
		p.int(n)
		for i := 0; i < n; i++ {
			m := t.ExplicitMethod(i)
			p.qualifiedName(m.Pkg(), m.Name())
			p.typ(m.Type())
		}

		// write embedded interfaces
		m := t.NumEmbeddeds()
		p.int(m)
		for i := 0; i < m; i++ {
			p.typ(t.Embedded(i))
		}

	case *types.Map:
		p.int(mapTag)
		p.typ(t.Key())
		p.typ(t.Elem())

	case *types.Chan:
		p.int(chanTag)
		p.int(int(t.Dir()))
		p.typ(t.Elem())

	case *types.Named:
		p.int(namedTag)

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
			p.typ(m.Type())
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
	// exported names don't need package
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
