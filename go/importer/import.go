// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This implementation is loosely based on the algorithm described
// in: "On the linearization of graphs and writing symbol files",
// by R. Griesemer, Technical Report 156, ETH Zürich, 1991.

// package importer implements an exporter and importer for Go export data.
package importer // import "golang.org/x/tools/go/importer"

import (
	"encoding/binary"
	"fmt"
	"go/token"

	"golang.org/x/tools/go/exact"
	"golang.org/x/tools/go/types"
)

// ImportData imports a package from the serialized package data
// and returns the number of bytes consumed and a reference to the package.
// If data is obviously malformed, an error is returned but in
// general it is not recommended to call ImportData on untrusted
// data.
func ImportData(imports map[string]*types.Package, data []byte) (int, *types.Package, error) {
	datalen := len(data)

	// check magic string
	var s string
	if len(data) >= len(magic) {
		s = string(data[:len(magic)])
		data = data[len(magic):]
	}
	if s != magic {
		return 0, nil, fmt.Errorf("incorrect magic string: got %q; want %q", s, magic)
	}

	// check low-level encoding format
	var m byte = 'm' // missing format
	if len(data) > 0 {
		m = data[0]
		data = data[1:]
	}
	if m != format() {
		return 0, nil, fmt.Errorf("incorrect low-level encoding format: got %c; want %c", m, format())
	}

	p := importer{
		data:    data,
		datalen: datalen,
		imports: imports,
	}

	// populate typList with predeclared types
	for _, t := range predeclared {
		p.typList = append(p.typList, t)
	}

	if v := p.string(); v != version {
		return 0, nil, fmt.Errorf("unknown version: got %s; want %s", v, version)
	}

	pkg := p.pkg()
	if debug && p.pkgList[0] != pkg {
		panic("imported packaged not found in pkgList[0]")
	}

	// read objects
	n := p.int()
	for i := 0; i < n; i++ {
		p.obj(pkg)
	}

	// complete interfaces
	for _, typ := range p.typList {
		if it, ok := typ.(*types.Interface); ok {
			it.Complete()
		}
	}

	// package was imported completely and without errors
	pkg.MarkComplete()

	return p.consumed(), pkg, nil
}

type importer struct {
	data    []byte
	datalen int
	imports map[string]*types.Package
	pkgList []*types.Package
	typList []types.Type
}

func (p *importer) pkg() *types.Package {
	// if the package was seen before, i is its index (>= 0)
	i := p.int()
	if i >= 0 {
		return p.pkgList[i]
	}

	// otherwise, i is the package tag (< 0)
	if i != packageTag {
		panic(fmt.Sprintf("unexpected package tag %d", i))
	}

	// read package data
	name := p.string()
	path := p.string()

	// if the package was imported before, use that one; otherwise create a new one
	pkg := p.imports[path]
	if pkg == nil {
		pkg = types.NewPackage(path, name)
		p.imports[path] = pkg
	}
	p.pkgList = append(p.pkgList, pkg)

	return pkg
}

func (p *importer) obj(pkg *types.Package) {
	var obj types.Object
	switch tag := p.int(); tag {
	case constTag:
		obj = types.NewConst(token.NoPos, pkg, p.string(), p.typ(), p.value())
	case typeTag:
		// type object is added to scope via respective named type
		_ = p.typ().(*types.Named)
		return
	case varTag:
		obj = types.NewVar(token.NoPos, pkg, p.string(), p.typ())
	case funcTag:
		obj = types.NewFunc(token.NoPos, pkg, p.string(), p.typ().(*types.Signature))
	default:
		panic(fmt.Sprintf("unexpected object tag %d", tag))
	}

	if alt := pkg.Scope().Insert(obj); alt != nil {
		panic(fmt.Sprintf("%s already declared", alt.Name()))
	}
}

func (p *importer) value() exact.Value {
	switch kind := exact.Kind(p.int()); kind {
	case falseTag:
		return exact.MakeBool(false)
	case trueTag:
		return exact.MakeBool(true)
	case int64Tag:
		return exact.MakeInt64(p.int64())
	case floatTag:
		return p.float()
	case fractionTag:
		return p.fraction()
	case complexTag:
		re := p.fraction()
		im := p.fraction()
		return exact.BinaryOp(re, token.ADD, exact.MakeImag(im))
	case stringTag:
		return exact.MakeString(p.string())
	default:
		panic(fmt.Sprintf("unexpected value kind %d", kind))
	}
}

func (p *importer) float() exact.Value {
	sign := p.int()
	if sign == 0 {
		return exact.MakeInt64(0)
	}

	x := p.ufloat()
	if sign < 0 {
		x = exact.UnaryOp(token.SUB, x, 0)
	}
	return x
}

func (p *importer) fraction() exact.Value {
	sign := p.int()
	if sign == 0 {
		return exact.MakeInt64(0)
	}

	x := exact.BinaryOp(p.ufloat(), token.QUO, p.ufloat())
	if sign < 0 {
		x = exact.UnaryOp(token.SUB, x, 0)
	}
	return x
}

func (p *importer) ufloat() exact.Value {
	exp := p.int()
	x := exact.MakeFromBytes(p.bytes())
	switch {
	case exp < 0:
		d := exact.Shift(exact.MakeInt64(1), token.SHL, uint(-exp))
		x = exact.BinaryOp(x, token.QUO, d)
	case exp > 0:
		x = exact.Shift(x, token.SHL, uint(exp))
	}
	return x
}

func (p *importer) record(t types.Type) {
	p.typList = append(p.typList, t)
}

func (p *importer) typ() types.Type {
	// if the type was seen before, i is its index (>= 0)
	i := p.int()
	if i >= 0 {
		return p.typList[i]
	}

	// otherwise, i is the type tag (< 0)
	switch i {
	case arrayTag:
		t := new(types.Array)
		p.record(t)

		n := p.int64()
		*t = *types.NewArray(p.typ(), n)
		return t

	case sliceTag:
		t := new(types.Slice)
		p.record(t)

		*t = *types.NewSlice(p.typ())
		return t

	case structTag:
		t := new(types.Struct)
		p.record(t)

		n := p.int()
		fields := make([]*types.Var, n)
		tags := make([]string, n)
		for i := range fields {
			fields[i] = p.field()
			tags[i] = p.string()
		}
		*t = *types.NewStruct(fields, tags)
		return t

	case pointerTag:
		t := new(types.Pointer)
		p.record(t)

		*t = *types.NewPointer(p.typ())
		return t

	case signatureTag:
		t := new(types.Signature)
		p.record(t)

		*t = *p.signature()
		return t

	case interfaceTag:
		// Create a dummy entry in the type list. This is safe because we
		// cannot expect the interface type to appear in a cycle, as any
		// such cycle must contain a named type which would have been
		// first defined earlier.
		n := len(p.typList)
		p.record(nil)

		// read embedded interfaces
		embeddeds := make([]*types.Named, p.int())
		for i := range embeddeds {
			embeddeds[i] = p.typ().(*types.Named)
		}

		// read methods
		methods := make([]*types.Func, p.int())
		for i := range methods {
			pkg, name := p.qualifiedName()
			methods[i] = types.NewFunc(token.NoPos, pkg, name, p.typ().(*types.Signature))
		}

		t := types.NewInterface(methods, embeddeds)
		p.typList[n] = t
		return t

	case mapTag:
		t := new(types.Map)
		p.record(t)

		*t = *types.NewMap(p.typ(), p.typ())
		return t

	case chanTag:
		t := new(types.Chan)
		p.record(t)

		*t = *types.NewChan(types.ChanDir(p.int()), p.typ())
		return t

	case namedTag:
		// read type object
		name := p.string()
		pkg := p.pkg()
		scope := pkg.Scope()
		obj := scope.Lookup(name)

		// if the object doesn't exist yet, create and insert it
		if obj == nil {
			obj = types.NewTypeName(token.NoPos, pkg, name, nil)
			scope.Insert(obj)
		}

		// associate new named type with obj if it doesn't exist yet
		t0 := types.NewNamed(obj.(*types.TypeName), nil, nil)

		// but record the existing type, if any
		t := obj.Type().(*types.Named)
		p.record(t)

		// read underlying type
		t0.SetUnderlying(p.typ())

		// read associated methods
		for i, n := 0, p.int(); i < n; i++ {
			t0.AddMethod(types.NewFunc(token.NoPos, pkg, p.string(), p.typ().(*types.Signature)))
		}

		return t

	default:
		panic(fmt.Sprintf("unexpected type tag %d", i))
	}
}

func deref(typ types.Type) types.Type {
	if p, _ := typ.(*types.Pointer); p != nil {
		return p.Elem()
	}
	return typ
}

func (p *importer) field() *types.Var {
	pkg, name := p.qualifiedName()
	typ := p.typ()

	anonymous := false
	if name == "" {
		// anonymous field - typ must be T or *T and T must be a type name
		switch typ := deref(typ).(type) {
		case *types.Basic: // basic types are named types
			pkg = nil
			name = typ.Name()
		case *types.Named:
			obj := typ.Obj()
			name = obj.Name()
			// correct the field package for anonymous fields
			if exported(name) {
				pkg = p.pkgList[0]
			}
		default:
			panic("anonymous field expected")
		}
		anonymous = true
	}

	return types.NewField(token.NoPos, pkg, name, typ, anonymous)
}

func (p *importer) qualifiedName() (*types.Package, string) {
	name := p.string()
	pkg := p.pkgList[0] // exported names assume current package
	if !exported(name) {
		pkg = p.pkg()
	}
	return pkg, name
}

func (p *importer) signature() *types.Signature {
	var recv *types.Var
	if p.int() != 0 {
		recv = p.param()
	}
	return types.NewSignature(recv, p.tuple(), p.tuple(), p.int() != 0)
}

func (p *importer) param() *types.Var {
	return types.NewVar(token.NoPos, nil, p.string(), p.typ())
}

func (p *importer) tuple() *types.Tuple {
	vars := make([]*types.Var, p.int())
	for i := range vars {
		vars[i] = p.param()
	}
	return types.NewTuple(vars...)
}

// ----------------------------------------------------------------------------
// decoders

func (p *importer) string() string {
	return string(p.bytes())
}

func (p *importer) int() int {
	return int(p.int64())
}

func (p *importer) int64() int64 {
	if debug {
		p.marker('i')
	}

	return p.rawInt64()
}

// Note: bytes() returns the respective byte slice w/o copy.
func (p *importer) bytes() []byte {
	if debug {
		p.marker('b')
	}

	var b []byte
	if n := int(p.rawInt64()); n > 0 {
		b = p.data[:n]
		p.data = p.data[n:]
	}
	return b
}

func (p *importer) marker(want byte) {
	if debug {
		if got := p.data[0]; got != want {
			panic(fmt.Sprintf("incorrect marker: got %c; want %c (pos = %d)", got, want, p.consumed()))
		}
		p.data = p.data[1:]

		pos := p.consumed()
		if n := int(p.rawInt64()); n != pos {
			panic(fmt.Sprintf("incorrect position: got %d; want %d", n, pos))
		}
	}
}

// rawInt64 should only be used by low-level decoders
func (p *importer) rawInt64() int64 {
	i, n := binary.Varint(p.data)
	p.data = p.data[n:]
	return i
}

func (p *importer) consumed() int {
	return p.datalen - len(p.data)
}
