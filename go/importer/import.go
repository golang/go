// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This implementation is loosely based on the algorithm described
// in: "On the linearization of graphs and writing symbol files",
// by R. Griesemer, Technical Report 156, ETH Zürich, 1991.

// package importer implements an exporter and importer for Go export data.
package importer

import (
	"encoding/binary"
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// ImportData imports a package from the serialized package data.
// If data is obviously malformed, an error is returned but in
// general it is not recommended to call ImportData on untrusted
// data.
func ImportData(imports map[string]*types.Package, data []byte) (*types.Package, error) {
	// check magic string
	if n := len(magic); len(data) < n || string(data[:n]) != magic {
		return nil, fmt.Errorf("incorrect magic string: got %q; want %q", data[:n], magic)
	}

	p := importer{
		data:     data[len(magic):],
		imports:  imports,
		pkgList:  []*types.Package{nil},
		typList:  []types.Type{nil},
		consumed: len(magic), // for debugging only
	}

	// populate typList with predeclared types
	for _, t := range types.Typ[1:] {
		p.typList = append(p.typList, t)
	}
	p.typList = append(p.typList, types.Universe.Lookup("error").Type())

	if v := p.string(); v != version {
		return nil, fmt.Errorf("unknown version: got %d; want %d", v, version)
	}

	pkg := p.pkg()
	if debug && p.pkgList[1] != pkg {
		panic("imported packaged not found in pkgList[1]")
	}

	// read objects
	n := p.int()
	for i := 0; i < n; i++ {
		p.obj(pkg)
	}

	if len(p.data) > 0 {
		return nil, fmt.Errorf("not all input data consumed")
	}

	// package was imported completely and without errors
	pkg.MarkComplete()

	return pkg, nil
}

type importer struct {
	data    []byte
	imports map[string]*types.Package
	pkgList []*types.Package
	typList []types.Type

	// debugging support
	consumed int
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
		pkg = types.NewPackage(path, name, types.NewScope(nil))
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
	case stringTag:
		return exact.MakeString(p.string())
	case intTag:
		return exact.MakeInt64(p.int64())
	case floatTag:
		return p.float()
	case fractionTag:
		return p.fraction()
	case complexTag:
		re := p.fraction()
		im := p.fraction()
		return exact.BinaryOp(re, token.ADD, exact.MakeImag(im))
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
	case basicTag:
		t := types.Universe.Lookup(p.string()).(*types.TypeName).Type().(*types.Basic)
		p.record(t)
		return t

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
		t := new(types.Interface)
		p.record(t)

		methods := make([]*types.Func, p.int())
		for i := range methods {
			pkg, name := p.qualifiedName()
			methods[i] = types.NewFunc(token.NoPos, pkg, name, p.typ().(*types.Signature))
		}

		embeddeds := make([]*types.Named, p.int())
		for i := range embeddeds {
			embeddeds[i] = p.typ().(*types.Named)
		}
		*t = *types.NewInterface(methods, embeddeds)
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
		// import type object
		name := p.string()
		pkg := p.pkg()
		scope := pkg.Scope()
		obj := scope.Lookup(name)
		if obj == nil {
			new := types.NewTypeName(token.NoPos, pkg, name, nil)
			types.NewNamed(new, nil, nil)
			scope.Insert(new)
			obj = new
		}
		t := obj.Type().(*types.Named)
		p.record(t)

		// import underlying type
		u := p.typ()
		if t.Underlying() == nil {
			t.SetUnderlying(u)
		}

		// read associated methods
		n := p.int()
		for i := 0; i < n; i++ {
			t.AddMethod(types.NewFunc(token.NoPos, pkg, p.string(), p.typ().(*types.Signature)))
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
			pkg = obj.Pkg() // TODO(gri) is this still correct?
			name = obj.Name()
		default:
			panic("anonymous field expected")
		}
		anonymous = true
	}

	return types.NewField(token.NoPos, pkg, name, typ, anonymous)
}

func (p *importer) qualifiedName() (*types.Package, string) {
	name := p.string()
	pkg := p.pkgList[1] // exported names assume current package
	if !exported(name) {
		pkg = p.pkg()
		if pkg == nil {
			panic(fmt.Sprintf("nil package for unexported qualified name %q", name))
		}
	}
	return pkg, name
}

func (p *importer) signature() *types.Signature {
	var recv *types.Var
	if p.bool() {
		recv = p.param()
	}
	return types.NewSignature(nil, recv, p.tuple(), p.tuple(), p.bool())
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

func (p *importer) bool() bool {
	return p.int64() != 0
}

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
		if debug {
			p.consumed += n
		}
	}
	return b
}

func (p *importer) marker(want byte) {
	if debug {
		if got := p.data[0]; got != want {
			panic(fmt.Sprintf("incorrect marker: got %c; want %c (pos = %d)", got, want, p.consumed))
		}
		p.data = p.data[1:]
		p.consumed++

		pos := p.consumed
		if n := int(p.rawInt64()); n != pos {
			panic(fmt.Sprintf("incorrect position: got %d; want %d", n, pos))
		}
	}
}

// rawInt64 should only be used by low-level decoders
func (p *importer) rawInt64() int64 {
	i, n := binary.Varint(p.data)
	p.data = p.data[n:]
	if debug {
		p.consumed += n
	}
	return i
}
