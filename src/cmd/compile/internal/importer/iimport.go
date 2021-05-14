// UNREVIEWED
// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See cmd/compile/internal/typecheck/iexport.go for the export data format.

package importer

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"io"
	"math/big"
	"sort"
)

type intReader struct {
	*bytes.Reader
	path string
}

func (r *intReader) int64() int64 {
	i, err := binary.ReadVarint(r.Reader)
	if err != nil {
		errorf("import %q: read varint error: %v", r.path, err)
	}
	return i
}

func (r *intReader) uint64() uint64 {
	i, err := binary.ReadUvarint(r.Reader)
	if err != nil {
		errorf("import %q: read varint error: %v", r.path, err)
	}
	return i
}

const predeclReserved = 32

type itag uint64

const (
	// Types
	definedType itag = iota
	pointerType
	sliceType
	arrayType
	chanType
	mapType
	signatureType
	structType
	interfaceType
)

const io_SeekCurrent = 1 // io.SeekCurrent (not defined in Go 1.4)

// iImportData imports a package from the serialized package data
// and returns the number of bytes consumed and a reference to the package.
// If the export data version is not recognized or the format is otherwise
// compromised, an error is returned.
func iImportData(imports map[string]*types2.Package, data []byte, path string) (_ int, pkg *types2.Package, err error) {
	const currentVersion = 1
	version := int64(-1)
	defer func() {
		if e := recover(); e != nil {
			if version > currentVersion {
				err = fmt.Errorf("cannot import %q (%v), export data is newer version - update tool", path, e)
			} else {
				err = fmt.Errorf("cannot import %q (%v), possibly version skew - reinstall package", path, e)
			}
		}
	}()

	r := &intReader{bytes.NewReader(data), path}

	version = int64(r.uint64())
	switch version {
	case currentVersion, 0:
	default:
		errorf("unknown iexport format version %d", version)
	}

	sLen := int64(r.uint64())
	dLen := int64(r.uint64())

	whence, _ := r.Seek(0, io_SeekCurrent)
	stringData := data[whence : whence+sLen]
	declData := data[whence+sLen : whence+sLen+dLen]
	r.Seek(sLen+dLen, io_SeekCurrent)

	p := iimporter{
		ipath:   path,
		version: int(version),

		stringData:  stringData,
		stringCache: make(map[uint64]string),
		pkgCache:    make(map[uint64]*types2.Package),

		declData: declData,
		pkgIndex: make(map[*types2.Package]map[string]uint64),
		typCache: make(map[uint64]types2.Type),
	}

	for i, pt := range predeclared {
		p.typCache[uint64(i)] = pt
	}

	pkgList := make([]*types2.Package, r.uint64())
	for i := range pkgList {
		pkgPathOff := r.uint64()
		pkgPath := p.stringAt(pkgPathOff)
		pkgName := p.stringAt(r.uint64())
		_ = r.uint64() // package height; unused by go/types

		if pkgPath == "" {
			pkgPath = path
		}
		pkg := imports[pkgPath]
		if pkg == nil {
			pkg = types2.NewPackage(pkgPath, pkgName)
			imports[pkgPath] = pkg
		} else if pkg.Name() != pkgName {
			errorf("conflicting names %s and %s for package %q", pkg.Name(), pkgName, path)
		}

		p.pkgCache[pkgPathOff] = pkg

		nameIndex := make(map[string]uint64)
		for nSyms := r.uint64(); nSyms > 0; nSyms-- {
			name := p.stringAt(r.uint64())
			nameIndex[name] = r.uint64()
		}

		p.pkgIndex[pkg] = nameIndex
		pkgList[i] = pkg
	}

	localpkg := pkgList[0]

	names := make([]string, 0, len(p.pkgIndex[localpkg]))
	for name := range p.pkgIndex[localpkg] {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		p.doDecl(localpkg, name)
	}

	for _, typ := range p.interfaceList {
		typ.Complete()
	}

	// record all referenced packages as imports
	list := append(([]*types2.Package)(nil), pkgList[1:]...)
	sort.Sort(byPath(list))
	localpkg.SetImports(list)

	// package was imported completely and without errors
	localpkg.MarkComplete()

	consumed, _ := r.Seek(0, io_SeekCurrent)
	return int(consumed), localpkg, nil
}

type iimporter struct {
	ipath   string
	version int

	stringData  []byte
	stringCache map[uint64]string
	pkgCache    map[uint64]*types2.Package

	declData []byte
	pkgIndex map[*types2.Package]map[string]uint64
	typCache map[uint64]types2.Type

	interfaceList []*types2.Interface
}

func (p *iimporter) doDecl(pkg *types2.Package, name string) {
	// See if we've already imported this declaration.
	if obj := pkg.Scope().Lookup(name); obj != nil {
		return
	}

	off, ok := p.pkgIndex[pkg][name]
	if !ok {
		errorf("%v.%v not in index", pkg, name)
	}

	r := &importReader{p: p, currPkg: pkg}
	// Reader.Reset is not available in Go 1.4.
	// Use bytes.NewReader for now.
	// r.declReader.Reset(p.declData[off:])
	r.declReader = *bytes.NewReader(p.declData[off:])

	r.obj(name)
}

func (p *iimporter) stringAt(off uint64) string {
	if s, ok := p.stringCache[off]; ok {
		return s
	}

	slen, n := binary.Uvarint(p.stringData[off:])
	if n <= 0 {
		errorf("varint failed")
	}
	spos := off + uint64(n)
	s := string(p.stringData[spos : spos+slen])
	p.stringCache[off] = s
	return s
}

func (p *iimporter) pkgAt(off uint64) *types2.Package {
	if pkg, ok := p.pkgCache[off]; ok {
		return pkg
	}
	path := p.stringAt(off)
	errorf("missing package %q in %q", path, p.ipath)
	return nil
}

func (p *iimporter) typAt(off uint64, base *types2.Named) types2.Type {
	if t, ok := p.typCache[off]; ok && (base == nil || !isInterface(t)) {
		return t
	}

	if off < predeclReserved {
		errorf("predeclared type missing from cache: %v", off)
	}

	r := &importReader{p: p}
	// Reader.Reset is not available in Go 1.4.
	// Use bytes.NewReader for now.
	// r.declReader.Reset(p.declData[off-predeclReserved:])
	r.declReader = *bytes.NewReader(p.declData[off-predeclReserved:])
	t := r.doType(base)

	if base == nil || !isInterface(t) {
		p.typCache[off] = t
	}
	return t
}

type importReader struct {
	p          *iimporter
	declReader bytes.Reader
	currPkg    *types2.Package
	prevFile   string
	prevLine   int64
	prevColumn int64
}

func (r *importReader) obj(name string) {
	tag := r.byte()
	pos := r.pos()

	switch tag {
	case 'A':
		typ := r.typ()

		r.declare(types2.NewTypeName(pos, r.currPkg, name, typ))

	case 'C':
		typ, val := r.value()

		r.declare(types2.NewConst(pos, r.currPkg, name, typ, val))

	case 'F':
		sig := r.signature(nil)

		r.declare(types2.NewFunc(pos, r.currPkg, name, sig))

	case 'T':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		obj := types2.NewTypeName(pos, r.currPkg, name, nil)
		named := types2.NewNamed(obj, nil, nil)
		r.declare(obj)

		underlying := r.p.typAt(r.uint64(), named).Underlying()
		named.SetUnderlying(underlying)

		if !isInterface(underlying) {
			for n := r.uint64(); n > 0; n-- {
				mpos := r.pos()
				mname := r.ident()
				recv := r.param()
				msig := r.signature(recv)

				named.AddMethod(types2.NewFunc(mpos, r.currPkg, mname, msig))
			}
		}

	case 'V':
		typ := r.typ()

		r.declare(types2.NewVar(pos, r.currPkg, name, typ))

	default:
		errorf("unexpected tag: %v", tag)
	}
}

func (r *importReader) declare(obj types2.Object) {
	obj.Pkg().Scope().Insert(obj)
}

func (r *importReader) value() (typ types2.Type, val constant.Value) {
	typ = r.typ()

	switch b := typ.Underlying().(*types2.Basic); b.Info() & types2.IsConstType {
	case types2.IsBoolean:
		val = constant.MakeBool(r.bool())

	case types2.IsString:
		val = constant.MakeString(r.string())

	case types2.IsInteger:
		var x big.Int
		r.mpint(&x, b)
		val = constant.Make(&x)

	case types2.IsFloat:
		val = r.mpfloat(b)

	case types2.IsComplex:
		re := r.mpfloat(b)
		im := r.mpfloat(b)
		val = constant.BinaryOp(re, token.ADD, constant.MakeImag(im))

	default:
		errorf("unexpected type %v", typ) // panics
		panic("unreachable")
	}

	return
}

func intSize(b *types2.Basic) (signed bool, maxBytes uint) {
	if (b.Info() & types2.IsUntyped) != 0 {
		return true, 64
	}

	switch b.Kind() {
	case types2.Float32, types2.Complex64:
		return true, 3
	case types2.Float64, types2.Complex128:
		return true, 7
	}

	signed = (b.Info() & types2.IsUnsigned) == 0
	switch b.Kind() {
	case types2.Int8, types2.Uint8:
		maxBytes = 1
	case types2.Int16, types2.Uint16:
		maxBytes = 2
	case types2.Int32, types2.Uint32:
		maxBytes = 4
	default:
		maxBytes = 8
	}

	return
}

func (r *importReader) mpint(x *big.Int, typ *types2.Basic) {
	signed, maxBytes := intSize(typ)

	maxSmall := 256 - maxBytes
	if signed {
		maxSmall = 256 - 2*maxBytes
	}
	if maxBytes == 1 {
		maxSmall = 256
	}

	n, _ := r.declReader.ReadByte()
	if uint(n) < maxSmall {
		v := int64(n)
		if signed {
			v >>= 1
			if n&1 != 0 {
				v = ^v
			}
		}
		x.SetInt64(v)
		return
	}

	v := -n
	if signed {
		v = -(n &^ 1) >> 1
	}
	if v < 1 || uint(v) > maxBytes {
		errorf("weird decoding: %v, %v => %v", n, signed, v)
	}
	b := make([]byte, v)
	io.ReadFull(&r.declReader, b)
	x.SetBytes(b)
	if signed && n&1 != 0 {
		x.Neg(x)
	}
}

func (r *importReader) mpfloat(typ *types2.Basic) constant.Value {
	var mant big.Int
	r.mpint(&mant, typ)
	var f big.Float
	f.SetInt(&mant)
	if f.Sign() != 0 {
		f.SetMantExp(&f, int(r.int64()))
	}
	return constant.Make(&f)
}

func (r *importReader) ident() string {
	return r.string()
}

func (r *importReader) qualifiedIdent() (*types2.Package, string) {
	name := r.string()
	pkg := r.pkg()
	return pkg, name
}

func (r *importReader) pos() syntax.Pos {
	if r.p.version >= 1 {
		r.posv1()
	} else {
		r.posv0()
	}

	if r.prevFile == "" && r.prevLine == 0 && r.prevColumn == 0 {
		return syntax.Pos{}
	}
	// TODO(gri) fix this
	// return r.p.fake.pos(r.prevFile, int(r.prevLine), int(r.prevColumn))
	return syntax.Pos{}
}

func (r *importReader) posv0() {
	delta := r.int64()
	if delta != deltaNewFile {
		r.prevLine += delta
	} else if l := r.int64(); l == -1 {
		r.prevLine += deltaNewFile
	} else {
		r.prevFile = r.string()
		r.prevLine = l
	}
}

func (r *importReader) posv1() {
	delta := r.int64()
	r.prevColumn += delta >> 1
	if delta&1 != 0 {
		delta = r.int64()
		r.prevLine += delta >> 1
		if delta&1 != 0 {
			r.prevFile = r.string()
		}
	}
}

func (r *importReader) typ() types2.Type {
	return r.p.typAt(r.uint64(), nil)
}

func isInterface(t types2.Type) bool {
	_, ok := t.(*types2.Interface)
	return ok
}

func (r *importReader) pkg() *types2.Package { return r.p.pkgAt(r.uint64()) }
func (r *importReader) string() string       { return r.p.stringAt(r.uint64()) }

func (r *importReader) doType(base *types2.Named) types2.Type {
	switch k := r.kind(); k {
	default:
		errorf("unexpected kind tag in %q: %v", r.p.ipath, k)
		return nil

	case definedType:
		pkg, name := r.qualifiedIdent()
		r.p.doDecl(pkg, name)
		return pkg.Scope().Lookup(name).(*types2.TypeName).Type()
	case pointerType:
		return types2.NewPointer(r.typ())
	case sliceType:
		return types2.NewSlice(r.typ())
	case arrayType:
		n := r.uint64()
		return types2.NewArray(r.typ(), int64(n))
	case chanType:
		dir := chanDir(int(r.uint64()))
		return types2.NewChan(dir, r.typ())
	case mapType:
		return types2.NewMap(r.typ(), r.typ())
	case signatureType:
		r.currPkg = r.pkg()
		return r.signature(nil)

	case structType:
		r.currPkg = r.pkg()

		fields := make([]*types2.Var, r.uint64())
		tags := make([]string, len(fields))
		for i := range fields {
			fpos := r.pos()
			fname := r.ident()
			ftyp := r.typ()
			emb := r.bool()
			tag := r.string()

			fields[i] = types2.NewField(fpos, r.currPkg, fname, ftyp, emb)
			tags[i] = tag
		}
		return types2.NewStruct(fields, tags)

	case interfaceType:
		r.currPkg = r.pkg()

		embeddeds := make([]types2.Type, r.uint64())
		for i := range embeddeds {
			_ = r.pos()
			embeddeds[i] = r.typ()
		}

		methods := make([]*types2.Func, r.uint64())
		for i := range methods {
			mpos := r.pos()
			mname := r.ident()

			// TODO(mdempsky): Matches bimport.go, but I
			// don't agree with this.
			var recv *types2.Var
			if base != nil {
				recv = types2.NewVar(syntax.Pos{}, r.currPkg, "", base)
			}

			msig := r.signature(recv)
			methods[i] = types2.NewFunc(mpos, r.currPkg, mname, msig)
		}

		typ := types2.NewInterfaceType(methods, embeddeds)
		r.p.interfaceList = append(r.p.interfaceList, typ)
		return typ
	}
}

func (r *importReader) kind() itag {
	return itag(r.uint64())
}

func (r *importReader) signature(recv *types2.Var) *types2.Signature {
	params := r.paramList()
	results := r.paramList()
	variadic := params.Len() > 0 && r.bool()
	return types2.NewSignature(recv, params, results, variadic)
}

func (r *importReader) paramList() *types2.Tuple {
	xs := make([]*types2.Var, r.uint64())
	for i := range xs {
		xs[i] = r.param()
	}
	return types2.NewTuple(xs...)
}

func (r *importReader) param() *types2.Var {
	pos := r.pos()
	name := r.ident()
	typ := r.typ()
	return types2.NewParam(pos, r.currPkg, name, typ)
}

func (r *importReader) bool() bool {
	return r.uint64() != 0
}

func (r *importReader) int64() int64 {
	n, err := binary.ReadVarint(&r.declReader)
	if err != nil {
		errorf("readVarint: %v", err)
	}
	return n
}

func (r *importReader) uint64() uint64 {
	n, err := binary.ReadUvarint(&r.declReader)
	if err != nil {
		errorf("readUvarint: %v", err)
	}
	return n
}

func (r *importReader) byte() byte {
	x, err := r.declReader.ReadByte()
	if err != nil {
		errorf("declReader.ReadByte: %v", err)
	}
	return x
}
