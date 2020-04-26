// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See cmd/compile/internal/gc/iexport.go for the export data format.

// This file is a copy of $GOROOT/src/go/internal/gcimporter/iimport.go.

package gcimporter

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"sort"
	"sync"
	"unicode"
	"unicode/utf8"
)

type intReader struct {
	*bytes.Reader
	path string
}

func errorf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
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

// IImportData imports a package from the serialized package data
// and returns the number of bytes consumed and a reference to the package.
// If the export data version is not recognized or the format is otherwise
// compromised, an error is returned.
func IImportData(fset *token.FileSet, imports map[string]*types.Package, data []byte, path string) (_ int, pkg *types.Package, err error) {
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

	whence, _ := r.Seek(0, io.SeekCurrent)
	stringData := data[whence : whence+sLen]
	declData := data[whence+sLen : whence+sLen+dLen]
	r.Seek(sLen+dLen, io.SeekCurrent)

	p := iimporter{
		ipath:   path,
		version: int(version),

		stringData:  stringData,
		stringCache: make(map[uint64]string),
		pkgCache:    make(map[uint64]*types.Package),

		declData: declData,
		pkgIndex: make(map[*types.Package]map[string]uint64),
		typCache: make(map[uint64]types.Type),

		fake: fakeFileSet{
			fset:  fset,
			files: make(map[string]*token.File),
		},
	}

	for i, pt := range predeclared() {
		p.typCache[uint64(i)] = pt
	}

	pkgList := make([]*types.Package, r.uint64())
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
			pkg = types.NewPackage(pkgPath, pkgName)
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
	if len(pkgList) == 0 {
		errorf("no packages found for %s", path)
		panic("unreachable")
	}
	p.ipkg = pkgList[0]
	names := make([]string, 0, len(p.pkgIndex[p.ipkg]))
	for name := range p.pkgIndex[p.ipkg] {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		p.doDecl(p.ipkg, name)
	}

	for _, typ := range p.interfaceList {
		typ.Complete()
	}

	// record all referenced packages as imports
	list := append(([]*types.Package)(nil), pkgList[1:]...)
	sort.Sort(byPath(list))
	p.ipkg.SetImports(list)

	// package was imported completely and without errors
	p.ipkg.MarkComplete()

	consumed, _ := r.Seek(0, io.SeekCurrent)
	return int(consumed), p.ipkg, nil
}

type iimporter struct {
	ipath   string
	ipkg    *types.Package
	version int

	stringData  []byte
	stringCache map[uint64]string
	pkgCache    map[uint64]*types.Package

	declData []byte
	pkgIndex map[*types.Package]map[string]uint64
	typCache map[uint64]types.Type

	fake          fakeFileSet
	interfaceList []*types.Interface
}

func (p *iimporter) doDecl(pkg *types.Package, name string) {
	// See if we've already imported this declaration.
	if obj := pkg.Scope().Lookup(name); obj != nil {
		return
	}

	off, ok := p.pkgIndex[pkg][name]
	if !ok {
		errorf("%v.%v not in index", pkg, name)
	}

	r := &importReader{p: p, currPkg: pkg}
	r.declReader.Reset(p.declData[off:])

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

func (p *iimporter) pkgAt(off uint64) *types.Package {
	if pkg, ok := p.pkgCache[off]; ok {
		return pkg
	}
	path := p.stringAt(off)
	if path == p.ipath {
		return p.ipkg
	}
	errorf("missing package %q in %q", path, p.ipath)
	return nil
}

func (p *iimporter) typAt(off uint64, base *types.Named) types.Type {
	if t, ok := p.typCache[off]; ok && (base == nil || !isInterface(t)) {
		return t
	}

	if off < predeclReserved {
		errorf("predeclared type missing from cache: %v", off)
	}

	r := &importReader{p: p}
	r.declReader.Reset(p.declData[off-predeclReserved:])
	t := r.doType(base)

	if base == nil || !isInterface(t) {
		p.typCache[off] = t
	}
	return t
}

type importReader struct {
	p          *iimporter
	declReader bytes.Reader
	currPkg    *types.Package
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

		r.declare(types.NewTypeName(pos, r.currPkg, name, typ))

	case 'C':
		typ, val := r.value()

		r.declare(types.NewConst(pos, r.currPkg, name, typ, val))

	case 'F':
		sig := r.signature(nil)

		r.declare(types.NewFunc(pos, r.currPkg, name, sig))

	case 'T':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		obj := types.NewTypeName(pos, r.currPkg, name, nil)
		named := types.NewNamed(obj, nil, nil)
		r.declare(obj)

		underlying := r.p.typAt(r.uint64(), named).Underlying()
		named.SetUnderlying(underlying)

		if !isInterface(underlying) {
			for n := r.uint64(); n > 0; n-- {
				mpos := r.pos()
				mname := r.ident()
				recv := r.param()
				msig := r.signature(recv)

				named.AddMethod(types.NewFunc(mpos, r.currPkg, mname, msig))
			}
		}

	case 'V':
		typ := r.typ()

		r.declare(types.NewVar(pos, r.currPkg, name, typ))

	default:
		errorf("unexpected tag: %v", tag)
	}
}

func (r *importReader) declare(obj types.Object) {
	obj.Pkg().Scope().Insert(obj)
}

func (r *importReader) value() (typ types.Type, val constant.Value) {
	typ = r.typ()

	switch b := typ.Underlying().(*types.Basic); b.Info() & types.IsConstType {
	case types.IsBoolean:
		val = constant.MakeBool(r.bool())

	case types.IsString:
		val = constant.MakeString(r.string())

	case types.IsInteger:
		val = r.mpint(b)

	case types.IsFloat:
		val = r.mpfloat(b)

	case types.IsComplex:
		re := r.mpfloat(b)
		im := r.mpfloat(b)
		val = constant.BinaryOp(re, token.ADD, constant.MakeImag(im))

	default:
		if b.Kind() == types.Invalid {
			val = constant.MakeUnknown()
			return
		}
		errorf("unexpected type %v", typ) // panics
		panic("unreachable")
	}

	return
}

func intSize(b *types.Basic) (signed bool, maxBytes uint) {
	if (b.Info() & types.IsUntyped) != 0 {
		return true, 64
	}

	switch b.Kind() {
	case types.Float32, types.Complex64:
		return true, 3
	case types.Float64, types.Complex128:
		return true, 7
	}

	signed = (b.Info() & types.IsUnsigned) == 0
	switch b.Kind() {
	case types.Int8, types.Uint8:
		maxBytes = 1
	case types.Int16, types.Uint16:
		maxBytes = 2
	case types.Int32, types.Uint32:
		maxBytes = 4
	default:
		maxBytes = 8
	}

	return
}

func (r *importReader) mpint(b *types.Basic) constant.Value {
	signed, maxBytes := intSize(b)

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
		return constant.MakeInt64(v)
	}

	v := -n
	if signed {
		v = -(n &^ 1) >> 1
	}
	if v < 1 || uint(v) > maxBytes {
		errorf("weird decoding: %v, %v => %v", n, signed, v)
	}

	buf := make([]byte, v)
	io.ReadFull(&r.declReader, buf)

	// convert to little endian
	// TODO(gri) go/constant should have a more direct conversion function
	//           (e.g., once it supports a big.Float based implementation)
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}

	x := constant.MakeFromBytes(buf)
	if signed && n&1 != 0 {
		x = constant.UnaryOp(token.SUB, x, 0)
	}
	return x
}

func (r *importReader) mpfloat(b *types.Basic) constant.Value {
	x := r.mpint(b)
	if constant.Sign(x) == 0 {
		return x
	}

	exp := r.int64()
	switch {
	case exp > 0:
		x = constant.Shift(x, token.SHL, uint(exp))
	case exp < 0:
		d := constant.Shift(constant.MakeInt64(1), token.SHL, uint(-exp))
		x = constant.BinaryOp(x, token.QUO, d)
	}
	return x
}

func (r *importReader) ident() string {
	return r.string()
}

func (r *importReader) qualifiedIdent() (*types.Package, string) {
	name := r.string()
	pkg := r.pkg()
	return pkg, name
}

func (r *importReader) pos() token.Pos {
	if r.p.version >= 1 {
		r.posv1()
	} else {
		r.posv0()
	}

	if r.prevFile == "" && r.prevLine == 0 && r.prevColumn == 0 {
		return token.NoPos
	}
	return r.p.fake.pos(r.prevFile, int(r.prevLine), int(r.prevColumn))
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

func (r *importReader) typ() types.Type {
	return r.p.typAt(r.uint64(), nil)
}

func isInterface(t types.Type) bool {
	_, ok := t.(*types.Interface)
	return ok
}

func (r *importReader) pkg() *types.Package { return r.p.pkgAt(r.uint64()) }
func (r *importReader) string() string      { return r.p.stringAt(r.uint64()) }

func (r *importReader) doType(base *types.Named) types.Type {
	switch k := r.kind(); k {
	default:
		errorf("unexpected kind tag in %q: %v", r.p.ipath, k)
		return nil

	case definedType:
		pkg, name := r.qualifiedIdent()
		r.p.doDecl(pkg, name)
		return pkg.Scope().Lookup(name).(*types.TypeName).Type()
	case pointerType:
		return types.NewPointer(r.typ())
	case sliceType:
		return types.NewSlice(r.typ())
	case arrayType:
		n := r.uint64()
		return types.NewArray(r.typ(), int64(n))
	case chanType:
		dir := chanDir(int(r.uint64()))
		return types.NewChan(dir, r.typ())
	case mapType:
		return types.NewMap(r.typ(), r.typ())
	case signatureType:
		r.currPkg = r.pkg()
		return r.signature(nil)

	case structType:
		r.currPkg = r.pkg()

		fields := make([]*types.Var, r.uint64())
		tags := make([]string, len(fields))
		for i := range fields {
			fpos := r.pos()
			fname := r.ident()
			ftyp := r.typ()
			emb := r.bool()
			tag := r.string()

			fields[i] = types.NewField(fpos, r.currPkg, fname, ftyp, emb)
			tags[i] = tag
		}
		return types.NewStruct(fields, tags)

	case interfaceType:
		r.currPkg = r.pkg()

		embeddeds := make([]types.Type, r.uint64())
		for i := range embeddeds {
			_ = r.pos()
			embeddeds[i] = r.typ()
		}

		methods := make([]*types.Func, r.uint64())
		for i := range methods {
			mpos := r.pos()
			mname := r.ident()

			// TODO(mdempsky): Matches bimport.go, but I
			// don't agree with this.
			var recv *types.Var
			if base != nil {
				recv = types.NewVar(token.NoPos, r.currPkg, "", base)
			}

			msig := r.signature(recv)
			methods[i] = types.NewFunc(mpos, r.currPkg, mname, msig)
		}

		typ := newInterface(methods, embeddeds)
		r.p.interfaceList = append(r.p.interfaceList, typ)
		return typ
	}
}

func (r *importReader) kind() itag {
	return itag(r.uint64())
}

func (r *importReader) signature(recv *types.Var) *types.Signature {
	params := r.paramList()
	results := r.paramList()
	variadic := params.Len() > 0 && r.bool()
	return types.NewSignature(recv, params, results, variadic)
}

func (r *importReader) paramList() *types.Tuple {
	xs := make([]*types.Var, r.uint64())
	for i := range xs {
		xs[i] = r.param()
	}
	return types.NewTuple(xs...)
}

func (r *importReader) param() *types.Var {
	pos := r.pos()
	name := r.ident()
	typ := r.typ()
	return types.NewParam(pos, r.currPkg, name, typ)
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

const deltaNewFile = -64 // see cmd/compile/internal/gc/bexport.go

// Synthesize a token.Pos
type fakeFileSet struct {
	fset  *token.FileSet
	files map[string]*token.File
}

func (s *fakeFileSet) pos(file string, line, column int) token.Pos {
	// TODO(mdempsky): Make use of column.

	// Since we don't know the set of needed file positions, we
	// reserve maxlines positions per file.
	const maxlines = 64 * 1024
	f := s.files[file]
	if f == nil {
		f = s.fset.AddFile(file, -1, maxlines)
		s.files[file] = f
		// Allocate the fake linebreak indices on first use.
		// TODO(adonovan): opt: save ~512KB using a more complex scheme?
		fakeLinesOnce.Do(func() {
			fakeLines = make([]int, maxlines)
			for i := range fakeLines {
				fakeLines[i] = i
			}
		})
		f.SetLines(fakeLines)
	}

	if line > maxlines {
		line = 1
	}

	// Treat the file as if it contained only newlines
	// and column=1: use the line number as the offset.
	return f.Pos(line - 1)
}

var (
	fakeLines     []int
	fakeLinesOnce sync.Once
)

func chanDir(d int) types.ChanDir {
	// tag values must match the constants in cmd/compile/internal/gc/go.go
	switch d {
	case 1 /* Crecv */ :
		return types.RecvOnly
	case 2 /* Csend */ :
		return types.SendOnly
	case 3 /* Cboth */ :
		return types.SendRecv
	default:
		errorf("unexpected channel dir %d", d)
		return 0
	}
}

func exported(name string) bool {
	ch, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(ch)
}

// ----------------------------------------------------------------------------
// Export format

// Tags. Must be < 0.
const (
	// Objects
	packageTag = -(iota + 1)
	constTag
	typeTag
	varTag
	funcTag
	endTag

	// Types
	namedTag
	arrayTag
	sliceTag
	dddTag
	structTag
	pointerTag
	signatureTag
	interfaceTag
	mapTag
	chanTag

	// Values
	falseTag
	trueTag
	int64Tag
	floatTag
	fractionTag // not used by gc
	complexTag
	stringTag
	nilTag     // only used by gc (appears in exported inlined function bodies)
	unknownTag // not used by gc (only appears in packages with errors)

	// Type aliases
	aliasTag
)

var predeclOnce sync.Once
var predecl []types.Type // initialized lazily

func predeclared() []types.Type {
	predeclOnce.Do(func() {
		// initialize lazily to be sure that all
		// elements have been initialized before
		predecl = []types.Type{ // basic types
			types.Typ[types.Bool],
			types.Typ[types.Int],
			types.Typ[types.Int8],
			types.Typ[types.Int16],
			types.Typ[types.Int32],
			types.Typ[types.Int64],
			types.Typ[types.Uint],
			types.Typ[types.Uint8],
			types.Typ[types.Uint16],
			types.Typ[types.Uint32],
			types.Typ[types.Uint64],
			types.Typ[types.Uintptr],
			types.Typ[types.Float32],
			types.Typ[types.Float64],
			types.Typ[types.Complex64],
			types.Typ[types.Complex128],
			types.Typ[types.String],

			// basic type aliases
			types.Universe.Lookup("byte").Type(),
			types.Universe.Lookup("rune").Type(),

			// error
			types.Universe.Lookup("error").Type(),

			// untyped types
			types.Typ[types.UntypedBool],
			types.Typ[types.UntypedInt],
			types.Typ[types.UntypedRune],
			types.Typ[types.UntypedFloat],
			types.Typ[types.UntypedComplex],
			types.Typ[types.UntypedString],
			types.Typ[types.UntypedNil],

			// package unsafe
			types.Typ[types.UnsafePointer],

			// invalid type
			types.Typ[types.Invalid], // only appears in packages with errors

			// used internally by gc; never used by this package or in .a files
			anyType{},
		}
	})
	return predecl
}

type anyType struct{}

func (t anyType) Underlying() types.Type { return t }
func (t anyType) String() string         { return "any" }
