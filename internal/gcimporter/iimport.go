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
	"math/big"
	"sort"
	"strings"

	"golang.org/x/tools/internal/typeparams"
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

// Keep this in sync with constants in iexport.go.
const (
	iexportVersionGo1_11   = 0
	iexportVersionPosCol   = 1
	iexportVersionGo1_18   = 2
	iexportVersionGenerics = 2

	iexportVersionCurrent = 2
)

type ident struct {
	pkg  *types.Package
	name string
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
	typeParamType
	instanceType
	unionType
)

// IImportData imports a package from the serialized package data
// and returns 0 and a reference to the package.
// If the export data version is not recognized or the format is otherwise
// compromised, an error is returned.
func IImportData(fset *token.FileSet, imports map[string]*types.Package, data []byte, path string) (int, *types.Package, error) {
	pkgs, err := iimportCommon(fset, GetPackagesFromMap(imports), data, false, path, false)
	if err != nil {
		return 0, nil, err
	}
	return 0, pkgs[0], nil
}

// IImportBundle imports a set of packages from the serialized package bundle.
func IImportBundle(fset *token.FileSet, imports map[string]*types.Package, data []byte) ([]*types.Package, error) {
	return iimportCommon(fset, GetPackagesFromMap(imports), data, true, "", false)
}

// A GetPackagesFunc function obtains the non-nil symbols for a set of
// packages, creating and recursively importing them as needed. An
// implementation should store each package symbol is in the Pkg
// field of the items array.
//
// Any error causes importing to fail. This can be used to quickly read
// the import manifest of an export data file without fully decoding it.
type GetPackagesFunc = func(items []GetPackagesItem) error

// A GetPackagesItem is a request from the importer for the package
// symbol of the specified name and path.
type GetPackagesItem struct {
	Name, Path string
	Pkg        *types.Package // to be filled in by GetPackagesFunc call

	// private importer state
	pathOffset uint64
	nameIndex  map[string]uint64
}

// GetPackagesFromMap returns a GetPackagesFunc that retrieves
// packages from the given map of package path to package.
//
// The returned function may mutate m: each requested package that is not
// found is created with types.NewPackage and inserted into m.
func GetPackagesFromMap(m map[string]*types.Package) GetPackagesFunc {
	return func(items []GetPackagesItem) error {
		for i, item := range items {
			pkg, ok := m[item.Path]
			if !ok {
				pkg = types.NewPackage(item.Path, item.Name)
				m[item.Path] = pkg
			}
			items[i].Pkg = pkg
		}
		return nil
	}
}

func iimportCommon(fset *token.FileSet, getPackages GetPackagesFunc, data []byte, bundle bool, path string, shallow bool) (pkgs []*types.Package, err error) {
	const currentVersion = iexportVersionCurrent
	version := int64(-1)
	if !debug {
		defer func() {
			if e := recover(); e != nil {
				if bundle {
					err = fmt.Errorf("%v", e)
				} else if version > currentVersion {
					err = fmt.Errorf("cannot import %q (%v), export data is newer version - update tool", path, e)
				} else {
					err = fmt.Errorf("internal error while importing %q (%v); please report an issue", path, e)
				}
			}
		}()
	}

	r := &intReader{bytes.NewReader(data), path}

	if bundle {
		if v := r.uint64(); v != bundleVersion {
			errorf("unknown bundle format version %d", v)
		}
	}

	version = int64(r.uint64())
	switch version {
	case iexportVersionGo1_18, iexportVersionPosCol, iexportVersionGo1_11:
	default:
		if version > iexportVersionGo1_18 {
			errorf("unstable iexport format version %d, just rebuild compiler and std library", version)
		} else {
			errorf("unknown iexport format version %d", version)
		}
	}

	sLen := int64(r.uint64())
	var fLen int64
	var fileOffset []uint64
	if shallow {
		// Shallow mode uses a different position encoding.
		fLen = int64(r.uint64())
		fileOffset = make([]uint64, r.uint64())
		for i := range fileOffset {
			fileOffset[i] = r.uint64()
		}
	}
	dLen := int64(r.uint64())

	whence, _ := r.Seek(0, io.SeekCurrent)
	stringData := data[whence : whence+sLen]
	fileData := data[whence+sLen : whence+sLen+fLen]
	declData := data[whence+sLen+fLen : whence+sLen+fLen+dLen]
	r.Seek(sLen+fLen+dLen, io.SeekCurrent)

	p := iimporter{
		version:  int(version),
		ipath:    path,
		usePosv2: shallow, // precise offsets are encoded only in shallow mode

		stringData:  stringData,
		stringCache: make(map[uint64]string),
		fileOffset:  fileOffset,
		fileData:    fileData,
		fileCache:   make([]*token.File, len(fileOffset)),
		pkgCache:    make(map[uint64]*types.Package),

		declData: declData,
		pkgIndex: make(map[*types.Package]map[string]uint64),
		typCache: make(map[uint64]types.Type),
		// Separate map for typeparams, keyed by their package and unique
		// name.
		tparamIndex: make(map[ident]types.Type),

		fake: fakeFileSet{
			fset:  fset,
			files: make(map[string]*fileInfo),
		},
	}
	defer p.fake.setLines() // set lines for files in fset

	for i, pt := range predeclared() {
		p.typCache[uint64(i)] = pt
	}

	// Gather the relevant packages from the manifest.
	items := make([]GetPackagesItem, r.uint64())
	for i := range items {
		pkgPathOff := r.uint64()
		pkgPath := p.stringAt(pkgPathOff)
		pkgName := p.stringAt(r.uint64())
		_ = r.uint64() // package height; unused by go/types

		if pkgPath == "" {
			pkgPath = path
		}
		items[i].Name = pkgName
		items[i].Path = pkgPath
		items[i].pathOffset = pkgPathOff

		// Read index for package.
		nameIndex := make(map[string]uint64)
		nSyms := r.uint64()
		// In shallow mode, only the current package (i=0) has an index.
		assert(!(shallow && i > 0 && nSyms != 0))
		for ; nSyms > 0; nSyms-- {
			name := p.stringAt(r.uint64())
			nameIndex[name] = r.uint64()
		}

		items[i].nameIndex = nameIndex
	}

	// Request packages all at once from the client,
	// enabling a parallel implementation.
	if err := getPackages(items); err != nil {
		return nil, err // don't wrap this error
	}

	// Check the results and complete the index.
	pkgList := make([]*types.Package, len(items))
	for i, item := range items {
		pkg := item.Pkg
		if pkg == nil {
			errorf("internal error: getPackages returned nil package for %q", item.Path)
		} else if pkg.Path() != item.Path {
			errorf("internal error: getPackages returned wrong path %q, want %q", pkg.Path(), item.Path)
		} else if pkg.Name() != item.Name {
			errorf("internal error: getPackages returned wrong name %s for package %q, want %s", pkg.Name(), item.Path, item.Name)
		}
		p.pkgCache[item.pathOffset] = pkg
		p.pkgIndex[pkg] = item.nameIndex
		pkgList[i] = pkg
	}

	if bundle {
		pkgs = make([]*types.Package, r.uint64())
		for i := range pkgs {
			pkg := p.pkgAt(r.uint64())
			imps := make([]*types.Package, r.uint64())
			for j := range imps {
				imps[j] = p.pkgAt(r.uint64())
			}
			pkg.SetImports(imps)
			pkgs[i] = pkg
		}
	} else {
		if len(pkgList) == 0 {
			errorf("no packages found for %s", path)
			panic("unreachable")
		}
		pkgs = pkgList[:1]

		// record all referenced packages as imports
		list := append(([]*types.Package)(nil), pkgList[1:]...)
		sort.Sort(byPath(list))
		pkgs[0].SetImports(list)
	}

	for _, pkg := range pkgs {
		if pkg.Complete() {
			continue
		}

		names := make([]string, 0, len(p.pkgIndex[pkg]))
		for name := range p.pkgIndex[pkg] {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			p.doDecl(pkg, name)
		}

		// package was imported completely and without errors
		pkg.MarkComplete()
	}

	// SetConstraint can't be called if the constraint type is not yet complete.
	// When type params are created in the 'P' case of (*importReader).obj(),
	// the associated constraint type may not be complete due to recursion.
	// Therefore, we defer calling SetConstraint there, and call it here instead
	// after all types are complete.
	for _, d := range p.later {
		typeparams.SetTypeParamConstraint(d.t, d.constraint)
	}

	for _, typ := range p.interfaceList {
		typ.Complete()
	}

	return pkgs, nil
}

type setConstraintArgs struct {
	t          *typeparams.TypeParam
	constraint types.Type
}

type iimporter struct {
	version int
	ipath   string

	usePosv2 bool

	stringData  []byte
	stringCache map[uint64]string
	fileOffset  []uint64 // fileOffset[i] is offset in fileData for info about file encoded as i
	fileData    []byte
	fileCache   []*token.File // memoized decoding of file encoded as i
	pkgCache    map[uint64]*types.Package

	declData    []byte
	pkgIndex    map[*types.Package]map[string]uint64
	typCache    map[uint64]types.Type
	tparamIndex map[ident]types.Type

	fake          fakeFileSet
	interfaceList []*types.Interface

	// Arguments for calls to SetConstraint that are deferred due to recursive types
	later []setConstraintArgs

	indent int // for tracing support
}

func (p *iimporter) trace(format string, args ...interface{}) {
	if !trace {
		// Call sites should also be guarded, but having this check here allows
		// easily enabling/disabling debug trace statements.
		return
	}
	fmt.Printf(strings.Repeat("..", p.indent)+format+"\n", args...)
}

func (p *iimporter) doDecl(pkg *types.Package, name string) {
	if debug {
		p.trace("import decl %s", name)
		p.indent++
		defer func() {
			p.indent--
			p.trace("=> %s", name)
		}()
	}
	// See if we've already imported this declaration.
	if obj := pkg.Scope().Lookup(name); obj != nil {
		return
	}

	off, ok := p.pkgIndex[pkg][name]
	if !ok {
		// In deep mode, the index should be complete. In shallow
		// mode, we should have already recursively loaded necessary
		// dependencies so the above Lookup succeeds.
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

func (p *iimporter) fileAt(index uint64) *token.File {
	file := p.fileCache[index]
	if file == nil {
		off := p.fileOffset[index]
		file = p.decodeFile(intReader{bytes.NewReader(p.fileData[off:]), p.ipath})
		p.fileCache[index] = file
	}
	return file
}

func (p *iimporter) decodeFile(rd intReader) *token.File {
	filename := p.stringAt(rd.uint64())
	size := int(rd.uint64())
	file := p.fake.fset.AddFile(filename, -1, size)

	// SetLines requires a nondecreasing sequence.
	// Because it is common for clients to derive the interval
	// [start, start+len(name)] from a start position, and we
	// want to ensure that the end offset is on the same line,
	// we fill in the gaps of the sparse encoding with values
	// that strictly increase by the largest possible amount.
	// This allows us to avoid having to record the actual end
	// offset of each needed line.

	lines := make([]int, int(rd.uint64()))
	var index, offset int
	for i, n := 0, int(rd.uint64()); i < n; i++ {
		index += int(rd.uint64())
		offset += int(rd.uint64())
		lines[index] = offset

		// Ensure monotonicity between points.
		for j := index - 1; j > 0 && lines[j] == 0; j-- {
			lines[j] = lines[j+1] - 1
		}
	}

	// Ensure monotonicity after last point.
	for j := len(lines) - 1; j > 0 && lines[j] == 0; j-- {
		size--
		lines[j] = size
	}

	if !file.SetLines(lines) {
		errorf("SetLines failed: %d", lines) // can't happen
	}
	return file
}

func (p *iimporter) pkgAt(off uint64) *types.Package {
	if pkg, ok := p.pkgCache[off]; ok {
		return pkg
	}
	path := p.stringAt(off)
	errorf("missing package %q in %q", path, p.ipath)
	return nil
}

func (p *iimporter) typAt(off uint64, base *types.Named) types.Type {
	if t, ok := p.typCache[off]; ok && canReuse(base, t) {
		return t
	}

	if off < predeclReserved {
		errorf("predeclared type missing from cache: %v", off)
	}

	r := &importReader{p: p}
	r.declReader.Reset(p.declData[off-predeclReserved:])
	t := r.doType(base)

	if canReuse(base, t) {
		p.typCache[off] = t
	}
	return t
}

// canReuse reports whether the type rhs on the RHS of the declaration for def
// may be re-used.
//
// Specifically, if def is non-nil and rhs is an interface type with methods, it
// may not be re-used because we have a convention of setting the receiver type
// for interface methods to def.
func canReuse(def *types.Named, rhs types.Type) bool {
	if def == nil {
		return true
	}
	iface, _ := rhs.(*types.Interface)
	if iface == nil {
		return true
	}
	// Don't use iface.Empty() here as iface may not be complete.
	return iface.NumEmbeddeds() == 0 && iface.NumExplicitMethods() == 0
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

	case 'F', 'G':
		var tparams []*typeparams.TypeParam
		if tag == 'G' {
			tparams = r.tparamList()
		}
		sig := r.signature(nil, nil, tparams)
		r.declare(types.NewFunc(pos, r.currPkg, name, sig))

	case 'T', 'U':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		obj := types.NewTypeName(pos, r.currPkg, name, nil)
		named := types.NewNamed(obj, nil, nil)
		// Declare obj before calling r.tparamList, so the new type name is recognized
		// if used in the constraint of one of its own typeparams (see #48280).
		r.declare(obj)
		if tag == 'U' {
			tparams := r.tparamList()
			typeparams.SetForNamed(named, tparams)
		}

		underlying := r.p.typAt(r.uint64(), named).Underlying()
		named.SetUnderlying(underlying)

		if !isInterface(underlying) {
			for n := r.uint64(); n > 0; n-- {
				mpos := r.pos()
				mname := r.ident()
				recv := r.param()

				// If the receiver has any targs, set those as the
				// rparams of the method (since those are the
				// typeparams being used in the method sig/body).
				base := baseType(recv.Type())
				assert(base != nil)
				targs := typeparams.NamedTypeArgs(base)
				var rparams []*typeparams.TypeParam
				if targs.Len() > 0 {
					rparams = make([]*typeparams.TypeParam, targs.Len())
					for i := range rparams {
						rparams[i] = targs.At(i).(*typeparams.TypeParam)
					}
				}
				msig := r.signature(recv, rparams, nil)

				named.AddMethod(types.NewFunc(mpos, r.currPkg, mname, msig))
			}
		}

	case 'P':
		// We need to "declare" a typeparam in order to have a name that
		// can be referenced recursively (if needed) in the type param's
		// bound.
		if r.p.version < iexportVersionGenerics {
			errorf("unexpected type param type")
		}
		name0 := tparamName(name)
		tn := types.NewTypeName(pos, r.currPkg, name0, nil)
		t := typeparams.NewTypeParam(tn, nil)

		// To handle recursive references to the typeparam within its
		// bound, save the partial type in tparamIndex before reading the bounds.
		id := ident{r.currPkg, name}
		r.p.tparamIndex[id] = t
		var implicit bool
		if r.p.version >= iexportVersionGo1_18 {
			implicit = r.bool()
		}
		constraint := r.typ()
		if implicit {
			iface, _ := constraint.(*types.Interface)
			if iface == nil {
				errorf("non-interface constraint marked implicit")
			}
			typeparams.MarkImplicit(iface)
		}
		// The constraint type may not be complete, if we
		// are in the middle of a type recursion involving type
		// constraints. So, we defer SetConstraint until we have
		// completely set up all types in ImportData.
		r.p.later = append(r.p.later, setConstraintArgs{t: t, constraint: constraint})

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
	if r.p.version >= iexportVersionGo1_18 {
		// TODO: add support for using the kind.
		_ = constant.Kind(r.int64())
	}

	switch b := typ.Underlying().(*types.Basic); b.Info() & types.IsConstType {
	case types.IsBoolean:
		val = constant.MakeBool(r.bool())

	case types.IsString:
		val = constant.MakeString(r.string())

	case types.IsInteger:
		var x big.Int
		r.mpint(&x, b)
		val = constant.Make(&x)

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

func (r *importReader) mpint(x *big.Int, typ *types.Basic) {
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

func (r *importReader) mpfloat(typ *types.Basic) constant.Value {
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

func (r *importReader) qualifiedIdent() (*types.Package, string) {
	name := r.string()
	pkg := r.pkg()
	return pkg, name
}

func (r *importReader) pos() token.Pos {
	if r.p.usePosv2 {
		return r.posv2()
	}
	if r.p.version >= iexportVersionPosCol {
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

func (r *importReader) posv2() token.Pos {
	file := r.uint64()
	if file == 0 {
		return token.NoPos
	}
	tf := r.p.fileAt(file - 1)
	return tf.Pos(int(r.uint64()))
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

func (r *importReader) doType(base *types.Named) (res types.Type) {
	k := r.kind()
	if debug {
		r.p.trace("importing type %d (base: %s)", k, base)
		r.p.indent++
		defer func() {
			r.p.indent--
			r.p.trace("=> %s", res)
		}()
	}
	switch k {
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
		return r.signature(nil, nil, nil)

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

			msig := r.signature(recv, nil, nil)
			methods[i] = types.NewFunc(mpos, r.currPkg, mname, msig)
		}

		typ := newInterface(methods, embeddeds)
		r.p.interfaceList = append(r.p.interfaceList, typ)
		return typ

	case typeParamType:
		if r.p.version < iexportVersionGenerics {
			errorf("unexpected type param type")
		}
		pkg, name := r.qualifiedIdent()
		id := ident{pkg, name}
		if t, ok := r.p.tparamIndex[id]; ok {
			// We're already in the process of importing this typeparam.
			return t
		}
		// Otherwise, import the definition of the typeparam now.
		r.p.doDecl(pkg, name)
		return r.p.tparamIndex[id]

	case instanceType:
		if r.p.version < iexportVersionGenerics {
			errorf("unexpected instantiation type")
		}
		// pos does not matter for instances: they are positioned on the original
		// type.
		_ = r.pos()
		len := r.uint64()
		targs := make([]types.Type, len)
		for i := range targs {
			targs[i] = r.typ()
		}
		baseType := r.typ()
		// The imported instantiated type doesn't include any methods, so
		// we must always use the methods of the base (orig) type.
		// TODO provide a non-nil *Environment
		t, _ := typeparams.Instantiate(nil, baseType, targs, false)
		return t

	case unionType:
		if r.p.version < iexportVersionGenerics {
			errorf("unexpected instantiation type")
		}
		terms := make([]*typeparams.Term, r.uint64())
		for i := range terms {
			terms[i] = typeparams.NewTerm(r.bool(), r.typ())
		}
		return typeparams.NewUnion(terms)
	}
}

func (r *importReader) kind() itag {
	return itag(r.uint64())
}

func (r *importReader) signature(recv *types.Var, rparams []*typeparams.TypeParam, tparams []*typeparams.TypeParam) *types.Signature {
	params := r.paramList()
	results := r.paramList()
	variadic := params.Len() > 0 && r.bool()
	return typeparams.NewSignatureType(recv, rparams, tparams, params, results, variadic)
}

func (r *importReader) tparamList() []*typeparams.TypeParam {
	n := r.uint64()
	if n == 0 {
		return nil
	}
	xs := make([]*typeparams.TypeParam, n)
	for i := range xs {
		// Note: the standard library importer is tolerant of nil types here,
		// though would panic in SetTypeParams.
		xs[i] = r.typ().(*typeparams.TypeParam)
	}
	return xs
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

func baseType(typ types.Type) *types.Named {
	// pointer receivers are never types.Named types
	if p, _ := typ.(*types.Pointer); p != nil {
		typ = p.Elem()
	}
	// receiver base types are always (possibly generic) types.Named types
	n, _ := typ.(*types.Named)
	return n
}
