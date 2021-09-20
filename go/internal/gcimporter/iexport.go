// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed binary package export.
// This file was derived from $GOROOT/src/cmd/compile/internal/gc/iexport.go;
// see that file for specification of the format.

package gcimporter

import (
	"bytes"
	"encoding/binary"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"math/big"
	"reflect"
	"sort"

	"golang.org/x/tools/internal/typeparams"
)

// Current bundled export format version. Increase with each format change.
// 0: initial implementation
const bundleVersion = 0

// IExportData writes indexed export data for pkg to out.
//
// If no file set is provided, position info will be missing.
// The package path of the top-level package will not be recorded,
// so that calls to IImportData can override with a provided package path.
func IExportData(out io.Writer, fset *token.FileSet, pkg *types.Package) error {
	return iexportCommon(out, fset, false, []*types.Package{pkg})
}

// IExportBundle writes an indexed export bundle for pkgs to out.
func IExportBundle(out io.Writer, fset *token.FileSet, pkgs []*types.Package) error {
	return iexportCommon(out, fset, true, pkgs)
}

func iexportCommon(out io.Writer, fset *token.FileSet, bundle bool, pkgs []*types.Package) (err error) {
	if !debug {
		defer func() {
			if e := recover(); e != nil {
				if ierr, ok := e.(internalError); ok {
					err = ierr
					return
				}
				// Not an internal error; panic again.
				panic(e)
			}
		}()
	}

	p := iexporter{
		fset:        fset,
		allPkgs:     map[*types.Package]bool{},
		stringIndex: map[string]uint64{},
		declIndex:   map[types.Object]uint64{},
		typIndex:    map[types.Type]uint64{},
	}
	if !bundle {
		p.localpkg = pkgs[0]
	}

	for i, pt := range predeclared() {
		p.typIndex[pt] = uint64(i)
	}
	if len(p.typIndex) > predeclReserved {
		panic(internalErrorf("too many predeclared types: %d > %d", len(p.typIndex), predeclReserved))
	}

	// Initialize work queue with exported declarations.
	for _, pkg := range pkgs {
		scope := pkg.Scope()
		for _, name := range scope.Names() {
			if ast.IsExported(name) {
				p.pushDecl(scope.Lookup(name))
			}
		}

		if bundle {
			// Ensure pkg and its imports are included in the index.
			p.allPkgs[pkg] = true
			for _, imp := range pkg.Imports() {
				p.allPkgs[imp] = true
			}
		}
	}

	// Loop until no more work.
	for !p.declTodo.empty() {
		p.doDecl(p.declTodo.popHead())
	}

	// Append indices to data0 section.
	dataLen := uint64(p.data0.Len())
	w := p.newWriter()
	w.writeIndex(p.declIndex)

	if bundle {
		w.uint64(uint64(len(pkgs)))
		for _, pkg := range pkgs {
			w.pkg(pkg)
			imps := pkg.Imports()
			w.uint64(uint64(len(imps)))
			for _, imp := range imps {
				w.pkg(imp)
			}
		}
	}
	w.flush()

	// Assemble header.
	var hdr intWriter
	if bundle {
		hdr.uint64(bundleVersion)
	}
	hdr.uint64(iexportVersion)
	hdr.uint64(uint64(p.strings.Len()))
	hdr.uint64(dataLen)

	// Flush output.
	io.Copy(out, &hdr)
	io.Copy(out, &p.strings)
	io.Copy(out, &p.data0)

	return nil
}

// writeIndex writes out an object index. mainIndex indicates whether
// we're writing out the main index, which is also read by
// non-compiler tools and includes a complete package description
// (i.e., name and height).
func (w *exportWriter) writeIndex(index map[types.Object]uint64) {
	// Build a map from packages to objects from that package.
	pkgObjs := map[*types.Package][]types.Object{}

	// For the main index, make sure to include every package that
	// we reference, even if we're not exporting (or reexporting)
	// any symbols from it.
	if w.p.localpkg != nil {
		pkgObjs[w.p.localpkg] = nil
	}
	for pkg := range w.p.allPkgs {
		pkgObjs[pkg] = nil
	}

	for obj := range index {
		pkgObjs[obj.Pkg()] = append(pkgObjs[obj.Pkg()], obj)
	}

	var pkgs []*types.Package
	for pkg, objs := range pkgObjs {
		pkgs = append(pkgs, pkg)

		sort.Slice(objs, func(i, j int) bool {
			return indexName(objs[i]) < indexName(objs[j])
		})
	}

	sort.Slice(pkgs, func(i, j int) bool {
		return w.exportPath(pkgs[i]) < w.exportPath(pkgs[j])
	})

	w.uint64(uint64(len(pkgs)))
	for _, pkg := range pkgs {
		w.string(w.exportPath(pkg))
		w.string(pkg.Name())
		w.uint64(uint64(0)) // package height is not needed for go/types

		objs := pkgObjs[pkg]
		w.uint64(uint64(len(objs)))
		for _, obj := range objs {
			w.string(indexName(obj))
			w.uint64(index[obj])
		}
	}
}

// indexName returns the 'indexed' name of an object. It differs from
// obj.Name() only for type parameter names, where we include the subscripted
// type parameter ID.
//
// TODO(rfindley): remove this once we no longer need subscripts.
func indexName(obj types.Object) (res string) {
	if _, ok := obj.(*types.TypeName); ok {
		if tparam, ok := obj.Type().(*typeparams.TypeParam); ok {
			return types.TypeString(tparam, func(*types.Package) string { return "" })
		}
	}
	return obj.Name()
}

type iexporter struct {
	fset *token.FileSet
	out  *bytes.Buffer

	localpkg *types.Package

	// allPkgs tracks all packages that have been referenced by
	// the export data, so we can ensure to include them in the
	// main index.
	allPkgs map[*types.Package]bool

	declTodo objQueue

	strings     intWriter
	stringIndex map[string]uint64

	data0     intWriter
	declIndex map[types.Object]uint64
	typIndex  map[types.Type]uint64
}

// stringOff returns the offset of s within the string section.
// If not already present, it's added to the end.
func (p *iexporter) stringOff(s string) uint64 {
	off, ok := p.stringIndex[s]
	if !ok {
		off = uint64(p.strings.Len())
		p.stringIndex[s] = off

		p.strings.uint64(uint64(len(s)))
		p.strings.WriteString(s)
	}
	return off
}

// pushDecl adds n to the declaration work queue, if not already present.
func (p *iexporter) pushDecl(obj types.Object) {
	// Package unsafe is known to the compiler and predeclared.
	assert(obj.Pkg() != types.Unsafe)

	if _, ok := p.declIndex[obj]; ok {
		return
	}

	p.declIndex[obj] = ^uint64(0) // mark n present in work queue
	p.declTodo.pushTail(obj)
}

// exportWriter handles writing out individual data section chunks.
type exportWriter struct {
	p *iexporter

	data       intWriter
	currPkg    *types.Package
	prevFile   string
	prevLine   int64
	prevColumn int64
}

func (w *exportWriter) exportPath(pkg *types.Package) string {
	if pkg == w.p.localpkg {
		return ""
	}
	return pkg.Path()
}

func (p *iexporter) doDecl(obj types.Object) {
	w := p.newWriter()
	w.setPkg(obj.Pkg(), false)

	switch obj := obj.(type) {
	case *types.Var:
		w.tag('V')
		w.pos(obj.Pos())
		w.typ(obj.Type(), obj.Pkg())

	case *types.Func:
		sig, _ := obj.Type().(*types.Signature)
		if sig.Recv() != nil {
			panic(internalErrorf("unexpected method: %v", sig))
		}

		// Function.
		if typeparams.ForSignature(sig).Len() == 0 {
			w.tag('F')
		} else {
			w.tag('G')
		}
		w.pos(obj.Pos())
		// The tparam list of the function type is the
		// declaration of the type params. So, write out the type
		// params right now. Then those type params will be
		// referenced via their type offset (via typOff) in all
		// other places in the signature and function that they
		// are used.
		if tparams := typeparams.ForSignature(sig); tparams.Len() > 0 {
			w.tparamList(tparams, obj.Pkg())
		}
		w.signature(sig)

	case *types.Const:
		w.tag('C')
		w.pos(obj.Pos())
		w.value(obj.Type(), obj.Val())

	case *types.TypeName:
		t := obj.Type()

		if tparam, ok := t.(*typeparams.TypeParam); ok {
			w.tag('P')
			w.pos(obj.Pos())
			w.typ(tparam.Constraint(), obj.Pkg())
			break
		}

		if obj.IsAlias() {
			w.tag('A')
			w.pos(obj.Pos())
			w.typ(t, obj.Pkg())
			break
		}

		// Defined type.
		named, ok := t.(*types.Named)
		if !ok {
			panic(internalErrorf("%s is not a defined type", t))
		}

		if typeparams.ForNamed(named).Len() == 0 {
			w.tag('T')
		} else {
			w.tag('U')
		}
		w.pos(obj.Pos())

		if typeparams.ForNamed(named).Len() > 0 {
			w.tparamList(typeparams.ForNamed(named), obj.Pkg())
		}

		underlying := obj.Type().Underlying()
		w.typ(underlying, obj.Pkg())

		if types.IsInterface(t) {
			break
		}

		n := named.NumMethods()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			m := named.Method(i)
			w.pos(m.Pos())
			w.string(m.Name())
			sig, _ := m.Type().(*types.Signature)
			w.param(sig.Recv())
			w.signature(sig)
		}

	default:
		panic(internalErrorf("unexpected object: %v", obj))
	}

	p.declIndex[obj] = w.flush()
}

func (w *exportWriter) tag(tag byte) {
	w.data.WriteByte(tag)
}

func (w *exportWriter) pos(pos token.Pos) {
	if iexportVersion >= iexportVersionPosCol {
		w.posV1(pos)
	} else {
		w.posV0(pos)
	}
}

func (w *exportWriter) posV1(pos token.Pos) {
	if w.p.fset == nil {
		w.int64(0)
		return
	}

	p := w.p.fset.Position(pos)
	file := p.Filename
	line := int64(p.Line)
	column := int64(p.Column)

	deltaColumn := (column - w.prevColumn) << 1
	deltaLine := (line - w.prevLine) << 1

	if file != w.prevFile {
		deltaLine |= 1
	}
	if deltaLine != 0 {
		deltaColumn |= 1
	}

	w.int64(deltaColumn)
	if deltaColumn&1 != 0 {
		w.int64(deltaLine)
		if deltaLine&1 != 0 {
			w.string(file)
		}
	}

	w.prevFile = file
	w.prevLine = line
	w.prevColumn = column
}

func (w *exportWriter) posV0(pos token.Pos) {
	if w.p.fset == nil {
		w.int64(0)
		return
	}

	p := w.p.fset.Position(pos)
	file := p.Filename
	line := int64(p.Line)

	// When file is the same as the last position (common case),
	// we can save a few bytes by delta encoding just the line
	// number.
	//
	// Note: Because data objects may be read out of order (or not
	// at all), we can only apply delta encoding within a single
	// object. This is handled implicitly by tracking prevFile and
	// prevLine as fields of exportWriter.

	if file == w.prevFile {
		delta := line - w.prevLine
		w.int64(delta)
		if delta == deltaNewFile {
			w.int64(-1)
		}
	} else {
		w.int64(deltaNewFile)
		w.int64(line) // line >= 0
		w.string(file)
		w.prevFile = file
	}
	w.prevLine = line
}

func (w *exportWriter) pkg(pkg *types.Package) {
	// Ensure any referenced packages are declared in the main index.
	w.p.allPkgs[pkg] = true

	w.string(w.exportPath(pkg))
}

func (w *exportWriter) qualifiedIdent(obj types.Object) {
	// Ensure any referenced declarations are written out too.
	w.p.pushDecl(obj)
	w.string(indexName(obj))
	w.pkg(obj.Pkg())
}

func (w *exportWriter) typ(t types.Type, pkg *types.Package) {
	w.data.uint64(w.p.typOff(t, pkg))
}

func (p *iexporter) newWriter() *exportWriter {
	return &exportWriter{p: p}
}

func (w *exportWriter) flush() uint64 {
	off := uint64(w.p.data0.Len())
	io.Copy(&w.p.data0, &w.data)
	return off
}

func (p *iexporter) typOff(t types.Type, pkg *types.Package) uint64 {
	off, ok := p.typIndex[t]
	if !ok {
		w := p.newWriter()
		w.doTyp(t, pkg)
		off = predeclReserved + w.flush()
		p.typIndex[t] = off
	}
	return off
}

func (w *exportWriter) startType(k itag) {
	w.data.uint64(uint64(k))
}

func (w *exportWriter) doTyp(t types.Type, pkg *types.Package) {
	switch t := t.(type) {
	case *types.Named:
		if targs := typeparams.NamedTypeArgs(t); targs.Len() > 0 {
			w.startType(instanceType)
			// TODO(rfindley): investigate if this position is correct, and if it
			// matters.
			w.pos(t.Obj().Pos())
			w.typeList(targs, pkg)
			w.typ(typeparams.NamedTypeOrigin(t), pkg)
			return
		}
		w.startType(definedType)
		w.qualifiedIdent(t.Obj())

	case *typeparams.TypeParam:
		w.startType(typeParamType)
		w.qualifiedIdent(t.Obj())

	case *types.Pointer:
		w.startType(pointerType)
		w.typ(t.Elem(), pkg)

	case *types.Slice:
		w.startType(sliceType)
		w.typ(t.Elem(), pkg)

	case *types.Array:
		w.startType(arrayType)
		w.uint64(uint64(t.Len()))
		w.typ(t.Elem(), pkg)

	case *types.Chan:
		w.startType(chanType)
		// 1 RecvOnly; 2 SendOnly; 3 SendRecv
		var dir uint64
		switch t.Dir() {
		case types.RecvOnly:
			dir = 1
		case types.SendOnly:
			dir = 2
		case types.SendRecv:
			dir = 3
		}
		w.uint64(dir)
		w.typ(t.Elem(), pkg)

	case *types.Map:
		w.startType(mapType)
		w.typ(t.Key(), pkg)
		w.typ(t.Elem(), pkg)

	case *types.Signature:
		w.startType(signatureType)
		w.setPkg(pkg, true)
		w.signature(t)

	case *types.Struct:
		w.startType(structType)
		w.setPkg(pkg, true)

		n := t.NumFields()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			f := t.Field(i)
			w.pos(f.Pos())
			w.string(f.Name())
			w.typ(f.Type(), pkg)
			w.bool(f.Anonymous())
			w.string(t.Tag(i)) // note (or tag)
		}

	case *types.Interface:
		w.startType(interfaceType)
		w.setPkg(pkg, true)

		n := t.NumEmbeddeds()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			ft := t.EmbeddedType(i)
			tPkg := pkg
			if named, _ := ft.(*types.Named); named != nil {
				w.pos(named.Obj().Pos())
			} else {
				w.pos(token.NoPos)
			}
			w.typ(ft, tPkg)
		}

		n = t.NumExplicitMethods()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			m := t.ExplicitMethod(i)
			w.pos(m.Pos())
			w.string(m.Name())
			sig, _ := m.Type().(*types.Signature)
			w.signature(sig)
		}

	case *typeparams.Union:
		w.startType(unionType)
		nt := t.Len()
		w.uint64(uint64(nt))
		for i := 0; i < nt; i++ {
			term := t.Term(i)
			w.bool(term.Tilde())
			w.typ(term.Type(), pkg)
		}

	default:
		panic(internalErrorf("unexpected type: %v, %v", t, reflect.TypeOf(t)))
	}
}

func (w *exportWriter) setPkg(pkg *types.Package, write bool) {
	if write {
		w.pkg(pkg)
	}

	w.currPkg = pkg
}

func (w *exportWriter) signature(sig *types.Signature) {
	w.paramList(sig.Params())
	w.paramList(sig.Results())
	if sig.Params().Len() > 0 {
		w.bool(sig.Variadic())
	}
}

func (w *exportWriter) typeList(ts *typeparams.TypeList, pkg *types.Package) {
	w.uint64(uint64(ts.Len()))
	for i := 0; i < ts.Len(); i++ {
		w.typ(ts.At(i), pkg)
	}
}

func (w *exportWriter) tparamList(list *typeparams.TypeParamList, pkg *types.Package) {
	ll := uint64(list.Len())
	w.uint64(ll)
	for i := 0; i < list.Len(); i++ {
		w.typ(list.At(i), pkg)
	}
}

func (w *exportWriter) paramList(tup *types.Tuple) {
	n := tup.Len()
	w.uint64(uint64(n))
	for i := 0; i < n; i++ {
		w.param(tup.At(i))
	}
}

func (w *exportWriter) param(obj types.Object) {
	w.pos(obj.Pos())
	w.localIdent(obj)
	w.typ(obj.Type(), obj.Pkg())
}

func (w *exportWriter) value(typ types.Type, v constant.Value) {
	w.typ(typ, nil)

	switch b := typ.Underlying().(*types.Basic); b.Info() & types.IsConstType {
	case types.IsBoolean:
		w.bool(constant.BoolVal(v))
	case types.IsInteger:
		var i big.Int
		if i64, exact := constant.Int64Val(v); exact {
			i.SetInt64(i64)
		} else if ui64, exact := constant.Uint64Val(v); exact {
			i.SetUint64(ui64)
		} else {
			i.SetString(v.ExactString(), 10)
		}
		w.mpint(&i, typ)
	case types.IsFloat:
		f := constantToFloat(v)
		w.mpfloat(f, typ)
	case types.IsComplex:
		w.mpfloat(constantToFloat(constant.Real(v)), typ)
		w.mpfloat(constantToFloat(constant.Imag(v)), typ)
	case types.IsString:
		w.string(constant.StringVal(v))
	default:
		if b.Kind() == types.Invalid {
			// package contains type errors
			break
		}
		panic(internalErrorf("unexpected type %v (%v)", typ, typ.Underlying()))
	}
}

// constantToFloat converts a constant.Value with kind constant.Float to a
// big.Float.
func constantToFloat(x constant.Value) *big.Float {
	x = constant.ToFloat(x)
	// Use the same floating-point precision (512) as cmd/compile
	// (see Mpprec in cmd/compile/internal/gc/mpfloat.go).
	const mpprec = 512
	var f big.Float
	f.SetPrec(mpprec)
	if v, exact := constant.Float64Val(x); exact {
		// float64
		f.SetFloat64(v)
	} else if num, denom := constant.Num(x), constant.Denom(x); num.Kind() == constant.Int {
		// TODO(gri): add big.Rat accessor to constant.Value.
		n := valueToRat(num)
		d := valueToRat(denom)
		f.SetRat(n.Quo(n, d))
	} else {
		// Value too large to represent as a fraction => inaccessible.
		// TODO(gri): add big.Float accessor to constant.Value.
		_, ok := f.SetString(x.ExactString())
		assert(ok)
	}
	return &f
}

// mpint exports a multi-precision integer.
//
// For unsigned types, small values are written out as a single
// byte. Larger values are written out as a length-prefixed big-endian
// byte string, where the length prefix is encoded as its complement.
// For example, bytes 0, 1, and 2 directly represent the integer
// values 0, 1, and 2; while bytes 255, 254, and 253 indicate a 1-,
// 2-, and 3-byte big-endian string follow.
//
// Encoding for signed types use the same general approach as for
// unsigned types, except small values use zig-zag encoding and the
// bottom bit of length prefix byte for large values is reserved as a
// sign bit.
//
// The exact boundary between small and large encodings varies
// according to the maximum number of bytes needed to encode a value
// of type typ. As a special case, 8-bit types are always encoded as a
// single byte.
//
// TODO(mdempsky): Is this level of complexity really worthwhile?
func (w *exportWriter) mpint(x *big.Int, typ types.Type) {
	basic, ok := typ.Underlying().(*types.Basic)
	if !ok {
		panic(internalErrorf("unexpected type %v (%T)", typ.Underlying(), typ.Underlying()))
	}

	signed, maxBytes := intSize(basic)

	negative := x.Sign() < 0
	if !signed && negative {
		panic(internalErrorf("negative unsigned integer; type %v, value %v", typ, x))
	}

	b := x.Bytes()
	if len(b) > 0 && b[0] == 0 {
		panic(internalErrorf("leading zeros"))
	}
	if uint(len(b)) > maxBytes {
		panic(internalErrorf("bad mpint length: %d > %d (type %v, value %v)", len(b), maxBytes, typ, x))
	}

	maxSmall := 256 - maxBytes
	if signed {
		maxSmall = 256 - 2*maxBytes
	}
	if maxBytes == 1 {
		maxSmall = 256
	}

	// Check if x can use small value encoding.
	if len(b) <= 1 {
		var ux uint
		if len(b) == 1 {
			ux = uint(b[0])
		}
		if signed {
			ux <<= 1
			if negative {
				ux--
			}
		}
		if ux < maxSmall {
			w.data.WriteByte(byte(ux))
			return
		}
	}

	n := 256 - uint(len(b))
	if signed {
		n = 256 - 2*uint(len(b))
		if negative {
			n |= 1
		}
	}
	if n < maxSmall || n >= 256 {
		panic(internalErrorf("encoding mistake: %d, %v, %v => %d", len(b), signed, negative, n))
	}

	w.data.WriteByte(byte(n))
	w.data.Write(b)
}

// mpfloat exports a multi-precision floating point number.
//
// The number's value is decomposed into mantissa × 2**exponent, where
// mantissa is an integer. The value is written out as mantissa (as a
// multi-precision integer) and then the exponent, except exponent is
// omitted if mantissa is zero.
func (w *exportWriter) mpfloat(f *big.Float, typ types.Type) {
	if f.IsInf() {
		panic("infinite constant")
	}

	// Break into f = mant × 2**exp, with 0.5 <= mant < 1.
	var mant big.Float
	exp := int64(f.MantExp(&mant))

	// Scale so that mant is an integer.
	prec := mant.MinPrec()
	mant.SetMantExp(&mant, int(prec))
	exp -= int64(prec)

	manti, acc := mant.Int(nil)
	if acc != big.Exact {
		panic(internalErrorf("mantissa scaling failed for %f (%s)", f, acc))
	}
	w.mpint(manti, typ)
	if manti.Sign() != 0 {
		w.int64(exp)
	}
}

func (w *exportWriter) bool(b bool) bool {
	var x uint64
	if b {
		x = 1
	}
	w.uint64(x)
	return b
}

func (w *exportWriter) int64(x int64)   { w.data.int64(x) }
func (w *exportWriter) uint64(x uint64) { w.data.uint64(x) }
func (w *exportWriter) string(s string) { w.uint64(w.p.stringOff(s)) }

func (w *exportWriter) localIdent(obj types.Object) {
	// Anonymous parameters.
	if obj == nil {
		w.string("")
		return
	}

	name := indexName(obj)
	if name == "_" {
		w.string("_")
		return
	}

	w.string(name)
}

type intWriter struct {
	bytes.Buffer
}

func (w *intWriter) int64(x int64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutVarint(buf[:], x)
	w.Write(buf[:n])
}

func (w *intWriter) uint64(x uint64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(buf[:], x)
	w.Write(buf[:n])
}

func assert(cond bool) {
	if !cond {
		panic("internal error: assertion failed")
	}
}

// The below is copied from go/src/cmd/compile/internal/gc/syntax.go.

// objQueue is a FIFO queue of types.Object. The zero value of objQueue is
// a ready-to-use empty queue.
type objQueue struct {
	ring       []types.Object
	head, tail int
}

// empty returns true if q contains no Nodes.
func (q *objQueue) empty() bool {
	return q.head == q.tail
}

// pushTail appends n to the tail of the queue.
func (q *objQueue) pushTail(obj types.Object) {
	if len(q.ring) == 0 {
		q.ring = make([]types.Object, 16)
	} else if q.head+len(q.ring) == q.tail {
		// Grow the ring.
		nring := make([]types.Object, len(q.ring)*2)
		// Copy the old elements.
		part := q.ring[q.head%len(q.ring):]
		if q.tail-q.head <= len(part) {
			part = part[:q.tail-q.head]
			copy(nring, part)
		} else {
			pos := copy(nring, part)
			copy(nring[pos:], q.ring[:q.tail%len(q.ring)])
		}
		q.ring, q.head, q.tail = nring, 0, q.tail-q.head
	}

	q.ring[q.tail%len(q.ring)] = obj
	q.tail++
}

// popHead pops a node from the head of the queue. It panics if q is empty.
func (q *objQueue) popHead() types.Object {
	if q.empty() {
		panic("dequeue empty")
	}
	obj := q.ring[q.head%len(q.ring)]
	q.head++
	return obj
}
