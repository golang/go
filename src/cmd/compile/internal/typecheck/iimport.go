// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See iexport.go for the export data format.

package typecheck

import (
	"encoding/binary"
	"fmt"
	"go/constant"
	"io"
	"math/big"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// An iimporterAndOffset identifies an importer and an offset within
// its data section.
type iimporterAndOffset struct {
	p   *iimporter
	off uint64
}

var (
	// DeclImporter maps from imported identifiers to an importer
	// and offset where that identifier's declaration can be read.
	DeclImporter = map[*types.Sym]iimporterAndOffset{}

	// inlineImporter is like DeclImporter, but for inline bodies
	// for function and method symbols.
	inlineImporter = map[*types.Sym]iimporterAndOffset{}
)

// expandDecl returns immediately if n is already a Name node. Otherwise, n should
// be an Ident node, and expandDecl reads in the definition of the specified
// identifier from the appropriate package.
func expandDecl(n ir.Node) ir.Node {
	if n, ok := n.(*ir.Name); ok {
		return n
	}

	id := n.(*ir.Ident)
	if n := id.Sym().PkgDef(); n != nil {
		return n.(*ir.Name)
	}

	r := importReaderFor(id.Sym(), DeclImporter)
	if r == nil {
		// Can happen if user tries to reference an undeclared name.
		return n
	}

	return r.doDecl(n.Sym())
}

// ImportBody reads in the dcls and body of an imported function (which should not
// yet have been read in).
func ImportBody(fn *ir.Func) {
	if fn.Inl.Body != nil {
		base.Fatalf("%v already has inline body", fn)
	}

	r := importReaderFor(fn.Nname.Sym(), inlineImporter)
	if r == nil {
		base.Fatalf("missing import reader for %v", fn)
	}

	if inimport {
		base.Fatalf("recursive inimport")
	}
	inimport = true
	r.doInline(fn)
	inimport = false
}

// HaveInlineBody reports whether we have fn's inline body available
// for inlining.
//
// It's a function literal so that it can be overriden for
// GOEXPERIMENT=unified.
var HaveInlineBody = func(fn *ir.Func) bool {
	if fn.Inl == nil {
		return false
	}

	if fn.Inl.Body != nil {
		return true
	}

	_, ok := inlineImporter[fn.Nname.Sym()]
	return ok
}

func importReaderFor(sym *types.Sym, importers map[*types.Sym]iimporterAndOffset) *importReader {
	x, ok := importers[sym]
	if !ok {
		return nil
	}

	return x.p.newReader(x.off, sym.Pkg)
}

type intReader struct {
	*strings.Reader
	pkg *types.Pkg
}

func (r *intReader) int64() int64 {
	i, err := binary.ReadVarint(r.Reader)
	if err != nil {
		base.Errorf("import %q: read error: %v", r.pkg.Path, err)
		base.ErrorExit()
	}
	return i
}

func (r *intReader) uint64() uint64 {
	i, err := binary.ReadUvarint(r.Reader)
	if err != nil {
		base.Errorf("import %q: read error: %v", r.pkg.Path, err)
		base.ErrorExit()
	}
	return i
}

func ReadImports(pkg *types.Pkg, data string) {
	ird := &intReader{strings.NewReader(data), pkg}

	version := ird.uint64()
	switch version {
	case iexportVersionGo1_18, iexportVersionPosCol, iexportVersionGo1_11:
	default:
		base.Errorf("import %q: unknown export format version %d", pkg.Path, version)
		base.ErrorExit()
	}

	sLen := int64(ird.uint64())
	dLen := int64(ird.uint64())

	whence, _ := ird.Seek(0, io.SeekCurrent)
	stringData := data[whence : whence+sLen]
	declData := data[whence+sLen : whence+sLen+dLen]
	ird.Seek(sLen+dLen, io.SeekCurrent)

	p := &iimporter{
		exportVersion: version,
		ipkg:          pkg,

		pkgCache:     map[uint64]*types.Pkg{},
		posBaseCache: map[uint64]*src.PosBase{},
		typCache:     map[uint64]*types.Type{},

		stringData: stringData,
		declData:   declData,
	}

	for i, pt := range predeclared() {
		p.typCache[uint64(i)] = pt
	}

	// Declaration index.
	for nPkgs := ird.uint64(); nPkgs > 0; nPkgs-- {
		pkg := p.pkgAt(ird.uint64())
		pkgName := p.stringAt(ird.uint64())
		_ = int(ird.uint64()) // was package height, but not necessary anymore.
		if pkg.Name == "" {
			pkg.Name = pkgName
			types.NumImport[pkgName]++

			// TODO(mdempsky): This belongs somewhere else.
			pkg.Lookup("_").Def = ir.BlankNode
		} else {
			if pkg.Name != pkgName {
				base.Fatalf("conflicting package names %v and %v for path %q", pkg.Name, pkgName, pkg.Path)
			}
		}

		for nSyms := ird.uint64(); nSyms > 0; nSyms-- {
			s := pkg.Lookup(p.stringAt(ird.uint64()))
			off := ird.uint64()

			if _, ok := DeclImporter[s]; !ok {
				DeclImporter[s] = iimporterAndOffset{p, off}
			}
		}
	}

	// Inline body index.
	for nPkgs := ird.uint64(); nPkgs > 0; nPkgs-- {
		pkg := p.pkgAt(ird.uint64())

		for nSyms := ird.uint64(); nSyms > 0; nSyms-- {
			s := pkg.Lookup(p.stringAt(ird.uint64()))
			off := ird.uint64()

			if _, ok := inlineImporter[s]; !ok {
				inlineImporter[s] = iimporterAndOffset{p, off}
			}
		}
	}
}

type iimporter struct {
	exportVersion uint64
	ipkg          *types.Pkg

	pkgCache     map[uint64]*types.Pkg
	posBaseCache map[uint64]*src.PosBase
	typCache     map[uint64]*types.Type

	stringData string
	declData   string
}

func (p *iimporter) stringAt(off uint64) string {
	var x [binary.MaxVarintLen64]byte
	n := copy(x[:], p.stringData[off:])

	slen, n := binary.Uvarint(x[:n])
	if n <= 0 {
		base.Fatalf("varint failed")
	}
	spos := off + uint64(n)
	return p.stringData[spos : spos+slen]
}

func (p *iimporter) posBaseAt(off uint64) *src.PosBase {
	if posBase, ok := p.posBaseCache[off]; ok {
		return posBase
	}

	file := p.stringAt(off)
	posBase := src.NewFileBase(file, file)
	p.posBaseCache[off] = posBase
	return posBase
}

func (p *iimporter) pkgAt(off uint64) *types.Pkg {
	if pkg, ok := p.pkgCache[off]; ok {
		return pkg
	}

	pkg := p.ipkg
	if pkgPath := p.stringAt(off); pkgPath != "" {
		pkg = types.NewPkg(pkgPath, "")
	}
	p.pkgCache[off] = pkg
	return pkg
}

// An importReader keeps state for reading an individual imported
// object (declaration or inline body).
type importReader struct {
	strings.Reader
	p *iimporter

	currPkg    *types.Pkg
	prevBase   *src.PosBase
	prevLine   int64
	prevColumn int64

	// curfn is the current function we're importing into.
	curfn *ir.Func
	// Slice of all dcls for function, including any interior closures
	allDcls        []*ir.Name
	allClosureVars []*ir.Name
	autotmpgen     int
}

func (p *iimporter) newReader(off uint64, pkg *types.Pkg) *importReader {
	r := &importReader{
		p:       p,
		currPkg: pkg,
	}
	// (*strings.Reader).Reset wasn't added until Go 1.7, and we
	// need to build with Go 1.4.
	r.Reader = *strings.NewReader(p.declData[off:])
	return r
}

func (r *importReader) string() string        { return r.p.stringAt(r.uint64()) }
func (r *importReader) posBase() *src.PosBase { return r.p.posBaseAt(r.uint64()) }
func (r *importReader) pkg() *types.Pkg       { return r.p.pkgAt(r.uint64()) }

func (r *importReader) setPkg() {
	r.currPkg = r.pkg()
}

func (r *importReader) doDecl(sym *types.Sym) *ir.Name {
	tag := r.byte()
	pos := r.pos()

	switch tag {
	case 'A':
		typ := r.typ()

		return importalias(pos, sym, typ)

	case 'C':
		typ := r.typ()
		val := r.value(typ)

		n := importconst(pos, sym, typ, val)
		r.constExt(n)
		return n

	case 'F', 'G':
		var tparams []*types.Field
		if tag == 'G' {
			tparams = r.tparamList()
		}
		typ := r.signature(nil, tparams)

		n := importfunc(pos, sym, typ)
		r.funcExt(n)
		return n

	case 'T', 'U':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		n := importtype(pos, sym)
		t := n.Type()

		// Because of recursion, we need to defer width calculations and
		// instantiations on intermediate types until the top-level type is
		// fully constructed. Note that we can have recursion via type
		// constraints.
		types.DeferCheckSize()
		deferDoInst()
		if tag == 'U' {
			rparams := r.typeList()
			t.SetRParams(rparams)
		}

		underlying := r.typ()
		t.SetUnderlying(underlying)

		if underlying.IsInterface() {
			// Finish up all type instantiations and CheckSize calls
			// now that a top-level type is fully constructed.
			resumeDoInst()
			types.ResumeCheckSize()
			r.typeExt(t)
			return n
		}

		ms := make([]*types.Field, r.uint64())
		for i := range ms {
			mpos := r.pos()
			msym := r.selector()
			recv := r.param()
			mtyp := r.signature(recv, nil)

			// MethodSym already marked m.Sym as a function.
			m := ir.NewNameAt(mpos, ir.MethodSym(recv.Type, msym))
			m.Class = ir.PFUNC
			m.SetType(mtyp)

			m.Func = ir.NewFunc(mpos)
			m.Func.Nname = m

			f := types.NewField(mpos, msym, mtyp)
			f.Nname = m
			ms[i] = f
		}
		t.Methods().Set(ms)

		// Finish up all instantiations and CheckSize calls now
		// that a top-level type is fully constructed.
		resumeDoInst()
		types.ResumeCheckSize()

		r.typeExt(t)
		for _, m := range ms {
			r.methExt(m)
		}
		return n

	case 'P':
		if r.p.exportVersion < iexportVersionGenerics {
			base.Fatalf("unexpected type param type")
		}
		if sym.Def != nil {
			// Make sure we use the same type param type for the same
			// name, whether it is created during types1-import or
			// this types2-to-types1 translation.
			return sym.Def.(*ir.Name)
		}
		n := importsym(pos, sym, ir.OTYPE, ir.PTYPEPARAM)
		// The typeparam index is set at the point where the containing type
		// param list is imported.
		t := types.NewTypeParam(n, 0)
		n.SetType(t)
		implicit := false
		if r.p.exportVersion >= iexportVersionGo1_18 {
			implicit = r.bool()
		}
		bound := r.typ()
		if implicit {
			bound.MarkImplicit()
		}
		t.SetBound(bound)
		return n

	case 'V':
		typ := r.typ()

		n := importvar(pos, sym, typ)
		r.varExt(n)
		return n

	default:
		base.Fatalf("unexpected tag: %v", tag)
		panic("unreachable")
	}
}

func (r *importReader) value(typ *types.Type) constant.Value {
	var kind constant.Kind
	var valType *types.Type

	if r.p.exportVersion >= iexportVersionGo1_18 {
		// TODO: add support for using the kind in the non-typeparam case.
		kind = constant.Kind(r.int64())
	}

	if typ.IsTypeParam() {
		if r.p.exportVersion < iexportVersionGo1_18 {
			// If a constant had a typeparam type, then we wrote out its
			// actual constant kind as well.
			kind = constant.Kind(r.int64())
		}
		switch kind {
		case constant.Int:
			valType = types.Types[types.TINT64]
		case constant.Float:
			valType = types.Types[types.TFLOAT64]
		case constant.Complex:
			valType = types.Types[types.TCOMPLEX128]
		}
	} else {
		kind = constTypeOf(typ)
		valType = typ
	}

	switch kind {
	case constant.Bool:
		return constant.MakeBool(r.bool())
	case constant.String:
		return constant.MakeString(r.string())
	case constant.Int:
		var i big.Int
		r.mpint(&i, valType)
		return constant.Make(&i)
	case constant.Float:
		return r.float(valType)
	case constant.Complex:
		return makeComplex(r.float(valType), r.float(valType))
	}

	base.Fatalf("unexpected value type: %v", typ)
	panic("unreachable")
}

func (r *importReader) mpint(x *big.Int, typ *types.Type) {
	signed, maxBytes := intSize(typ)

	maxSmall := 256 - maxBytes
	if signed {
		maxSmall = 256 - 2*maxBytes
	}
	if maxBytes == 1 {
		maxSmall = 256
	}

	n, _ := r.ReadByte()
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
		base.Fatalf("weird decoding: %v, %v => %v", n, signed, v)
	}
	b := make([]byte, v)
	r.Read(b)
	x.SetBytes(b)
	if signed && n&1 != 0 {
		x.Neg(x)
	}
}

func (r *importReader) float(typ *types.Type) constant.Value {
	var mant big.Int
	r.mpint(&mant, typ)
	var f big.Float
	f.SetInt(&mant)
	if f.Sign() != 0 {
		f.SetMantExp(&f, int(r.int64()))
	}
	return constant.Make(&f)
}

func (r *importReader) mprat(orig constant.Value) constant.Value {
	if !r.bool() {
		return orig
	}
	var rat big.Rat
	rat.SetString(r.string())
	return constant.Make(&rat)
}

func (r *importReader) ident(selector bool) *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	pkg := r.currPkg
	if selector {
		if types.IsExported(name) {
			pkg = types.LocalPkg
		}
	} else {
		if name == "$autotmp" {
			name = autotmpname(r.autotmpgen)
			r.autotmpgen++
		}
	}
	return pkg.Lookup(name)
}

func (r *importReader) localIdent() *types.Sym { return r.ident(false) }
func (r *importReader) selector() *types.Sym   { return r.ident(true) }

func (r *importReader) qualifiedIdent() *ir.Ident {
	name := r.string()
	pkg := r.pkg()
	sym := pkg.Lookup(name)
	return ir.NewIdent(src.NoXPos, sym)
}

func (r *importReader) pos() src.XPos {
	delta := r.int64()
	r.prevColumn += delta >> 1
	if delta&1 != 0 {
		delta = r.int64()
		r.prevLine += delta >> 1
		if delta&1 != 0 {
			r.prevBase = r.posBase()
		}
	}

	if (r.prevBase == nil || r.prevBase.AbsFilename() == "") && r.prevLine == 0 && r.prevColumn == 0 {
		// TODO(mdempsky): Remove once we reliably write
		// position information for all nodes.
		return src.NoXPos
	}

	if r.prevBase == nil {
		base.Fatalf("missing posbase")
	}
	pos := src.MakePos(r.prevBase, uint(r.prevLine), uint(r.prevColumn))
	return base.Ctxt.PosTable.XPos(pos)
}

func (r *importReader) typ() *types.Type {
	// If this is a top-level type call, defer type instantiations until the
	// type is fully constructed.
	types.DeferCheckSize()
	deferDoInst()
	t := r.p.typAt(r.uint64())
	resumeDoInst()
	types.ResumeCheckSize()
	return t
}

func (r *importReader) exoticType() *types.Type {
	switch r.uint64() {
	case exoticTypeNil:
		return nil
	case exoticTypeTuple:
		funarg := types.Funarg(r.uint64())
		n := r.uint64()
		fs := make([]*types.Field, n)
		for i := range fs {
			pos := r.pos()
			var sym *types.Sym
			switch r.uint64() {
			case exoticTypeSymNil:
				sym = nil
			case exoticTypeSymNoPkg:
				sym = types.NoPkg.Lookup(r.string())
			case exoticTypeSymWithPkg:
				pkg := r.pkg()
				sym = pkg.Lookup(r.string())
			default:
				base.Fatalf("unknown symbol kind")
			}
			typ := r.typ()
			f := types.NewField(pos, sym, typ)
			fs[i] = f
		}
		t := types.NewStruct(types.NoPkg, fs)
		t.StructType().Funarg = funarg
		return t
	case exoticTypeRecv:
		var rcvr *types.Field
		if r.bool() { // isFakeRecv
			rcvr = types.FakeRecv()
		} else {
			rcvr = r.exoticParam()
		}
		return r.exoticSignature(rcvr)
	case exoticTypeRegular:
		return r.typ()
	default:
		base.Fatalf("bad kind of call type")
		return nil
	}
}

func (r *importReader) exoticSelector() *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	pkg := r.currPkg
	if types.IsExported(name) {
		pkg = types.LocalPkg
	}
	if r.uint64() != 0 {
		pkg = r.pkg()
	}
	return pkg.Lookup(name)
}

func (r *importReader) exoticSignature(recv *types.Field) *types.Type {
	var pkg *types.Pkg
	if r.bool() { // hasPkg
		pkg = r.pkg()
	}
	params := r.exoticParamList()
	results := r.exoticParamList()
	return types.NewSignature(pkg, recv, nil, params, results)
}

func (r *importReader) exoticParamList() []*types.Field {
	n := r.uint64()
	fs := make([]*types.Field, n)
	for i := range fs {
		fs[i] = r.exoticParam()
	}
	return fs
}

func (r *importReader) exoticParam() *types.Field {
	pos := r.pos()
	sym := r.exoticSym()
	off := r.uint64()
	typ := r.exoticType()
	ddd := r.bool()
	f := types.NewField(pos, sym, typ)
	f.Offset = int64(off)
	if sym != nil {
		f.Nname = ir.NewNameAt(pos, sym)
	}
	f.SetIsDDD(ddd)
	return f
}

func (r *importReader) exoticField() *types.Field {
	pos := r.pos()
	sym := r.exoticSym()
	off := r.uint64()
	typ := r.exoticType()
	note := r.string()
	f := types.NewField(pos, sym, typ)
	f.Offset = int64(off)
	if sym != nil {
		f.Nname = ir.NewNameAt(pos, sym)
	}
	f.Note = note
	return f
}

func (r *importReader) exoticSym() *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	var pkg *types.Pkg
	if types.IsExported(name) {
		pkg = types.LocalPkg
	} else {
		pkg = r.pkg()
	}
	return pkg.Lookup(name)
}

func (p *iimporter) typAt(off uint64) *types.Type {
	t, ok := p.typCache[off]
	if !ok {
		if off < predeclReserved {
			base.Fatalf("predeclared type missing from cache: %d", off)
		}
		t = p.newReader(off-predeclReserved, nil).typ1()
		// Ensure size is calculated for imported types. Since CL 283313, the compiler
		// does not compile the function immediately when it sees them. Instead, functions
		// are pushed to compile queue, then draining from the queue for compiling.
		// During this process, the size calculation is disabled, so it is not safe for
		// calculating size during SSA generation anymore. See issue #44732.
		//
		// No need to calc sizes for re-instantiated generic types, and
		// they are not necessarily resolved until the top-level type is
		// defined (because of recursive types).
		if t.OrigType() == nil || !t.HasTParam() {
			types.CheckSize(t)
		}
		p.typCache[off] = t
	}
	return t
}

func (r *importReader) typ1() *types.Type {
	switch k := r.kind(); k {
	default:
		base.Fatalf("unexpected kind tag in %q: %v", r.p.ipkg.Path, k)
		return nil

	case definedType:
		// We might be called from within doInline, in which
		// case Sym.Def can point to declared parameters
		// instead of the top-level types. Also, we don't
		// support inlining functions with local defined
		// types. Therefore, this must be a package-scope
		// type.
		n := expandDecl(r.qualifiedIdent())
		if n.Op() != ir.OTYPE {
			base.Fatalf("expected OTYPE, got %v: %v, %v", n.Op(), n.Sym(), n)
		}
		return n.Type()
	case pointerType:
		return types.NewPtr(r.typ())
	case sliceType:
		return types.NewSlice(r.typ())
	case arrayType:
		n := r.uint64()
		return types.NewArray(r.typ(), int64(n))
	case chanType:
		dir := types.ChanDir(r.uint64())
		return types.NewChan(r.typ(), dir)
	case mapType:
		return types.NewMap(r.typ(), r.typ())

	case signatureType:
		r.setPkg()
		return r.signature(nil, nil)

	case structType:
		r.setPkg()

		fs := make([]*types.Field, r.uint64())
		for i := range fs {
			pos := r.pos()
			sym := r.selector()
			typ := r.typ()
			emb := r.bool()
			note := r.string()

			f := types.NewField(pos, sym, typ)
			if emb {
				f.Embedded = 1
			}
			f.Note = note
			fs[i] = f
		}

		return types.NewStruct(r.currPkg, fs)

	case interfaceType:
		r.setPkg()

		embeddeds := make([]*types.Field, r.uint64())
		for i := range embeddeds {
			pos := r.pos()
			typ := r.typ()

			embeddeds[i] = types.NewField(pos, nil, typ)
		}

		methods := make([]*types.Field, r.uint64())
		for i := range methods {
			pos := r.pos()
			sym := r.selector()
			typ := r.signature(types.FakeRecv(), nil)

			methods[i] = types.NewField(pos, sym, typ)
		}

		if len(embeddeds)+len(methods) == 0 {
			return types.Types[types.TINTER]
		}

		t := types.NewInterface(r.currPkg, append(embeddeds, methods...), false)

		// Ensure we expand the interface in the frontend (#25055).
		types.CheckSize(t)
		return t

	case typeParamType:
		if r.p.exportVersion < iexportVersionGenerics {
			base.Fatalf("unexpected type param type")
		}
		// Similar to code for defined types, since we "declared"
		// typeparams to deal with recursion (typeparam is used within its
		// own type bound).
		ident := r.qualifiedIdent()
		if ident.Sym().Def != nil {
			return ident.Sym().Def.(*ir.Name).Type()
		}
		n := expandDecl(ident)
		if n.Op() != ir.OTYPE {
			base.Fatalf("expected OTYPE, got %v: %v, %v", n.Op(), n.Sym(), n)
		}
		return n.Type()

	case instanceType:
		if r.p.exportVersion < iexportVersionGenerics {
			base.Fatalf("unexpected instantiation type")
		}
		pos := r.pos()
		len := r.uint64()
		targs := make([]*types.Type, len)
		for i := range targs {
			targs[i] = r.typ()
		}
		baseType := r.typ()
		t := Instantiate(pos, baseType, targs)
		return t

	case unionType:
		if r.p.exportVersion < iexportVersionGenerics {
			base.Fatalf("unexpected instantiation type")
		}
		nt := int(r.uint64())
		terms := make([]*types.Type, nt)
		tildes := make([]bool, nt)
		for i := range terms {
			tildes[i] = r.bool()
			terms[i] = r.typ()
		}
		return types.NewUnion(terms, tildes)
	}
}

func (r *importReader) kind() itag {
	return itag(r.uint64())
}

func (r *importReader) signature(recv *types.Field, tparams []*types.Field) *types.Type {
	params := r.paramList()
	results := r.paramList()
	if n := len(params); n > 0 {
		params[n-1].SetIsDDD(r.bool())
	}
	return types.NewSignature(r.currPkg, recv, tparams, params, results)
}

func (r *importReader) typeList() []*types.Type {
	n := r.uint64()
	if n == 0 {
		return nil
	}
	ts := make([]*types.Type, n)
	for i := range ts {
		ts[i] = r.typ()
		if ts[i].IsTypeParam() {
			ts[i].SetIndex(i)
		}
	}
	return ts
}

func (r *importReader) tparamList() []*types.Field {
	n := r.uint64()
	if n == 0 {
		return nil
	}
	fs := make([]*types.Field, n)
	for i := range fs {
		typ := r.typ()
		typ.SetIndex(i)
		fs[i] = types.NewField(typ.Pos(), typ.Sym(), typ)
	}
	return fs
}

func (r *importReader) paramList() []*types.Field {
	fs := make([]*types.Field, r.uint64())
	for i := range fs {
		fs[i] = r.param()
	}
	return fs
}

func (r *importReader) param() *types.Field {
	return types.NewField(r.pos(), r.localIdent(), r.typ())
}

func (r *importReader) bool() bool {
	return r.uint64() != 0
}

func (r *importReader) int64() int64 {
	n, err := binary.ReadVarint(r)
	if err != nil {
		base.Fatalf("readVarint: %v", err)
	}
	return n
}

func (r *importReader) uint64() uint64 {
	n, err := binary.ReadUvarint(r)
	if err != nil {
		base.Fatalf("readVarint: %v", err)
	}
	return n
}

func (r *importReader) byte() byte {
	x, err := r.ReadByte()
	if err != nil {
		base.Fatalf("declReader.ReadByte: %v", err)
	}
	return x
}

// Compiler-specific extensions.

func (r *importReader) constExt(n *ir.Name) {
	switch n.Type() {
	case types.UntypedFloat:
		n.SetVal(r.mprat(n.Val()))
	case types.UntypedComplex:
		v := n.Val()
		re := r.mprat(constant.Real(v))
		im := r.mprat(constant.Imag(v))
		n.SetVal(makeComplex(re, im))
	}
}

func (r *importReader) varExt(n *ir.Name) {
	r.linkname(n.Sym())
	r.symIdx(n.Sym())
}

func (r *importReader) funcExt(n *ir.Name) {
	r.linkname(n.Sym())
	r.symIdx(n.Sym())

	n.Func.ABI = obj.ABI(r.uint64())

	// Make sure //go:noinline pragma is imported (so stenciled functions have
	// same noinline status as the corresponding generic function.)
	n.Func.Pragma = ir.PragmaFlag(r.uint64())

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(n.Type()).FieldSlice() {
			f.Note = r.string()
		}
	}

	// Inline body.
	if u := r.uint64(); u > 0 {
		n.Func.Inl = &ir.Inline{
			Cost:            int32(u - 1),
			CanDelayResults: r.bool(),
		}
		n.Func.Endlineno = r.pos()
	}
}

func (r *importReader) methExt(m *types.Field) {
	if r.bool() {
		m.SetNointerface(true)
	}
	r.funcExt(m.Nname.(*ir.Name))
}

func (r *importReader) linkname(s *types.Sym) {
	s.Linkname = r.string()
}

func (r *importReader) symIdx(s *types.Sym) {
	lsym := s.Linksym()
	idx := int32(r.int64())
	if idx != -1 {
		if s.Linkname != "" {
			base.Fatalf("bad index for linknamed symbol: %v %d\n", lsym, idx)
		}
		lsym.SymIdx = idx
		lsym.Set(obj.AttrIndexed, true)
	}
}

func (r *importReader) typeExt(t *types.Type) {
	t.SetNotInHeap(r.bool())
	SetBaseTypeIndex(t, r.int64(), r.int64())
}

func SetBaseTypeIndex(t *types.Type, i, pi int64) {
	if t.Obj() == nil {
		base.Fatalf("SetBaseTypeIndex on non-defined type %v", t)
	}
	if i != -1 && pi != -1 {
		typeSymIdx[t] = [2]int64{i, pi}
	}
}

// Map imported type T to the index of type descriptor symbols of T and *T,
// so we can use index to reference the symbol.
// TODO(mdempsky): Store this information directly in the Type's Name.
var typeSymIdx = make(map[*types.Type][2]int64)

func BaseTypeIndex(t *types.Type) int64 {
	tbase := t
	if t.IsPtr() && t.Sym() == nil && t.Elem().Sym() != nil {
		tbase = t.Elem()
	}
	i, ok := typeSymIdx[tbase]
	if !ok {
		return -1
	}
	if t != tbase {
		return i[1]
	}
	return i[0]
}

func (r *importReader) doInline(fn *ir.Func) {
	if len(fn.Inl.Body) != 0 {
		base.Fatalf("%v already has inline body", fn)
	}

	//fmt.Printf("Importing %s\n", fn.Nname.Sym().Name)
	r.funcBody(fn)

	importlist = append(importlist, fn)

	if base.Flag.E > 0 && base.Flag.LowerM > 2 {
		if base.Flag.LowerM > 3 {
			fmt.Printf("inl body for %v %v: %+v\n", fn, fn.Type(), ir.Nodes(fn.Inl.Body))
		} else {
			fmt.Printf("inl body for %v %v: %v\n", fn, fn.Type(), ir.Nodes(fn.Inl.Body))
		}
	}
}

// ----------------------------------------------------------------------------
// Inlined function bodies

// Approach: Read nodes and use them to create/declare the same data structures
// as done originally by the (hidden) parser by closely following the parser's
// original code. In other words, "parsing" the import data (which happens to
// be encoded in binary rather textual form) is the best way at the moment to
// re-establish the syntax tree's invariants. At some future point we might be
// able to avoid this round-about way and create the rewritten nodes directly,
// possibly avoiding a lot of duplicate work (name resolution, type checking).
//
// Refined nodes (e.g., ODOTPTR as a refinement of OXDOT) are exported as their
// unrefined nodes (since this is what the importer uses). The respective case
// entries are unreachable in the importer.

func (r *importReader) funcBody(fn *ir.Func) {
	outerfn := r.curfn
	r.curfn = fn

	// Import local declarations.
	fn.Inl.Dcl = r.readFuncDcls(fn)

	// Import function body.
	body := r.stmtList()
	if body == nil {
		// Make sure empty body is not interpreted as
		// no inlineable body (see also parser.fnbody)
		// (not doing so can cause significant performance
		// degradation due to unnecessary calls to empty
		// functions).
		body = []ir.Node{}
	}
	ir.VisitList(body, func(n ir.Node) {
		n.SetTypecheck(1)
	})
	fn.Inl.Body = body

	r.curfn = outerfn
	if base.Flag.W >= 3 {
		fmt.Printf("Imported for %v", fn)
		ir.DumpList("", fn.Inl.Body)
	}
}

func (r *importReader) readNames(fn *ir.Func) []*ir.Name {
	dcls := make([]*ir.Name, r.int64())
	for i := range dcls {
		n := ir.NewDeclNameAt(r.pos(), ir.ONAME, r.localIdent())
		n.Class = ir.PAUTO // overwritten below for parameters/results
		n.Curfn = fn
		n.SetType(r.typ())
		dcls[i] = n
	}
	r.allDcls = append(r.allDcls, dcls...)
	return dcls
}

func (r *importReader) readFuncDcls(fn *ir.Func) []*ir.Name {
	dcls := r.readNames(fn)

	// Fixup parameter classes and associate with their
	// signature's type fields.
	i := 0
	fix := func(f *types.Field, class ir.Class) {
		if class == ir.PPARAM && (f.Sym == nil || f.Sym.Name == "_") {
			return
		}
		n := dcls[i]
		n.Class = class
		f.Nname = n
		i++
	}

	typ := fn.Type()
	if recv := typ.Recv(); recv != nil {
		fix(recv, ir.PPARAM)
	}
	for _, f := range typ.Params().FieldSlice() {
		fix(f, ir.PPARAM)
	}
	for _, f := range typ.Results().FieldSlice() {
		fix(f, ir.PPARAMOUT)
	}
	return dcls
}

func (r *importReader) localName() *ir.Name {
	i := r.int64()
	if i == -1 {
		return ir.BlankNode.(*ir.Name)
	}
	if i < 0 {
		return r.allClosureVars[-i-2]
	}
	return r.allDcls[i]
}

func (r *importReader) stmtList() []ir.Node {
	var list []ir.Node
	for {
		n := r.node()
		if n == nil {
			break
		}
		// OBLOCK nodes are not written to the import data directly,
		// but the handling of ODCL calls liststmt, which creates one.
		// Inline them into the statement list.
		if n.Op() == ir.OBLOCK {
			n := n.(*ir.BlockStmt)
			list = append(list, n.List...)
			continue
		}
		if len(list) > 0 {
			// check for an optional label that can only immediately
			// precede a for/range/select/switch statement.
			if last := list[len(list)-1]; last.Op() == ir.OLABEL {
				label := last.(*ir.LabelStmt).Label
				switch n.Op() {
				case ir.OFOR:
					n.(*ir.ForStmt).Label = label
				case ir.ORANGE:
					n.(*ir.RangeStmt).Label = label
				case ir.OSELECT:
					n.(*ir.SelectStmt).Label = label
				case ir.OSWITCH:
					n.(*ir.SwitchStmt).Label = label
				}
			}
		}
		list = append(list, n)
	}
	return list
}

func (r *importReader) caseList(switchExpr ir.Node) []*ir.CaseClause {
	namedTypeSwitch := isNamedTypeSwitch(switchExpr)

	cases := make([]*ir.CaseClause, r.uint64())
	for i := range cases {
		cas := ir.NewCaseStmt(r.pos(), nil, nil)
		cas.List = r.stmtList()
		if namedTypeSwitch {
			cas.Var = r.localName()
			cas.Var.Defn = switchExpr
		}
		cas.Body = r.stmtList()
		cases[i] = cas
	}
	return cases
}

func (r *importReader) commList() []*ir.CommClause {
	cases := make([]*ir.CommClause, r.uint64())
	for i := range cases {
		pos := r.pos()
		defaultCase := r.bool()
		var comm ir.Node
		if !defaultCase {
			comm = r.node()
		}
		cases[i] = ir.NewCommStmt(pos, comm, r.stmtList())
	}
	return cases
}

func (r *importReader) exprList() []ir.Node {
	var list []ir.Node
	for {
		n := r.expr()
		if n == nil {
			break
		}
		list = append(list, n)
	}
	return list
}

func (r *importReader) expr() ir.Node {
	n := r.node()
	if n != nil && n.Op() == ir.OBLOCK {
		n := n.(*ir.BlockStmt)
		base.Fatalf("unexpected block node: %v", n)
	}
	return n
}

// TODO(gri) split into expr and stmt
func (r *importReader) node() ir.Node {
	op := r.op()
	switch op {
	// expressions
	// case OPAREN:
	// 	unreachable - unpacked by exporter

	case ir.ONIL:
		pos := r.pos()
		typ := r.typ()

		n := ir.NewNilExpr(pos)
		n.SetType(typ)
		return n

	case ir.OLITERAL:
		pos := r.pos()
		typ := r.typ()

		n := ir.NewBasicLit(pos, r.value(typ))
		n.SetType(typ)
		return n

	case ir.ONONAME:
		isKey := r.bool()
		var n ir.Node = r.qualifiedIdent()
		// Key ONONAME entries should not be resolved - they should
		// stay as identifiers.
		if !isKey {
			n = Resolve(n)
		}
		typ := r.typ()
		if n.Type() == nil {
			n.SetType(typ)
		}
		return n

	case ir.ONAME:
		isBuiltin := r.bool()
		if isBuiltin {
			pkg := types.BuiltinPkg
			if r.bool() {
				pkg = types.UnsafePkg
			}
			return pkg.Lookup(r.string()).Def.(*ir.Name)
		}
		return r.localName()

	// case OPACK, ONONAME:
	// 	unreachable - should have been resolved by typechecking

	case ir.OTYPE:
		return ir.TypeNode(r.typ())

	case ir.ODYNAMICTYPE:
		n := ir.NewDynamicType(r.pos(), r.expr())
		if r.bool() {
			n.ITab = r.expr()
		}
		n.SetType(r.typ())
		return n

	case ir.OTYPESW:
		pos := r.pos()
		var tag *ir.Ident
		if s := r.localIdent(); s != nil {
			tag = ir.NewIdent(pos, s)
		}
		return ir.NewTypeSwitchGuard(pos, tag, r.expr())

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	//      unreachable - should have been resolved by typechecking

	case ir.OCLOSURE:
		//println("Importing CLOSURE")
		pos := r.pos()
		r.setPkg()
		typ := r.signature(nil, nil)
		r.setPkg()

		// All the remaining code below is similar to (*noder).funcLit(), but
		// with Dcls and ClosureVars lists already set up
		fn := ir.NewClosureFunc(pos, true)
		fn.Nname.SetType(typ)

		cvars := make([]*ir.Name, r.int64())
		for i := range cvars {
			cvars[i] = ir.CaptureName(r.pos(), fn, r.localName().Canonical())
			if cvars[i].Defn == nil {
				base.Fatalf("bad import of closure variable")
			}
		}
		fn.ClosureVars = cvars
		r.allClosureVars = append(r.allClosureVars, cvars...)

		fn.Inl = &ir.Inline{}
		// Read in the Dcls and Body of the closure after temporarily
		// setting r.curfn to fn.
		r.funcBody(fn)
		fn.Dcl = fn.Inl.Dcl
		fn.Body = fn.Inl.Body
		if len(fn.Body) == 0 {
			// An empty closure must be represented as a single empty
			// block statement, else it will be dropped.
			fn.Body = []ir.Node{ir.NewBlockStmt(src.NoXPos, nil)}
		}
		fn.Inl = nil

		ir.FinishCaptureNames(pos, r.curfn, fn)

		clo := fn.OClosure
		clo.SetType(typ)
		return clo

	case ir.OSTRUCTLIT:
		pos := r.pos()
		typ := r.typ()
		list := r.fieldList()
		return ir.NewCompLitExpr(pos, ir.OSTRUCTLIT, typ, list)

	case ir.OCOMPLIT:
		pos := r.pos()
		t := r.typ()
		return ir.NewCompLitExpr(pos, ir.OCOMPLIT, t, r.exprList())

	case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT:
		pos := r.pos()
		typ := r.typ()
		list := r.exprList()
		n := ir.NewCompLitExpr(pos, op, typ, list)
		if op == ir.OSLICELIT {
			n.Len = int64(r.uint64())
		}
		return n

	case ir.OKEY:
		return ir.NewKeyExpr(r.pos(), r.expr(), r.expr())

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	case ir.OXDOT, ir.ODOT, ir.ODOTPTR, ir.ODOTINTER, ir.ODOTMETH, ir.OMETHVALUE, ir.OMETHEXPR:
		pos := r.pos()
		expr := r.expr()
		sel := r.exoticSelector()
		n := ir.NewSelectorExpr(pos, op, expr, sel)
		n.SetType(r.exoticType())
		switch op {
		case ir.OXDOT:
			hasSelection := r.bool()
			// We reconstruct n.Selection for method calls on
			// generic types and method calls due to type param
			// bounds.  Otherwise, n.Selection is nil.
			if hasSelection {
				n1 := ir.NewSelectorExpr(pos, op, expr, sel)
				AddImplicitDots(n1)
				var m *types.Field
				if n1.X.Type().IsTypeParam() {
					genType := n1.X.Type().Bound()
					m = Lookdot1(n1, sel, genType, genType.AllMethods(), 1)
				} else {
					genType := types.ReceiverBaseType(n1.X.Type())
					if genType.IsInstantiatedGeneric() {
						genType = genType.OrigType()
					}
					m = Lookdot1(n1, sel, genType, genType.Methods(), 1)
				}
				assert(m != nil)
				n.Selection = m
			}
		case ir.ODOT, ir.ODOTPTR, ir.ODOTINTER:
			n.Selection = r.exoticField()
		case ir.OMETHEXPR:
			n = typecheckMethodExpr(n).(*ir.SelectorExpr)
		case ir.ODOTMETH, ir.OMETHVALUE:
			// These require a Lookup to link to the correct declaration.
			rcvrType := expr.Type()
			typ := n.Type()
			n.Selection = Lookdot(n, rcvrType, 1)
			if op == ir.OMETHVALUE {
				// Lookdot clobbers the opcode and type, undo that.
				n.SetOp(op)
				n.SetType(typ)
			}
		}
		return n

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := ir.NewTypeAssertExpr(r.pos(), r.expr(), r.typ())
		n.SetOp(op)
		return n

	case ir.ODYNAMICDOTTYPE, ir.ODYNAMICDOTTYPE2:
		n := ir.NewDynamicTypeAssertExpr(r.pos(), op, r.expr(), nil)
		if r.bool() {
			n.RType = r.expr()
		}
		if r.bool() {
			n.ITab = r.expr()
		}
		n.SetType(r.typ())
		return n

	case ir.OINDEX, ir.OINDEXMAP:
		n := ir.NewIndexExpr(r.pos(), r.expr(), r.expr())
		n.SetOp(op)
		n.SetType(r.exoticType())
		if op == ir.OINDEXMAP {
			n.Assigned = r.bool()
		}
		return n

	case ir.OSLICE, ir.OSLICESTR, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR:
		pos, x := r.pos(), r.expr()
		low, high := r.exprsOrNil()
		var max ir.Node
		if op.IsSlice3() {
			max = r.expr()
		}
		n := ir.NewSliceExpr(pos, op, x, low, high, max)
		n.SetType(r.typ())
		return n

	case ir.OCONV, ir.OCONVIFACE, ir.OCONVIDATA, ir.OCONVNOP, ir.OBYTES2STR, ir.ORUNES2STR, ir.OSTR2BYTES, ir.OSTR2RUNES, ir.ORUNESTR, ir.OSLICE2ARRPTR:
		n := ir.NewConvExpr(r.pos(), op, r.typ(), r.expr())
		n.SetImplicit(r.bool())
		return n

	case ir.OCOPY, ir.OCOMPLEX, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.OCAP, ir.OCLOSE, ir.ODELETE, ir.OLEN, ir.OMAKE,
		ir.ONEW, ir.OPANIC, ir.ORECOVER, ir.OPRINT, ir.OPRINTN,
		ir.OUNSAFEADD, ir.OUNSAFESLICE, ir.OUNSAFESLICEDATA, ir.OUNSAFESTRING, ir.OUNSAFESTRINGDATA:
		pos := r.pos()
		switch op {
		case ir.OCOPY, ir.OCOMPLEX, ir.OUNSAFEADD, ir.OUNSAFESLICE, ir.OUNSAFESTRING:
			init := r.stmtList()
			n := ir.NewBinaryExpr(pos, op, r.expr(), r.expr())
			n.SetInit(init)
			n.SetType(r.typ())
			return n
		case ir.OREAL, ir.OIMAG, ir.OCAP, ir.OCLOSE, ir.OLEN, ir.ONEW, ir.OPANIC, ir.OUNSAFESTRINGDATA, ir.OUNSAFESLICEDATA:
			n := ir.NewUnaryExpr(pos, op, r.expr())
			if op != ir.OPANIC {
				n.SetType(r.typ())
			}
			return n
		case ir.OAPPEND, ir.ODELETE, ir.ORECOVER, ir.OPRINT, ir.OPRINTN:
			init := r.stmtList()
			n := ir.NewCallExpr(pos, op, nil, r.exprList())
			n.SetInit(init)
			if op == ir.OAPPEND {
				n.IsDDD = r.bool()
			}
			if op == ir.OAPPEND || op == ir.ORECOVER {
				n.SetType(r.typ())
			}
			return n
		}
		// ir.OMAKE
		goto error

	case ir.OCALL, ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER, ir.OGETG:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewCallExpr(pos, ir.OCALL, r.expr(), r.exprList())
		n.SetOp(op)
		n.SetInit(init)
		n.IsDDD = r.bool()
		n.SetType(r.exoticType())
		return n

	case ir.OMAKEMAP, ir.OMAKECHAN, ir.OMAKESLICE:
		pos := r.pos()
		typ := r.typ()
		list := r.exprList()
		var len_, cap_ ir.Node
		if len(list) > 0 {
			len_ = list[0]
		}
		if len(list) > 1 {
			cap_ = list[1]
		}
		n := ir.NewMakeExpr(pos, op, len_, cap_)
		n.SetType(typ)
		return n

	case ir.OLINKSYMOFFSET:
		pos := r.pos()
		name := r.string()
		off := r.uint64()
		typ := r.typ()
		return ir.NewLinksymOffsetExpr(pos, Lookup(name).Linksym(), int64(off), typ)

	// unary expressions
	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT, ir.ORECV, ir.OIDATA:
		n := ir.NewUnaryExpr(r.pos(), op, r.expr())
		n.SetType(r.typ())
		return n

	case ir.OADDR, ir.OPTRLIT:
		pos := r.pos()
		expr := r.expr()
		expr.SetTypecheck(1) // we do this for all nodes after importing, but do it now so markAddrOf can see it.
		n := NodAddrAt(pos, expr)
		n.SetOp(op)
		n.SetType(r.typ())
		return n

	case ir.ODEREF:
		n := ir.NewStarExpr(r.pos(), r.expr())
		n.SetType(r.typ())
		return n

	// binary expressions
	case ir.OADD, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OEQ, ir.OGE, ir.OGT, ir.OLE, ir.OLT,
		ir.OLSH, ir.OMOD, ir.OMUL, ir.ONE, ir.OOR, ir.ORSH, ir.OSUB, ir.OXOR, ir.OEFACE:
		n := ir.NewBinaryExpr(r.pos(), op, r.expr(), r.expr())
		n.SetType(r.typ())
		return n

	case ir.OANDAND, ir.OOROR:
		n := ir.NewLogicalExpr(r.pos(), op, r.expr(), r.expr())
		n.SetType(r.typ())
		return n

	case ir.OSEND:
		return ir.NewSendStmt(r.pos(), r.expr(), r.expr())

	case ir.OADDSTR:
		pos := r.pos()
		list := r.exprList()
		n := ir.NewAddStringExpr(pos, list)
		n.SetType(r.typ())
		return n

	// --------------------------------------------------------------------
	// statements
	case ir.ODCL:
		var stmts ir.Nodes
		n := r.localName()
		stmts.Append(ir.NewDecl(n.Pos(), ir.ODCL, n))
		stmts.Append(ir.NewAssignStmt(n.Pos(), n, nil))
		return ir.NewBlockStmt(n.Pos(), stmts)

	// case OASWB:
	// 	unreachable - never exported

	case ir.OAS:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewAssignStmt(pos, r.expr(), r.expr())
		n.SetInit(init)
		n.Def = r.bool()
		return n

	case ir.OASOP:
		n := ir.NewAssignOpStmt(r.pos(), r.op(), r.expr(), nil)
		if !r.bool() {
			n.Y = ir.NewInt(1)
			n.IncDec = true
		} else {
			n.Y = r.expr()
		}
		return n

	case ir.OAS2, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewAssignListStmt(pos, op, r.exprList(), r.exprList())
		n.SetInit(init)
		n.Def = r.bool()
		return n

	case ir.ORETURN:
		return ir.NewReturnStmt(r.pos(), r.exprList())

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines (not exported)

	case ir.OGO, ir.ODEFER:
		return ir.NewGoDeferStmt(r.pos(), op, r.expr())

	case ir.OIF:
		pos, init := r.pos(), r.stmtList()
		n := ir.NewIfStmt(pos, r.expr(), r.stmtList(), r.stmtList())
		n.SetInit(init)
		return n

	case ir.OFOR:
		pos, init := r.pos(), r.stmtList()
		cond, post := r.exprsOrNil()
		n := ir.NewForStmt(pos, nil, cond, post, r.stmtList())
		n.SetInit(init)
		return n

	case ir.ORANGE:
		pos, init := r.pos(), r.stmtList()
		k, v := r.exprsOrNil()
		n := ir.NewRangeStmt(pos, k, v, r.expr(), r.stmtList())
		n.SetInit(init)
		return n

	case ir.OSELECT:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewSelectStmt(pos, r.commList())
		n.SetInit(init)
		return n

	case ir.OSWITCH:
		pos := r.pos()
		init := r.stmtList()
		x, _ := r.exprsOrNil()
		n := ir.NewSwitchStmt(pos, x, r.caseList(x))
		n.SetInit(init)
		return n

	// case OCASE:
	//	handled by caseList

	case ir.OFALL:
		return ir.NewBranchStmt(r.pos(), ir.OFALL, nil)

	// case OEMPTY:
	// 	unreachable - not emitted by exporter

	case ir.OBREAK, ir.OCONTINUE, ir.OGOTO:
		pos := r.pos()
		var sym *types.Sym
		if label := r.string(); label != "" {
			sym = Lookup(label)
		}
		return ir.NewBranchStmt(pos, op, sym)

	case ir.OLABEL:
		return ir.NewLabelStmt(r.pos(), Lookup(r.string()))

	case ir.OEND:
		return nil

	case ir.OFUNCINST:
		pos := r.pos()
		x := r.expr()
		targs := make([]ir.Ntype, r.uint64())
		for i := range targs {
			targs[i] = ir.TypeNode(r.typ())
		}
		n := ir.NewInstExpr(pos, ir.OFUNCINST, x, targs)
		n.SetType(r.typ())
		return n

	case ir.OSELRECV2:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewAssignListStmt(pos, ir.OSELRECV2, r.exprList(), r.exprList())
		n.SetInit(init)
		n.Def = r.bool()
		return n

	default:
		base.Fatalf("cannot import %v (%d) node\n"+
			"\t==> please file an issue and assign to gri@", op, int(op))
		panic("unreachable") // satisfy compiler
	}
error:
	base.Fatalf("cannot import %v (%d) node\n"+
		"\t==> please file an issue and assign to khr@", op, int(op))
	panic("unreachable") // satisfy compiler
}

func (r *importReader) op() ir.Op {
	if debug && r.uint64() != magic {
		base.Fatalf("import stream has desynchronized")
	}
	return ir.Op(r.uint64())
}

func (r *importReader) fieldList() []ir.Node {
	list := make([]ir.Node, r.uint64())
	for i := range list {
		list[i] = ir.NewStructKeyExpr(r.pos(), r.exoticField(), r.expr())
	}
	return list
}

func (r *importReader) exprsOrNil() (a, b ir.Node) {
	ab := r.uint64()
	if ab&1 != 0 {
		a = r.expr()
	}
	if ab&2 != 0 {
		b = r.node()
	}
	return
}

// NewIncompleteNamedType returns a TFORW type t with name specified by sym, such
// that t.nod and sym.Def are set correctly. If there are any RParams for the type,
// they should be set soon after creating the TFORW type, before creating the
// underlying type. That ensures that the HasTParam and HasShape flags will be set
// properly, in case this type is part of some mutually recursive type.
func NewIncompleteNamedType(pos src.XPos, sym *types.Sym) *types.Type {
	name := ir.NewDeclNameAt(pos, ir.OTYPE, sym)
	forw := types.NewNamed(name)
	name.SetType(forw)
	sym.Def = name
	return forw
}

// Instantiate creates a new named type which is the instantiation of the base
// named generic type, with the specified type args.
func Instantiate(pos src.XPos, baseType *types.Type, targs []*types.Type) *types.Type {
	baseSym := baseType.Sym()
	if strings.Index(baseSym.Name, "[") >= 0 {
		base.Fatalf("arg to Instantiate is not a base generic type")
	}
	name := InstTypeName(baseSym.Name, targs)
	instSym := baseSym.Pkg.Lookup(name)
	if instSym.Def != nil {
		// May match existing type from previous import or
		// types2-to-types1 conversion.
		t := instSym.Def.Type()
		if t.Kind() != types.TFORW {
			return t
		}
		// Or, we have started creating this type in (*TSubster).Typ, but its
		// underlying type was not completed yet, so we need to add this type
		// to deferredInstStack, if not already there.
		found := false
		for _, t2 := range deferredInstStack {
			if t2 == t {
				found = true
				break
			}
		}
		if !found {
			deferredInstStack = append(deferredInstStack, t)
		}
		return t
	}

	t := NewIncompleteNamedType(baseType.Pos(), instSym)
	t.SetRParams(targs)
	t.SetOrigType(baseType)

	// baseType may still be TFORW or its methods may not be fully filled in
	// (since we are in the middle of importing it). So, delay call to
	// substInstType until we get back up to the top of the current top-most
	// type import.
	deferredInstStack = append(deferredInstStack, t)

	return t
}

var deferredInstStack []*types.Type
var deferInst int

// deferDoInst defers substitution on instantiated types until we are at the
// top-most defined type, so the base types are fully defined.
func deferDoInst() {
	deferInst++
}

func resumeDoInst() {
	if deferInst == 1 {
		for len(deferredInstStack) > 0 {
			t := deferredInstStack[0]
			deferredInstStack = deferredInstStack[1:]
			substInstType(t, t.OrigType(), t.RParams())
		}
	}
	deferInst--
}

// doInst creates a new instantiation type (which will be added to
// deferredInstStack for completion later) for an incomplete type encountered
// during a type substitution for an instantiation. This is needed for
// instantiations of mutually recursive types.
func doInst(t *types.Type) *types.Type {
	assert(t.Kind() == types.TFORW)
	return Instantiate(t.Pos(), t.OrigType(), t.RParams())
}

// substInstType completes the instantiation of a generic type by doing a
// substitution on the underlying type itself and any methods. t is the
// instantiation being created, baseType is the base generic type, and targs are
// the type arguments that baseType is being instantiated with.
func substInstType(t *types.Type, baseType *types.Type, targs []*types.Type) {
	assert(t.Kind() == types.TFORW)
	subst := Tsubster{
		Tparams:       baseType.RParams(),
		Targs:         targs,
		SubstForwFunc: doInst,
	}
	t.SetUnderlying(subst.Typ(baseType.Underlying()))

	newfields := make([]*types.Field, baseType.Methods().Len())
	for i, f := range baseType.Methods().Slice() {
		if !f.IsMethod() || types.IsInterfaceMethod(f.Type) {
			// Do a normal substitution if this is a non-method (which
			// means this must be an interface used as a constraint) or
			// an interface method.
			t2 := subst.Typ(f.Type)
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
			continue
		}
		recvType := f.Type.Recv().Type
		if recvType.IsPtr() {
			recvType = recvType.Elem()
		}
		// Substitute in the method using the type params used in the
		// method (not the type params in the definition of the generic type).
		msubst := Tsubster{
			Tparams:       recvType.RParams(),
			Targs:         targs,
			SubstForwFunc: doInst,
		}
		t2 := msubst.Typ(f.Type)
		oldsym := f.Nname.Sym()
		newsym := MakeFuncInstSym(oldsym, targs, true, true)
		var nname *ir.Name
		if newsym.Def != nil {
			nname = newsym.Def.(*ir.Name)
		} else {
			nname = ir.NewNameAt(f.Pos, newsym)
			nname.SetType(t2)
			ir.MarkFunc(nname)
			newsym.Def = nname
		}
		newfields[i] = types.NewField(f.Pos, f.Sym, t2)
		newfields[i].Nname = nname
	}
	t.Methods().Set(newfields)
	if !t.HasTParam() && !t.HasShape() && t.Kind() != types.TINTER && t.Methods().Len() > 0 {
		// Generate all the methods for a new fully-instantiated,
		// non-interface, non-shape type.
		NeedInstType(t)
	}
}
