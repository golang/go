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
	"os"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/goobj"
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

func importReaderFor(sym *types.Sym, importers map[*types.Sym]iimporterAndOffset) *importReader {
	x, ok := importers[sym]
	if !ok {
		return nil
	}

	return x.p.newReader(x.off, sym.Pkg)
}

type intReader struct {
	*bio.Reader
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

func ReadImports(pkg *types.Pkg, in *bio.Reader) (fingerprint goobj.FingerprintType) {
	ird := &intReader{in, pkg}

	version := ird.uint64()
	if version != iexportVersion {
		base.Errorf("import %q: unknown export format version %d", pkg.Path, version)
		base.ErrorExit()
	}

	sLen := ird.uint64()
	dLen := ird.uint64()

	// Map string (and data) section into memory as a single large
	// string. This reduces heap fragmentation and allows
	// returning individual substrings very efficiently.
	data, err := mapFile(in.File(), in.Offset(), int64(sLen+dLen))
	if err != nil {
		base.Errorf("import %q: mapping input: %v", pkg.Path, err)
		base.ErrorExit()
	}
	stringData := data[:sLen]
	declData := data[sLen:]

	in.MustSeek(int64(sLen+dLen), os.SEEK_CUR)

	p := &iimporter{
		ipkg: pkg,

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
		pkgHeight := int(ird.uint64())
		if pkg.Name == "" {
			pkg.Name = pkgName
			pkg.Height = pkgHeight
			types.NumImport[pkgName]++

			// TODO(mdempsky): This belongs somewhere else.
			pkg.Lookup("_").Def = ir.BlankNode
		} else {
			if pkg.Name != pkgName {
				base.Fatalf("conflicting package names %v and %v for path %q", pkg.Name, pkgName, pkg.Path)
			}
			if pkg.Height != pkgHeight {
				base.Fatalf("conflicting package heights %v and %v for path %q", pkg.Height, pkgHeight, pkg.Path)
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

	// Fingerprint.
	_, err = io.ReadFull(in, fingerprint[:])
	if err != nil {
		base.Errorf("import %s: error reading fingerprint", pkg.Path)
		base.ErrorExit()
	}
	return fingerprint
}

type iimporter struct {
	ipkg *types.Pkg

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

		return importalias(r.p.ipkg, pos, sym, typ)

	case 'C':
		typ := r.typ()
		val := r.value(typ)

		n := importconst(r.p.ipkg, pos, sym, typ, val)
		r.constExt(n)
		return n

	case 'F':
		typ := r.signature(nil)

		n := importfunc(r.p.ipkg, pos, sym, typ)
		r.funcExt(n)
		return n

	case 'T':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		n := importtype(r.p.ipkg, pos, sym)
		t := n.Type()

		// We also need to defer width calculations until
		// after the underlying type has been assigned.
		types.DeferCheckSize()
		underlying := r.typ()
		t.SetUnderlying(underlying)
		types.ResumeCheckSize()

		if underlying.IsInterface() {
			r.typeExt(t)
			return n
		}

		ms := make([]*types.Field, r.uint64())
		for i := range ms {
			mpos := r.pos()
			msym := r.selector()
			recv := r.param()
			mtyp := r.signature(recv)

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

		r.typeExt(t)
		for _, m := range ms {
			r.methExt(m)
		}
		return n

	case 'V':
		typ := r.typ()

		n := importvar(r.p.ipkg, pos, sym, typ)
		r.varExt(n)
		return n

	default:
		base.Fatalf("unexpected tag: %v", tag)
		panic("unreachable")
	}
}

func (p *importReader) value(typ *types.Type) constant.Value {
	switch constTypeOf(typ) {
	case constant.Bool:
		return constant.MakeBool(p.bool())
	case constant.String:
		return constant.MakeString(p.string())
	case constant.Int:
		var i big.Int
		p.mpint(&i, typ)
		return constant.Make(&i)
	case constant.Float:
		return p.float(typ)
	case constant.Complex:
		return makeComplex(p.float(typ), p.float(typ))
	}

	base.Fatalf("unexpected value type: %v", typ)
	panic("unreachable")
}

func (p *importReader) mpint(x *big.Int, typ *types.Type) {
	signed, maxBytes := intSize(typ)

	maxSmall := 256 - maxBytes
	if signed {
		maxSmall = 256 - 2*maxBytes
	}
	if maxBytes == 1 {
		maxSmall = 256
	}

	n, _ := p.ReadByte()
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
	p.Read(b)
	x.SetBytes(b)
	if signed && n&1 != 0 {
		x.Neg(x)
	}
}

func (p *importReader) float(typ *types.Type) constant.Value {
	var mant big.Int
	p.mpint(&mant, typ)
	var f big.Float
	f.SetInt(&mant)
	if f.Sign() != 0 {
		f.SetMantExp(&f, int(p.int64()))
	}
	return constant.Make(&f)
}

func (p *importReader) mprat(orig constant.Value) constant.Value {
	if !p.bool() {
		return orig
	}
	var rat big.Rat
	rat.SetString(p.string())
	return constant.Make(&rat)
}

func (r *importReader) ident(selector bool) *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	pkg := r.currPkg
	if selector && types.IsExported(name) {
		pkg = types.LocalPkg
	}
	return pkg.Lookup(name)
}

func (r *importReader) localIdent() *types.Sym { return r.ident(false) }
func (r *importReader) selector() *types.Sym   { return r.ident(true) }

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
	return r.p.typAt(r.uint64())
}

func (p *iimporter) typAt(off uint64) *types.Type {
	t, ok := p.typCache[off]
	if !ok {
		if off < predeclReserved {
			base.Fatalf("predeclared type missing from cache: %d", off)
		}
		t = p.newReader(off-predeclReserved, nil).typ1()
		// Ensure size is calculated for imported types. Since CL 283313, the compiler
		// does not compile the function immediately when it sees them. Instead, funtions
		// are pushed to compile queue, then draining from the queue for compiling.
		// During this process, the size calculation is disabled, so it is not safe for
		// calculating size during SSA generation anymore. See issue #44732.
		types.CheckSize(t)
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
		return r.signature(nil)

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
			typ := r.signature(fakeRecvField())

			methods[i] = types.NewField(pos, sym, typ)
		}

		t := types.NewInterface(r.currPkg, append(embeddeds, methods...))

		// Ensure we expand the interface in the frontend (#25055).
		types.CheckSize(t)
		return t
	}
}

func (r *importReader) kind() itag {
	return itag(r.uint64())
}

func (r *importReader) signature(recv *types.Field) *types.Type {
	params := r.paramList()
	results := r.paramList()
	if n := len(params); n > 0 {
		params[n-1].SetIsDDD(r.bool())
	}
	return types.NewSignature(r.currPkg, recv, nil, params, results)
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

	n.SetPragma(ir.PragmaFlag(r.uint64()))

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(n.Type()).FieldSlice() {
			f.Note = r.string()
		}
	}

	// Inline body.
	if u := r.uint64(); u > 0 {
		n.Func.Inl = &ir.Inline{
			Cost: int32(u - 1),
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
	i, pi := r.int64(), r.int64()
	if i != -1 && pi != -1 {
		typeSymIdx[t] = [2]int64{i, pi}
	}
}

// Map imported type T to the index of type descriptor symbols of T and *T,
// so we can use index to reference the symbol.
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
	fn.Inl.Body = body

	r.curfn = outerfn
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
		} else {
			list = append(list, n)
		}

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
		cases[i] = ir.NewCommStmt(r.pos(), r.node(), r.stmtList())
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
	switch op := r.op(); op {
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
		return r.qualifiedIdent()

	case ir.ONAME:
		return r.localName()

	// case OPACK, ONONAME:
	// 	unreachable - should have been resolved by typechecking

	case ir.OTYPE:
		return ir.TypeNode(r.typ())

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
		typ := r.signature(nil)

		// All the remaining code below is similar to (*noder).funcLit(), but
		// with Dcls and ClosureVars lists already set up
		fn := ir.NewFunc(pos)
		fn.SetIsHiddenClosure(true)
		fn.Nname = ir.NewNameAt(pos, ir.BlankNode.Sym())
		fn.Nname.Func = fn
		fn.Nname.Ntype = ir.TypeNode(typ)
		fn.Nname.Defn = fn
		fn.Nname.SetType(typ)

		cvars := make([]*ir.Name, r.int64())
		for i := range cvars {
			cvars[i] = ir.CaptureName(r.pos(), fn, r.localName().Canonical())
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

		clo := ir.NewClosureExpr(pos, fn)
		fn.OClosure = clo

		return clo

	// case OPTRLIT:
	//	unreachable - mapped to case OADDR below by exporter

	case ir.OSTRUCTLIT:
		return ir.NewCompLitExpr(r.pos(), ir.OCOMPLIT, ir.TypeNode(r.typ()), r.fieldList())

	// case OARRAYLIT, OSLICELIT, OMAPLIT:
	// 	unreachable - mapped to case OCOMPLIT below by exporter

	case ir.OCOMPLIT:
		return ir.NewCompLitExpr(r.pos(), ir.OCOMPLIT, ir.TypeNode(r.typ()), r.exprList())

	case ir.OKEY:
		return ir.NewKeyExpr(r.pos(), r.expr(), r.expr())

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	// case OCALLPART:
	//	unreachable - mapped to case OXDOT below by exporter

	// case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
	// 	unreachable - mapped to case OXDOT below by exporter

	case ir.OXDOT:
		// see parser.new_dotname
		return ir.NewSelectorExpr(r.pos(), ir.OXDOT, r.expr(), r.exoticSelector())

	// case ODOTTYPE, ODOTTYPE2:
	// 	unreachable - mapped to case ODOTTYPE below by exporter

	case ir.ODOTTYPE:
		n := ir.NewTypeAssertExpr(r.pos(), r.expr(), nil)
		n.SetType(r.typ())
		return n

	// case OINDEX, OINDEXMAP, OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
	// 	unreachable - mapped to cases below by exporter

	case ir.OINDEX:
		return ir.NewIndexExpr(r.pos(), r.expr(), r.expr())

	case ir.OSLICE, ir.OSLICE3:
		pos, x := r.pos(), r.expr()
		low, high := r.exprsOrNil()
		var max ir.Node
		if op.IsSlice3() {
			max = r.expr()
		}
		return ir.NewSliceExpr(pos, op, x, low, high, max)

	// case OCONV, OCONVIFACE, OCONVNOP, OBYTES2STR, ORUNES2STR, OSTR2BYTES, OSTR2RUNES, ORUNESTR:
	// 	unreachable - mapped to OCONV case below by exporter

	case ir.OCONV:
		return ir.NewConvExpr(r.pos(), ir.OCONV, r.typ(), r.expr())

	case ir.OCOPY, ir.OCOMPLEX, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.OCAP, ir.OCLOSE, ir.ODELETE, ir.OLEN, ir.OMAKE, ir.ONEW, ir.OPANIC, ir.ORECOVER, ir.OPRINT, ir.OPRINTN:
		n := builtinCall(r.pos(), op)
		n.Args = r.exprList()
		if op == ir.OAPPEND {
			n.IsDDD = r.bool()
		}
		return n

	// case OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
	// 	unreachable - mapped to OCALL case below by exporter

	case ir.OCALL:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewCallExpr(pos, ir.OCALL, r.expr(), r.exprList())
		*n.PtrInit() = init
		n.IsDDD = r.bool()
		return n

	case ir.OMAKEMAP, ir.OMAKECHAN, ir.OMAKESLICE:
		n := builtinCall(r.pos(), ir.OMAKE)
		n.Args.Append(ir.TypeNode(r.typ()))
		n.Args.Append(r.exprList()...)
		return n

	// unary expressions
	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT, ir.ORECV:
		return ir.NewUnaryExpr(r.pos(), op, r.expr())

	case ir.OADDR:
		return NodAddrAt(r.pos(), r.expr())

	case ir.ODEREF:
		return ir.NewStarExpr(r.pos(), r.expr())

	// binary expressions
	case ir.OADD, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OEQ, ir.OGE, ir.OGT, ir.OLE, ir.OLT,
		ir.OLSH, ir.OMOD, ir.OMUL, ir.ONE, ir.OOR, ir.ORSH, ir.OSUB, ir.OXOR:
		return ir.NewBinaryExpr(r.pos(), op, r.expr(), r.expr())

	case ir.OANDAND, ir.OOROR:
		return ir.NewLogicalExpr(r.pos(), op, r.expr(), r.expr())

	case ir.OSEND:
		return ir.NewSendStmt(r.pos(), r.expr(), r.expr())

	case ir.OADDSTR:
		pos := r.pos()
		list := r.exprList()
		x := list[0]
		for _, y := range list[1:] {
			x = ir.NewBinaryExpr(pos, ir.OADD, x, y)
		}
		return x

	// --------------------------------------------------------------------
	// statements
	case ir.ODCL:
		var stmts ir.Nodes
		n := r.localName()
		stmts.Append(ir.NewDecl(n.Pos(), ir.ODCL, n))
		stmts.Append(ir.NewAssignStmt(n.Pos(), n, nil))
		return ir.NewBlockStmt(n.Pos(), stmts)

	// case OAS, OASWB:
	// 	unreachable - mapped to OAS case below by exporter

	case ir.OAS:
		return ir.NewAssignStmt(r.pos(), r.expr(), r.expr())

	case ir.OASOP:
		n := ir.NewAssignOpStmt(r.pos(), r.op(), r.expr(), nil)
		if !r.bool() {
			n.Y = ir.NewInt(1)
			n.IncDec = true
		} else {
			n.Y = r.expr()
		}
		return n

	// case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
	// 	unreachable - mapped to OAS2 case below by exporter

	case ir.OAS2:
		return ir.NewAssignListStmt(r.pos(), ir.OAS2, r.exprList(), r.exprList())

	case ir.ORETURN:
		return ir.NewReturnStmt(r.pos(), r.exprList())

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines (not exported)

	case ir.OGO, ir.ODEFER:
		return ir.NewGoDeferStmt(r.pos(), op, r.expr())

	case ir.OIF:
		pos, init := r.pos(), r.stmtList()
		n := ir.NewIfStmt(pos, r.expr(), r.stmtList(), r.stmtList())
		*n.PtrInit() = init
		return n

	case ir.OFOR:
		pos, init := r.pos(), r.stmtList()
		cond, post := r.exprsOrNil()
		n := ir.NewForStmt(pos, nil, cond, post, r.stmtList())
		*n.PtrInit() = init
		return n

	case ir.ORANGE:
		pos := r.pos()
		k, v := r.exprsOrNil()
		return ir.NewRangeStmt(pos, k, v, r.expr(), r.stmtList())

	case ir.OSELECT:
		pos := r.pos()
		init := r.stmtList()
		n := ir.NewSelectStmt(pos, r.commList())
		*n.PtrInit() = init
		return n

	case ir.OSWITCH:
		pos := r.pos()
		init := r.stmtList()
		x, _ := r.exprsOrNil()
		n := ir.NewSwitchStmt(pos, x, r.caseList(x))
		*n.PtrInit() = init
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

	default:
		base.Fatalf("cannot import %v (%d) node\n"+
			"\t==> please file an issue and assign to gri@", op, int(op))
		panic("unreachable") // satisfy compiler
	}
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
		list[i] = ir.NewStructKeyExpr(r.pos(), r.selector(), r.expr())
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

func builtinCall(pos src.XPos, op ir.Op) *ir.CallExpr {
	return ir.NewCallExpr(pos, ir.OCALL, ir.NewIdent(base.Pos, types.BuiltinPkg.Lookup(ir.OpNames[op])), nil)
}
