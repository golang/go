// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See iexport.go for the export data format.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/goobj"
	"cmd/internal/obj"
	"cmd/internal/src"
	"encoding/binary"
	"fmt"
	"go/constant"
	"io"
	"math/big"
	"os"
	"strings"
)

// An iimporterAndOffset identifies an importer and an offset within
// its data section.
type iimporterAndOffset struct {
	p   *iimporter
	off uint64
}

var (
	// declImporter maps from imported identifiers to an importer
	// and offset where that identifier's declaration can be read.
	declImporter = map[*types.Sym]iimporterAndOffset{}

	// inlineImporter is like declImporter, but for inline bodies
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

	r := importReaderFor(id.Sym(), declImporter)
	if r == nil {
		// Can happen if user tries to reference an undeclared name.
		return n
	}

	return r.doDecl(n.Sym())
}

func expandInline(fn *ir.Func) {
	if fn.Inl.Body != nil {
		return
	}

	r := importReaderFor(fn.Nname.Sym(), inlineImporter)
	if r == nil {
		base.Fatalf("missing import reader for %v", fn)
	}

	r.doInline(fn)
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

func iimport(pkg *types.Pkg, in *bio.Reader) (fingerprint goobj.FingerprintType) {
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

			if _, ok := declImporter[s]; !ok {
				declImporter[s] = iimporterAndOffset{p, off}
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

		return importconst(r.p.ipkg, pos, sym, typ, val)

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
		defercheckwidth()
		underlying := r.typ()
		t.SetUnderlying(underlying)
		resumecheckwidth()

		if underlying.IsInterface() {
			r.typeExt(t)
			return n
		}

		ms := make([]*types.Field, r.uint64())
		for i := range ms {
			mpos := r.pos()
			msym := r.ident()
			recv := r.param()
			mtyp := r.signature(recv)

			fn := ir.NewFunc(mpos)
			fn.SetType(mtyp)
			m := newFuncNameAt(mpos, methodSym(recv.Type, msym), fn)
			m.SetType(mtyp)
			m.SetClass(ir.PFUNC)
			// methodSym already marked m.Sym as a function.

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
		return makeInt(&i)
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

func (r *importReader) ident() *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	pkg := r.currPkg
	if types.IsExported(name) {
		pkg = types.LocalPkg
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
			sym := r.ident()
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
			sym := r.ident()
			typ := r.signature(fakeRecvField())

			methods[i] = types.NewField(pos, sym, typ)
		}

		t := types.NewInterface(r.currPkg, append(embeddeds, methods...))

		// Ensure we expand the interface in the frontend (#25055).
		checkwidth(t)
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
	return types.NewSignature(r.currPkg, recv, params, results)
}

func (r *importReader) paramList() []*types.Field {
	fs := make([]*types.Field, r.uint64())
	for i := range fs {
		fs[i] = r.param()
	}
	return fs
}

func (r *importReader) param() *types.Field {
	return types.NewField(r.pos(), r.ident(), r.typ())
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

func (r *importReader) varExt(n ir.Node) {
	r.linkname(n.Sym())
	r.symIdx(n.Sym())
}

func (r *importReader) funcExt(n ir.Node) {
	r.linkname(n.Sym())
	r.symIdx(n.Sym())

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(n.Type()).FieldSlice() {
			f.Note = r.string()
		}
	}

	// Inline body.
	if u := r.uint64(); u > 0 {
		n.Func().Inl = &ir.Inline{
			Cost: int32(u - 1),
		}
		n.Func().Endlineno = r.pos()
	}
}

func (r *importReader) methExt(m *types.Field) {
	if r.bool() {
		m.SetNointerface(true)
	}
	r.funcExt(ir.AsNode(m.Nname))
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

func (r *importReader) doInline(fn *ir.Func) {
	if len(fn.Inl.Body) != 0 {
		base.Fatalf("%v already has inline body", fn)
	}

	funchdr(fn)
	body := r.stmtList()
	funcbody()
	if body == nil {
		//
		// Make sure empty body is not interpreted as
		// no inlineable body (see also parser.fnbody)
		// (not doing so can cause significant performance
		// degradation due to unnecessary calls to empty
		// functions).
		body = []ir.Node{}
	}
	fn.Inl.Body = body

	importlist = append(importlist, fn)

	if base.Flag.E > 0 && base.Flag.LowerM > 2 {
		if base.Flag.LowerM > 3 {
			fmt.Printf("inl body for %v %v: %+v\n", fn, fn.Type(), ir.AsNodes(fn.Inl.Body))
		} else {
			fmt.Printf("inl body for %v %v: %v\n", fn, fn.Type(), ir.AsNodes(fn.Inl.Body))
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
			list = append(list, n.List().Slice()...)
		} else {
			list = append(list, n)
		}

	}
	return list
}

func (r *importReader) caseList(sw ir.Node) []ir.Node {
	namedTypeSwitch := isNamedTypeSwitch(sw)

	cases := make([]ir.Node, r.uint64())
	for i := range cases {
		cas := ir.NodAt(r.pos(), ir.OCASE, nil, nil)
		cas.PtrList().Set(r.stmtList())
		if namedTypeSwitch {
			// Note: per-case variables will have distinct, dotted
			// names after import. That's okay: swt.go only needs
			// Sym for diagnostics anyway.
			caseVar := ir.NewNameAt(cas.Pos(), r.ident())
			declare(caseVar, dclcontext)
			cas.PtrRlist().Set1(caseVar)
			caseVar.Defn = sw.(*ir.SwitchStmt).Left()
		}
		cas.PtrBody().Set(r.stmtList())
		cases[i] = cas
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

		n := npos(pos, nodnil())
		n.SetType(typ)
		return n

	case ir.OLITERAL:
		pos := r.pos()
		typ := r.typ()

		n := npos(pos, ir.NewLiteral(r.value(typ)))
		n.SetType(typ)
		return n

	case ir.ONONAME:
		return r.qualifiedIdent()

	case ir.ONAME:
		return r.ident().Def.(*ir.Name)

	// case OPACK, ONONAME:
	// 	unreachable - should have been resolved by typechecking

	case ir.OTYPE:
		return ir.TypeNode(r.typ())

	case ir.OTYPESW:
		pos := r.pos()
		var tag *ir.Ident
		if s := r.ident(); s != nil {
			tag = ir.NewIdent(pos, s)
		}
		expr, _ := r.exprsOrNil()
		return ir.NewTypeSwitchGuard(pos, tag, expr)

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	//      unreachable - should have been resolved by typechecking

	// case OCLOSURE:
	//	unimplemented

	// case OPTRLIT:
	//	unreachable - mapped to case OADDR below by exporter

	case ir.OSTRUCTLIT:
		// TODO(mdempsky): Export position information for OSTRUCTKEY nodes.
		savedlineno := base.Pos
		base.Pos = r.pos()
		n := ir.NodAt(base.Pos, ir.OCOMPLIT, nil, ir.TypeNode(r.typ()))
		n.PtrList().Set(r.elemList()) // special handling of field names
		base.Pos = savedlineno
		return n

	// case OARRAYLIT, OSLICELIT, OMAPLIT:
	// 	unreachable - mapped to case OCOMPLIT below by exporter

	case ir.OCOMPLIT:
		n := ir.NodAt(r.pos(), ir.OCOMPLIT, nil, ir.TypeNode(r.typ()))
		n.PtrList().Set(r.exprList())
		return n

	case ir.OKEY:
		pos := r.pos()
		left, right := r.exprsOrNil()
		return ir.NodAt(pos, ir.OKEY, left, right)

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	// case OCALLPART:
	//	unreachable - mapped to case OXDOT below by exporter

	// case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
	// 	unreachable - mapped to case OXDOT below by exporter

	case ir.OXDOT:
		// see parser.new_dotname
		return npos(r.pos(), nodSym(ir.OXDOT, r.expr(), r.ident()))

	// case ODOTTYPE, ODOTTYPE2:
	// 	unreachable - mapped to case ODOTTYPE below by exporter

	case ir.ODOTTYPE:
		n := ir.NodAt(r.pos(), ir.ODOTTYPE, r.expr(), nil)
		n.SetType(r.typ())
		return n

	// case OINDEX, OINDEXMAP, OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
	// 	unreachable - mapped to cases below by exporter

	case ir.OINDEX:
		return ir.NodAt(r.pos(), ir.OINDEX, r.expr(), r.expr())

	case ir.OSLICE, ir.OSLICE3:
		n := ir.NewSliceExpr(r.pos(), op, r.expr())
		low, high := r.exprsOrNil()
		var max ir.Node
		if n.Op().IsSlice3() {
			max = r.expr()
		}
		n.SetSliceBounds(low, high, max)
		return n

	// case OCONV, OCONVIFACE, OCONVNOP, OBYTES2STR, ORUNES2STR, OSTR2BYTES, OSTR2RUNES, ORUNESTR:
	// 	unreachable - mapped to OCONV case below by exporter

	case ir.OCONV:
		n := ir.NodAt(r.pos(), ir.OCONV, r.expr(), nil)
		n.SetType(r.typ())
		return n

	case ir.OCOPY, ir.OCOMPLEX, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.OCAP, ir.OCLOSE, ir.ODELETE, ir.OLEN, ir.OMAKE, ir.ONEW, ir.OPANIC, ir.ORECOVER, ir.OPRINT, ir.OPRINTN:
		n := builtinCall(r.pos(), op)
		n.PtrList().Set(r.exprList())
		if op == ir.OAPPEND {
			n.SetIsDDD(r.bool())
		}
		return n

	// case OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
	// 	unreachable - mapped to OCALL case below by exporter

	case ir.OCALL:
		n := ir.NodAt(r.pos(), ir.OCALL, nil, nil)
		n.PtrInit().Set(r.stmtList())
		n.SetLeft(r.expr())
		n.PtrList().Set(r.exprList())
		n.SetIsDDD(r.bool())
		return n

	case ir.OMAKEMAP, ir.OMAKECHAN, ir.OMAKESLICE:
		n := builtinCall(r.pos(), ir.OMAKE)
		n.PtrList().Append(ir.TypeNode(r.typ()))
		n.PtrList().Append(r.exprList()...)
		return n

	// unary expressions
	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT, ir.ORECV:
		return ir.NewUnaryExpr(r.pos(), op, r.expr())

	case ir.OADDR:
		return nodAddrAt(r.pos(), r.expr())

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
		x := npos(pos, list[0])
		for _, y := range list[1:] {
			x = ir.NodAt(pos, ir.OADD, x, y)
		}
		return x

	// --------------------------------------------------------------------
	// statements
	case ir.ODCL:
		pos := r.pos()
		lhs := ir.NewDeclNameAt(pos, r.ident())
		typ := ir.TypeNode(r.typ())
		return npos(pos, liststmt(variter([]ir.Node{lhs}, typ, nil))) // TODO(gri) avoid list creation

	// case OAS, OASWB:
	// 	unreachable - mapped to OAS case below by exporter

	case ir.OAS:
		return ir.NodAt(r.pos(), ir.OAS, r.expr(), r.expr())

	case ir.OASOP:
		n := ir.NodAt(r.pos(), ir.OASOP, nil, nil)
		n.SetSubOp(r.op())
		n.SetLeft(r.expr())
		if !r.bool() {
			n.SetRight(nodintconst(1))
			n.SetImplicit(true)
		} else {
			n.SetRight(r.expr())
		}
		return n

	// case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
	// 	unreachable - mapped to OAS2 case below by exporter

	case ir.OAS2:
		n := ir.NodAt(r.pos(), ir.OAS2, nil, nil)
		n.PtrList().Set(r.exprList())
		n.PtrRlist().Set(r.exprList())
		return n

	case ir.ORETURN:
		n := ir.NodAt(r.pos(), ir.ORETURN, nil, nil)
		n.PtrList().Set(r.exprList())
		return n

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines (not exported)

	case ir.OGO, ir.ODEFER:
		return ir.NewGoDeferStmt(r.pos(), op, r.expr())

	case ir.OIF:
		n := ir.NodAt(r.pos(), ir.OIF, nil, nil)
		n.PtrInit().Set(r.stmtList())
		n.SetLeft(r.expr())
		n.PtrBody().Set(r.stmtList())
		n.PtrRlist().Set(r.stmtList())
		return n

	case ir.OFOR:
		n := ir.NodAt(r.pos(), ir.OFOR, nil, nil)
		n.PtrInit().Set(r.stmtList())
		left, right := r.exprsOrNil()
		n.SetLeft(left)
		n.SetRight(right)
		n.PtrBody().Set(r.stmtList())
		return n

	case ir.ORANGE:
		n := ir.NodAt(r.pos(), ir.ORANGE, nil, nil)
		n.PtrList().Set(r.stmtList())
		n.SetRight(r.expr())
		n.PtrBody().Set(r.stmtList())
		return n

	case ir.OSELECT:
		n := ir.NodAt(r.pos(), ir.OSELECT, nil, nil)
		n.PtrInit().Set(r.stmtList())
		r.exprsOrNil() // TODO(rsc): Delete (and fix exporter). These are always nil.
		n.PtrList().Set(r.caseList(n))
		return n

	case ir.OSWITCH:
		n := ir.NodAt(r.pos(), ir.OSWITCH, nil, nil)
		n.PtrInit().Set(r.stmtList())
		left, _ := r.exprsOrNil()
		n.SetLeft(left)
		n.PtrList().Set(r.caseList(n))
		return n

	// case OCASE:
	//	handled by caseList

	case ir.OFALL:
		n := ir.NodAt(r.pos(), ir.OFALL, nil, nil)
		return n

	// case OEMPTY:
	// 	unreachable - not emitted by exporter

	case ir.OBREAK, ir.OCONTINUE, ir.OGOTO:
		var sym *types.Sym
		pos := r.pos()
		if label := r.string(); label != "" {
			sym = lookup(label)
		}
		return ir.NewBranchStmt(pos, op, sym)

	case ir.OLABEL:
		return ir.NewLabelStmt(r.pos(), lookup(r.string()))

	case ir.OEND:
		return nil

	default:
		base.Fatalf("cannot import %v (%d) node\n"+
			"\t==> please file an issue and assign to gri@", op, int(op))
		panic("unreachable") // satisfy compiler
	}
}

func (r *importReader) op() ir.Op {
	return ir.Op(r.uint64())
}

func (r *importReader) elemList() []ir.Node {
	c := r.uint64()
	list := make([]ir.Node, c)
	for i := range list {
		s := r.ident()
		list[i] = nodSym(ir.OSTRUCTKEY, r.expr(), s)
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

func npos(pos src.XPos, n ir.Node) ir.Node {
	n.SetPos(pos)
	return n
}
