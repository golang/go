// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package import.
// See iexport.go for the export data format.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/goobj2"
	"cmd/internal/obj"
	"cmd/internal/src"
	"encoding/binary"
	"fmt"
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

func expandDecl(n *Node) {
	if n.Op != ONONAME {
		return
	}

	r := importReaderFor(n, declImporter)
	if r == nil {
		// Can happen if user tries to reference an undeclared name.
		return
	}

	r.doDecl(n)
}

func expandInline(fn *Node) {
	if fn.Func.Inl.Body != nil {
		return
	}

	r := importReaderFor(fn, inlineImporter)
	if r == nil {
		Fatalf("missing import reader for %v", fn)
	}

	r.doInline(fn)
}

func importReaderFor(n *Node, importers map[*types.Sym]iimporterAndOffset) *importReader {
	x, ok := importers[n.Sym]
	if !ok {
		return nil
	}

	return x.p.newReader(x.off, n.Sym.Pkg)
}

type intReader struct {
	*bio.Reader
	pkg *types.Pkg
}

func (r *intReader) int64() int64 {
	i, err := binary.ReadVarint(r.Reader)
	if err != nil {
		yyerror("import %q: read error: %v", r.pkg.Path, err)
		errorexit()
	}
	return i
}

func (r *intReader) uint64() uint64 {
	i, err := binary.ReadUvarint(r.Reader)
	if err != nil {
		yyerror("import %q: read error: %v", r.pkg.Path, err)
		errorexit()
	}
	return i
}

func iimport(pkg *types.Pkg, in *bio.Reader) (fingerprint goobj2.FingerprintType) {
	ir := &intReader{in, pkg}

	version := ir.uint64()
	if version != iexportVersion {
		yyerror("import %q: unknown export format version %d", pkg.Path, version)
		errorexit()
	}

	sLen := ir.uint64()
	dLen := ir.uint64()

	// Map string (and data) section into memory as a single large
	// string. This reduces heap fragmentation and allows
	// returning individual substrings very efficiently.
	data, err := mapFile(in.File(), in.Offset(), int64(sLen+dLen))
	if err != nil {
		yyerror("import %q: mapping input: %v", pkg.Path, err)
		errorexit()
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
	for nPkgs := ir.uint64(); nPkgs > 0; nPkgs-- {
		pkg := p.pkgAt(ir.uint64())
		pkgName := p.stringAt(ir.uint64())
		pkgHeight := int(ir.uint64())
		if pkg.Name == "" {
			pkg.Name = pkgName
			pkg.Height = pkgHeight
			numImport[pkgName]++

			// TODO(mdempsky): This belongs somewhere else.
			pkg.Lookup("_").Def = asTypesNode(nblank)
		} else {
			if pkg.Name != pkgName {
				Fatalf("conflicting package names %v and %v for path %q", pkg.Name, pkgName, pkg.Path)
			}
			if pkg.Height != pkgHeight {
				Fatalf("conflicting package heights %v and %v for path %q", pkg.Height, pkgHeight, pkg.Path)
			}
		}

		for nSyms := ir.uint64(); nSyms > 0; nSyms-- {
			s := pkg.Lookup(p.stringAt(ir.uint64()))
			off := ir.uint64()

			if _, ok := declImporter[s]; ok {
				continue
			}
			declImporter[s] = iimporterAndOffset{p, off}

			// Create stub declaration. If used, this will
			// be overwritten by expandDecl.
			if s.Def != nil {
				Fatalf("unexpected definition for %v: %v", s, asNode(s.Def))
			}
			s.Def = asTypesNode(npos(src.NoXPos, dclname(s)))
		}
	}

	// Inline body index.
	for nPkgs := ir.uint64(); nPkgs > 0; nPkgs-- {
		pkg := p.pkgAt(ir.uint64())

		for nSyms := ir.uint64(); nSyms > 0; nSyms-- {
			s := pkg.Lookup(p.stringAt(ir.uint64()))
			off := ir.uint64()

			if _, ok := inlineImporter[s]; ok {
				continue
			}
			inlineImporter[s] = iimporterAndOffset{p, off}
		}
	}

	// Fingerprint
	n, err := io.ReadFull(in, fingerprint[:])
	if err != nil || n != len(fingerprint) {
		yyerror("import %s: error reading fingerprint", pkg.Path)
		errorexit()
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
		Fatalf("varint failed")
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

func (r *importReader) doDecl(n *Node) {
	if n.Op != ONONAME {
		Fatalf("doDecl: unexpected Op for %v: %v", n.Sym, n.Op)
	}

	tag := r.byte()
	pos := r.pos()

	switch tag {
	case 'A':
		typ := r.typ()

		importalias(r.p.ipkg, pos, n.Sym, typ)

	case 'C':
		typ, val := r.value()

		importconst(r.p.ipkg, pos, n.Sym, typ, val)

	case 'F':
		typ := r.signature(nil)

		importfunc(r.p.ipkg, pos, n.Sym, typ)
		r.funcExt(n)

	case 'T':
		// Types can be recursive. We need to setup a stub
		// declaration before recursing.
		t := importtype(r.p.ipkg, pos, n.Sym)

		// We also need to defer width calculations until
		// after the underlying type has been assigned.
		defercheckwidth()
		underlying := r.typ()
		setUnderlying(t, underlying)
		resumecheckwidth()

		if underlying.IsInterface() {
			break
		}

		ms := make([]*types.Field, r.uint64())
		for i := range ms {
			mpos := r.pos()
			msym := r.ident()
			recv := r.param()
			mtyp := r.signature(recv)

			f := types.NewField()
			f.Pos = mpos
			f.Sym = msym
			f.Type = mtyp
			ms[i] = f

			m := newfuncnamel(mpos, methodSym(recv.Type, msym))
			m.Type = mtyp
			m.SetClass(PFUNC)
			// methodSym already marked m.Sym as a function.

			// (comment from parser.go)
			// inl.C's inlnode in on a dotmeth node expects to find the inlineable body as
			// (dotmeth's type).Nname.Inl, and dotmeth's type has been pulled
			// out by typecheck's lookdot as this $$.ttype. So by providing
			// this back link here we avoid special casing there.
			mtyp.SetNname(asTypesNode(m))
		}
		t.Methods().Set(ms)

		for _, m := range ms {
			r.methExt(m)
		}

	case 'V':
		typ := r.typ()

		importvar(r.p.ipkg, pos, n.Sym, typ)
		r.varExt(n)

	default:
		Fatalf("unexpected tag: %v", tag)
	}
}

func (p *importReader) value() (typ *types.Type, v Val) {
	typ = p.typ()

	switch constTypeOf(typ) {
	case CTNIL:
		v.U = &NilVal{}
	case CTBOOL:
		v.U = p.bool()
	case CTSTR:
		v.U = p.string()
	case CTINT:
		x := new(Mpint)
		x.Rune = typ == types.Idealrune
		p.mpint(&x.Val, typ)
		v.U = x
	case CTFLT:
		x := newMpflt()
		p.float(x, typ)
		v.U = x
	case CTCPLX:
		x := newMpcmplx()
		p.float(&x.Real, typ)
		p.float(&x.Imag, typ)
		v.U = x
	}
	return
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
		Fatalf("weird decoding: %v, %v => %v", n, signed, v)
	}
	b := make([]byte, v)
	p.Read(b)
	x.SetBytes(b)
	if signed && n&1 != 0 {
		x.Neg(x)
	}
}

func (p *importReader) float(x *Mpflt, typ *types.Type) {
	var mant big.Int
	p.mpint(&mant, typ)
	m := x.Val.SetInt(&mant)
	if m.Sign() == 0 {
		return
	}
	m.SetMantExp(m, int(p.int64()))
}

func (r *importReader) ident() *types.Sym {
	name := r.string()
	if name == "" {
		return nil
	}
	pkg := r.currPkg
	if types.IsExported(name) {
		pkg = localpkg
	}
	return pkg.Lookup(name)
}

func (r *importReader) qualifiedIdent() *types.Sym {
	name := r.string()
	pkg := r.pkg()
	return pkg.Lookup(name)
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
		Fatalf("missing posbase")
	}
	pos := src.MakePos(r.prevBase, uint(r.prevLine), uint(r.prevColumn))
	return Ctxt.PosTable.XPos(pos)
}

func (r *importReader) typ() *types.Type {
	return r.p.typAt(r.uint64())
}

func (p *iimporter) typAt(off uint64) *types.Type {
	t, ok := p.typCache[off]
	if !ok {
		if off < predeclReserved {
			Fatalf("predeclared type missing from cache: %d", off)
		}
		t = p.newReader(off-predeclReserved, nil).typ1()
		p.typCache[off] = t
	}
	return t
}

func (r *importReader) typ1() *types.Type {
	switch k := r.kind(); k {
	default:
		Fatalf("unexpected kind tag in %q: %v", r.p.ipkg.Path, k)
		return nil

	case definedType:
		// We might be called from within doInline, in which
		// case Sym.Def can point to declared parameters
		// instead of the top-level types. Also, we don't
		// support inlining functions with local defined
		// types. Therefore, this must be a package-scope
		// type.
		n := asNode(r.qualifiedIdent().PkgDef())
		if n.Op == ONONAME {
			expandDecl(n)
		}
		if n.Op != OTYPE {
			Fatalf("expected OTYPE, got %v: %v, %v", n.Op, n.Sym, n)
		}
		return n.Type
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

			f := types.NewField()
			f.Pos = pos
			f.Sym = sym
			f.Type = typ
			if emb {
				f.Embedded = 1
			}
			f.Note = note
			fs[i] = f
		}

		t := types.New(TSTRUCT)
		t.SetPkg(r.currPkg)
		t.SetFields(fs)
		return t

	case interfaceType:
		r.setPkg()

		embeddeds := make([]*types.Field, r.uint64())
		for i := range embeddeds {
			pos := r.pos()
			typ := r.typ()

			f := types.NewField()
			f.Pos = pos
			f.Type = typ
			embeddeds[i] = f
		}

		methods := make([]*types.Field, r.uint64())
		for i := range methods {
			pos := r.pos()
			sym := r.ident()
			typ := r.signature(fakeRecvField())

			f := types.NewField()
			f.Pos = pos
			f.Sym = sym
			f.Type = typ
			methods[i] = f
		}

		t := types.New(TINTER)
		t.SetPkg(r.currPkg)
		t.SetInterface(append(embeddeds, methods...))

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
	t := functypefield(recv, params, results)
	t.SetPkg(r.currPkg)
	return t
}

func (r *importReader) paramList() []*types.Field {
	fs := make([]*types.Field, r.uint64())
	for i := range fs {
		fs[i] = r.param()
	}
	return fs
}

func (r *importReader) param() *types.Field {
	f := types.NewField()
	f.Pos = r.pos()
	f.Sym = r.ident()
	f.Type = r.typ()
	return f
}

func (r *importReader) bool() bool {
	return r.uint64() != 0
}

func (r *importReader) int64() int64 {
	n, err := binary.ReadVarint(r)
	if err != nil {
		Fatalf("readVarint: %v", err)
	}
	return n
}

func (r *importReader) uint64() uint64 {
	n, err := binary.ReadUvarint(r)
	if err != nil {
		Fatalf("readVarint: %v", err)
	}
	return n
}

func (r *importReader) byte() byte {
	x, err := r.ReadByte()
	if err != nil {
		Fatalf("declReader.ReadByte: %v", err)
	}
	return x
}

// Compiler-specific extensions.

func (r *importReader) varExt(n *Node) {
	r.linkname(n.Sym)
	r.symIdx(n.Sym)
}

func (r *importReader) funcExt(n *Node) {
	r.linkname(n.Sym)
	r.symIdx(n.Sym)

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(n.Type).FieldSlice() {
			f.Note = r.string()
		}
	}

	// Inline body.
	if u := r.uint64(); u > 0 {
		n.Func.Inl = &Inline{
			Cost: int32(u - 1),
		}
		n.Func.Endlineno = r.pos()
	}
}

func (r *importReader) methExt(m *types.Field) {
	if r.bool() {
		m.SetNointerface(true)
	}
	r.funcExt(asNode(m.Type.Nname()))
}

func (r *importReader) linkname(s *types.Sym) {
	s.Linkname = r.string()
}

func (r *importReader) symIdx(s *types.Sym) {
	if Ctxt.Flag_go115newobj {
		lsym := s.Linksym()
		idx := int32(r.int64())
		if idx != -1 {
			if s.Linkname != "" {
				Fatalf("bad index for linknamed symbol: %v %d\n", lsym, idx)
			}
			lsym.SymIdx = idx
			lsym.Set(obj.AttrIndexed, true)
		}
	}
}

func (r *importReader) doInline(n *Node) {
	if len(n.Func.Inl.Body) != 0 {
		Fatalf("%v already has inline body", n)
	}

	funchdr(n)
	body := r.stmtList()
	funcbody()
	if body == nil {
		//
		// Make sure empty body is not interpreted as
		// no inlineable body (see also parser.fnbody)
		// (not doing so can cause significant performance
		// degradation due to unnecessary calls to empty
		// functions).
		body = []*Node{}
	}
	n.Func.Inl.Body = body

	importlist = append(importlist, n)

	if Debug['E'] > 0 && Debug['m'] > 2 {
		if Debug['m'] > 3 {
			fmt.Printf("inl body for %v %#v: %+v\n", n, n.Type, asNodes(n.Func.Inl.Body))
		} else {
			fmt.Printf("inl body for %v %#v: %v\n", n, n.Type, asNodes(n.Func.Inl.Body))
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

func (r *importReader) stmtList() []*Node {
	var list []*Node
	for {
		n := r.node()
		if n == nil {
			break
		}
		// OBLOCK nodes may be created when importing ODCL nodes - unpack them
		if n.Op == OBLOCK {
			list = append(list, n.List.Slice()...)
		} else {
			list = append(list, n)
		}

	}
	return list
}

func (r *importReader) exprList() []*Node {
	var list []*Node
	for {
		n := r.expr()
		if n == nil {
			break
		}
		list = append(list, n)
	}
	return list
}

func (r *importReader) expr() *Node {
	n := r.node()
	if n != nil && n.Op == OBLOCK {
		Fatalf("unexpected block node: %v", n)
	}
	return n
}

// TODO(gri) split into expr and stmt
func (r *importReader) node() *Node {
	switch op := r.op(); op {
	// expressions
	// case OPAREN:
	// 	unreachable - unpacked by exporter

	case OLITERAL:
		pos := r.pos()
		typ, val := r.value()

		n := npos(pos, nodlit(val))
		n.Type = typ
		return n

	case ONONAME:
		return mkname(r.qualifiedIdent())

	case ONAME:
		return mkname(r.ident())

	// case OPACK, ONONAME:
	// 	unreachable - should have been resolved by typechecking

	case OTYPE:
		return typenod(r.typ())

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	//      unreachable - should have been resolved by typechecking

	// case OCLOSURE:
	//	unimplemented

	// case OPTRLIT:
	//	unreachable - mapped to case OADDR below by exporter

	case OSTRUCTLIT:
		// TODO(mdempsky): Export position information for OSTRUCTKEY nodes.
		savedlineno := lineno
		lineno = r.pos()
		n := nodl(lineno, OCOMPLIT, nil, typenod(r.typ()))
		n.List.Set(r.elemList()) // special handling of field names
		lineno = savedlineno
		return n

	// case OARRAYLIT, OSLICELIT, OMAPLIT:
	// 	unreachable - mapped to case OCOMPLIT below by exporter

	case OCOMPLIT:
		n := nodl(r.pos(), OCOMPLIT, nil, typenod(r.typ()))
		n.List.Set(r.exprList())
		return n

	case OKEY:
		pos := r.pos()
		left, right := r.exprsOrNil()
		return nodl(pos, OKEY, left, right)

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	// case OCALLPART:
	//	unimplemented

	// case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
	// 	unreachable - mapped to case OXDOT below by exporter

	case OXDOT:
		// see parser.new_dotname
		return npos(r.pos(), nodSym(OXDOT, r.expr(), r.ident()))

	// case ODOTTYPE, ODOTTYPE2:
	// 	unreachable - mapped to case ODOTTYPE below by exporter

	case ODOTTYPE:
		n := nodl(r.pos(), ODOTTYPE, r.expr(), nil)
		n.Type = r.typ()
		return n

	// case OINDEX, OINDEXMAP, OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
	// 	unreachable - mapped to cases below by exporter

	case OINDEX:
		return nodl(r.pos(), op, r.expr(), r.expr())

	case OSLICE, OSLICE3:
		n := nodl(r.pos(), op, r.expr(), nil)
		low, high := r.exprsOrNil()
		var max *Node
		if n.Op.IsSlice3() {
			max = r.expr()
		}
		n.SetSliceBounds(low, high, max)
		return n

	// case OCONV, OCONVIFACE, OCONVNOP, OBYTES2STR, ORUNES2STR, OSTR2BYTES, OSTR2RUNES, ORUNESTR:
	// 	unreachable - mapped to OCONV case below by exporter

	case OCONV:
		n := nodl(r.pos(), OCONV, r.expr(), nil)
		n.Type = r.typ()
		return n

	case OCOPY, OCOMPLEX, OREAL, OIMAG, OAPPEND, OCAP, OCLOSE, ODELETE, OLEN, OMAKE, ONEW, OPANIC, ORECOVER, OPRINT, OPRINTN:
		n := npos(r.pos(), builtinCall(op))
		n.List.Set(r.exprList())
		if op == OAPPEND {
			n.SetIsDDD(r.bool())
		}
		return n

	// case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
	// 	unreachable - mapped to OCALL case below by exporter

	case OCALL:
		n := nodl(r.pos(), OCALL, nil, nil)
		n.Ninit.Set(r.stmtList())
		n.Left = r.expr()
		n.List.Set(r.exprList())
		n.SetIsDDD(r.bool())
		return n

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		n := npos(r.pos(), builtinCall(OMAKE))
		n.List.Append(typenod(r.typ()))
		n.List.Append(r.exprList()...)
		return n

	// unary expressions
	case OPLUS, ONEG, OADDR, OBITNOT, ODEREF, ONOT, ORECV:
		return nodl(r.pos(), op, r.expr(), nil)

	// binary expressions
	case OADD, OAND, OANDAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE, OLT,
		OLSH, OMOD, OMUL, ONE, OOR, OOROR, ORSH, OSEND, OSUB, OXOR:
		return nodl(r.pos(), op, r.expr(), r.expr())

	case OADDSTR:
		pos := r.pos()
		list := r.exprList()
		x := npos(pos, list[0])
		for _, y := range list[1:] {
			x = nodl(pos, OADD, x, y)
		}
		return x

	// --------------------------------------------------------------------
	// statements
	case ODCL:
		pos := r.pos()
		lhs := npos(pos, dclname(r.ident()))
		typ := typenod(r.typ())
		return npos(pos, liststmt(variter([]*Node{lhs}, typ, nil))) // TODO(gri) avoid list creation

	// case ODCLFIELD:
	//	unimplemented

	// case OAS, OASWB:
	// 	unreachable - mapped to OAS case below by exporter

	case OAS:
		return nodl(r.pos(), OAS, r.expr(), r.expr())

	case OASOP:
		n := nodl(r.pos(), OASOP, nil, nil)
		n.SetSubOp(r.op())
		n.Left = r.expr()
		if !r.bool() {
			n.Right = nodintconst(1)
			n.SetImplicit(true)
		} else {
			n.Right = r.expr()
		}
		return n

	// case OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
	// 	unreachable - mapped to OAS2 case below by exporter

	case OAS2:
		n := nodl(r.pos(), OAS2, nil, nil)
		n.List.Set(r.exprList())
		n.Rlist.Set(r.exprList())
		return n

	case ORETURN:
		n := nodl(r.pos(), ORETURN, nil, nil)
		n.List.Set(r.exprList())
		return n

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines (not exported)

	case OGO, ODEFER:
		return nodl(r.pos(), op, r.expr(), nil)

	case OIF:
		n := nodl(r.pos(), OIF, nil, nil)
		n.Ninit.Set(r.stmtList())
		n.Left = r.expr()
		n.Nbody.Set(r.stmtList())
		n.Rlist.Set(r.stmtList())
		return n

	case OFOR:
		n := nodl(r.pos(), OFOR, nil, nil)
		n.Ninit.Set(r.stmtList())
		n.Left, n.Right = r.exprsOrNil()
		n.Nbody.Set(r.stmtList())
		return n

	case ORANGE:
		n := nodl(r.pos(), ORANGE, nil, nil)
		n.List.Set(r.stmtList())
		n.Right = r.expr()
		n.Nbody.Set(r.stmtList())
		return n

	case OSELECT, OSWITCH:
		n := nodl(r.pos(), op, nil, nil)
		n.Ninit.Set(r.stmtList())
		n.Left, _ = r.exprsOrNil()
		n.List.Set(r.stmtList())
		return n

	case OCASE:
		n := nodl(r.pos(), OCASE, nil, nil)
		n.List.Set(r.exprList())
		// TODO(gri) eventually we must declare variables for type switch
		// statements (type switch statements are not yet exported)
		n.Nbody.Set(r.stmtList())
		return n

	case OFALL:
		n := nodl(r.pos(), OFALL, nil, nil)
		return n

	case OBREAK, OCONTINUE:
		pos := r.pos()
		left, _ := r.exprsOrNil()
		if left != nil {
			left = newname(left.Sym)
		}
		return nodl(pos, op, left, nil)

	// case OEMPTY:
	// 	unreachable - not emitted by exporter

	case OGOTO, OLABEL:
		n := nodl(r.pos(), op, nil, nil)
		n.Sym = lookup(r.string())
		return n

	case OEND:
		return nil

	default:
		Fatalf("cannot import %v (%d) node\n"+
			"\t==> please file an issue and assign to gri@", op, int(op))
		panic("unreachable") // satisfy compiler
	}
}

func (r *importReader) op() Op {
	return Op(r.uint64())
}

func (r *importReader) elemList() []*Node {
	c := r.uint64()
	list := make([]*Node, c)
	for i := range list {
		s := r.ident()
		list[i] = nodSym(OSTRUCTKEY, r.expr(), s)
	}
	return list
}

func (r *importReader) exprsOrNil() (a, b *Node) {
	ab := r.uint64()
	if ab&1 != 0 {
		a = r.expr()
	}
	if ab&2 != 0 {
		b = r.node()
	}
	return
}
