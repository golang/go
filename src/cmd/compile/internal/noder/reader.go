// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"fmt"
	"go/constant"
	"internal/buildcfg"
	"internal/pkgbits"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/deadcode"
	"cmd/compile/internal/dwarfgen"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

type pkgReader struct {
	pkgbits.PkgDecoder

	posBases []*src.PosBase
	pkgs     []*types.Pkg
	typs     []*types.Type

	// offset for rewriting the given index into the output,
	// but bitwise inverted so we can detect if we're missing the entry or not.
	newindex []int
}

func newPkgReader(pr pkgbits.PkgDecoder) *pkgReader {
	return &pkgReader{
		PkgDecoder: pr,

		posBases: make([]*src.PosBase, pr.NumElems(pkgbits.RelocPosBase)),
		pkgs:     make([]*types.Pkg, pr.NumElems(pkgbits.RelocPkg)),
		typs:     make([]*types.Type, pr.NumElems(pkgbits.RelocType)),

		newindex: make([]int, pr.TotalElems()),
	}
}

type pkgReaderIndex struct {
	pr   *pkgReader
	idx  int
	dict *readerDict
}

func (pri pkgReaderIndex) asReader(k pkgbits.RelocKind, marker pkgbits.SyncMarker) *reader {
	r := pri.pr.newReader(k, pri.idx, marker)
	r.dict = pri.dict
	return r
}

func (pr *pkgReader) newReader(k pkgbits.RelocKind, idx int, marker pkgbits.SyncMarker) *reader {
	return &reader{
		Decoder: pr.NewDecoder(k, idx, marker),
		p:       pr,
	}
}

type reader struct {
	pkgbits.Decoder

	p *pkgReader

	dict *readerDict

	// TODO(mdempsky): The state below is all specific to reading
	// function bodies. It probably makes sense to split it out
	// separately so that it doesn't take up space in every reader
	// instance.

	curfn       *ir.Func
	locals      []*ir.Name
	closureVars []*ir.Name

	funarghack bool

	// scopeVars is a stack tracking the number of variables declared in
	// the current function at the moment each open scope was opened.
	scopeVars         []int
	marker            dwarfgen.ScopeMarker
	lastCloseScopePos src.XPos

	// === details for handling inline body expansion ===

	// If we're reading in a function body because of inlining, this is
	// the call that we're inlining for.
	inlCaller    *ir.Func
	inlCall      *ir.CallExpr
	inlFunc      *ir.Func
	inlTreeIndex int
	inlPosBases  map[*src.PosBase]*src.PosBase

	delayResults bool

	// Label to return to.
	retlabel *types.Sym

	inlvars, retvars ir.Nodes
}

type readerDict struct {
	// targs holds the implicit and explicit type arguments in use for
	// reading the current object. For example:
	//
	//	func F[T any]() {
	//		type X[U any] struct { t T; u U }
	//		var _ X[string]
	//	}
	//
	//	var _ = F[int]
	//
	// While instantiating F[int], we need to in turn instantiate
	// X[string]. [int] and [string] are explicit type arguments for F
	// and X, respectively; but [int] is also the implicit type
	// arguments for X.
	//
	// (As an analogy to function literals, explicits are the function
	// literal's formal parameters, while implicits are variables
	// captured by the function literal.)
	targs []*types.Type

	// implicits counts how many of types within targs are implicit type
	// arguments; the rest are explicit.
	implicits int

	derived      []derivedInfo // reloc index of the derived type's descriptor
	derivedTypes []*types.Type // slice of previously computed derived types

	funcs    []objInfo
	funcsObj []ir.Node

	itabs []itabInfo2
}

type itabInfo2 struct {
	typ  *types.Type
	lsym *obj.LSym
}

func setType(n ir.Node, typ *types.Type) {
	n.SetType(typ)
	n.SetTypecheck(1)

	if name, ok := n.(*ir.Name); ok {
		name.Ntype = ir.TypeNode(name.Type())
	}
}

func setValue(name *ir.Name, val constant.Value) {
	name.SetVal(val)
	name.Defn = nil
}

// @@@ Positions

func (r *reader) pos() src.XPos {
	return base.Ctxt.PosTable.XPos(r.pos0())
}

func (r *reader) pos0() src.Pos {
	r.Sync(pkgbits.SyncPos)
	if !r.Bool() {
		return src.NoPos
	}

	posBase := r.posBase()
	line := r.Uint()
	col := r.Uint()
	return src.MakePos(posBase, line, col)
}

func (r *reader) posBase() *src.PosBase {
	return r.inlPosBase(r.p.posBaseIdx(r.Reloc(pkgbits.RelocPosBase)))
}

func (pr *pkgReader) posBaseIdx(idx int) *src.PosBase {
	if b := pr.posBases[idx]; b != nil {
		return b
	}

	r := pr.newReader(pkgbits.RelocPosBase, idx, pkgbits.SyncPosBase)
	var b *src.PosBase

	absFilename := r.String()
	filename := absFilename

	// For build artifact stability, the export data format only
	// contains the "absolute" filename as returned by objabi.AbsFile.
	// However, some tests (e.g., test/run.go's asmcheck tests) expect
	// to see the full, original filename printed out. Re-expanding
	// "$GOROOT" to buildcfg.GOROOT is a close-enough approximation to
	// satisfy this.
	//
	// TODO(mdempsky): De-duplicate this logic with similar logic in
	// cmd/link/internal/ld's expandGoroot. However, this will probably
	// require being more consistent about when we use native vs UNIX
	// file paths.
	const dollarGOROOT = "$GOROOT"
	if buildcfg.GOROOT != "" && strings.HasPrefix(filename, dollarGOROOT) {
		filename = buildcfg.GOROOT + filename[len(dollarGOROOT):]
	}

	if r.Bool() {
		b = src.NewFileBase(filename, absFilename)
	} else {
		pos := r.pos0()
		line := r.Uint()
		col := r.Uint()
		b = src.NewLinePragmaBase(pos, filename, absFilename, line, col)
	}

	pr.posBases[idx] = b
	return b
}

func (r *reader) inlPosBase(oldBase *src.PosBase) *src.PosBase {
	if r.inlCall == nil {
		return oldBase
	}

	if newBase, ok := r.inlPosBases[oldBase]; ok {
		return newBase
	}

	newBase := src.NewInliningBase(oldBase, r.inlTreeIndex)
	r.inlPosBases[oldBase] = newBase
	return newBase
}

func (r *reader) updatePos(xpos src.XPos) src.XPos {
	pos := base.Ctxt.PosTable.Pos(xpos)
	pos.SetBase(r.inlPosBase(pos.Base()))
	return base.Ctxt.PosTable.XPos(pos)
}

func (r *reader) origPos(xpos src.XPos) src.XPos {
	if r.inlCall == nil {
		return xpos
	}

	pos := base.Ctxt.PosTable.Pos(xpos)
	for old, new := range r.inlPosBases {
		if pos.Base() == new {
			pos.SetBase(old)
			return base.Ctxt.PosTable.XPos(pos)
		}
	}

	base.FatalfAt(xpos, "pos base missing from inlPosBases")
	panic("unreachable")
}

// @@@ Packages

func (r *reader) pkg() *types.Pkg {
	r.Sync(pkgbits.SyncPkg)
	return r.p.pkgIdx(r.Reloc(pkgbits.RelocPkg))
}

func (pr *pkgReader) pkgIdx(idx int) *types.Pkg {
	if pkg := pr.pkgs[idx]; pkg != nil {
		return pkg
	}

	pkg := pr.newReader(pkgbits.RelocPkg, idx, pkgbits.SyncPkgDef).doPkg()
	pr.pkgs[idx] = pkg
	return pkg
}

func (r *reader) doPkg() *types.Pkg {
	path := r.String()
	if path == "builtin" {
		return types.BuiltinPkg
	}
	if path == "" {
		path = r.p.PkgPath()
	}

	name := r.String()
	height := r.Len()

	pkg := types.NewPkg(path, "")

	if pkg.Name == "" {
		pkg.Name = name
	} else {
		assert(pkg.Name == name)
	}

	if pkg.Height == 0 {
		pkg.Height = height
	} else {
		assert(pkg.Height == height)
	}

	return pkg
}

// @@@ Types

func (r *reader) typ() *types.Type {
	return r.typWrapped(true)
}

// typWrapped is like typ, but allows suppressing generation of
// unnecessary wrappers as a compile-time optimization.
func (r *reader) typWrapped(wrapped bool) *types.Type {
	return r.p.typIdx(r.typInfo(), r.dict, wrapped)
}

func (r *reader) typInfo() typeInfo {
	r.Sync(pkgbits.SyncType)
	if r.Bool() {
		return typeInfo{idx: r.Len(), derived: true}
	}
	return typeInfo{idx: r.Reloc(pkgbits.RelocType), derived: false}
}

func (pr *pkgReader) typIdx(info typeInfo, dict *readerDict, wrapped bool) *types.Type {
	idx := info.idx
	var where **types.Type
	if info.derived {
		where = &dict.derivedTypes[idx]
		idx = dict.derived[idx].idx
	} else {
		where = &pr.typs[idx]
	}

	if typ := *where; typ != nil {
		return typ
	}

	r := pr.newReader(pkgbits.RelocType, idx, pkgbits.SyncTypeIdx)
	r.dict = dict

	typ := r.doTyp()
	assert(typ != nil)

	// For recursive type declarations involving interfaces and aliases,
	// above r.doTyp() call may have already set pr.typs[idx], so just
	// double check and return the type.
	//
	// Example:
	//
	//     type F = func(I)
	//
	//     type I interface {
	//         m(F)
	//     }
	//
	// The writer writes data types in following index order:
	//
	//     0: func(I)
	//     1: I
	//     2: interface{m(func(I))}
	//
	// The reader resolves it in following index order:
	//
	//     0 -> 1 -> 2 -> 0 -> 1
	//
	// and can divide in logically 2 steps:
	//
	//  - 0 -> 1     : first time the reader reach type I,
	//                 it creates new named type with symbol I.
	//
	//  - 2 -> 0 -> 1: the reader ends up reaching symbol I again,
	//                 now the symbol I was setup in above step, so
	//                 the reader just return the named type.
	//
	// Now, the functions called return, the pr.typs looks like below:
	//
	//  - 0 -> 1 -> 2 -> 0 : [<T> I <T>]
	//  - 0 -> 1 -> 2      : [func(I) I <T>]
	//  - 0 -> 1           : [func(I) I interface { "".m(func("".I)) }]
	//
	// The idx 1, corresponding with type I was resolved successfully
	// after r.doTyp() call.

	if prev := *where; prev != nil {
		return prev
	}

	if wrapped {
		// Only cache if we're adding wrappers, so that other callers that
		// find a cached type know it was wrapped.
		*where = typ

		r.needWrapper(typ)
	}

	if !typ.IsUntyped() {
		types.CheckSize(typ)
	}

	return typ
}

func (r *reader) doTyp() *types.Type {
	switch tag := pkgbits.CodeType(r.Code(pkgbits.SyncType)); tag {
	default:
		panic(fmt.Sprintf("unexpected type: %v", tag))

	case pkgbits.TypeBasic:
		return *basics[r.Len()]

	case pkgbits.TypeNamed:
		obj := r.obj()
		assert(obj.Op() == ir.OTYPE)
		return obj.Type()

	case pkgbits.TypeTypeParam:
		return r.dict.targs[r.Len()]

	case pkgbits.TypeArray:
		len := int64(r.Uint64())
		return types.NewArray(r.typ(), len)
	case pkgbits.TypeChan:
		dir := dirs[r.Len()]
		return types.NewChan(r.typ(), dir)
	case pkgbits.TypeMap:
		return types.NewMap(r.typ(), r.typ())
	case pkgbits.TypePointer:
		return types.NewPtr(r.typ())
	case pkgbits.TypeSignature:
		return r.signature(types.LocalPkg, nil)
	case pkgbits.TypeSlice:
		return types.NewSlice(r.typ())
	case pkgbits.TypeStruct:
		return r.structType()
	case pkgbits.TypeInterface:
		return r.interfaceType()
	case pkgbits.TypeUnion:
		return r.unionType()
	}
}

func (r *reader) unionType() *types.Type {
	terms := make([]*types.Type, r.Len())
	tildes := make([]bool, len(terms))
	for i := range terms {
		tildes[i] = r.Bool()
		terms[i] = r.typ()
	}
	return types.NewUnion(terms, tildes)
}

func (r *reader) interfaceType() *types.Type {
	tpkg := types.LocalPkg // TODO(mdempsky): Remove after iexport is gone.

	nmethods, nembeddeds := r.Len(), r.Len()
	implicit := nmethods == 0 && nembeddeds == 1 && r.Bool()
	assert(!implicit) // implicit interfaces only appear in constraints

	fields := make([]*types.Field, nmethods+nembeddeds)
	methods, embeddeds := fields[:nmethods], fields[nmethods:]

	for i := range methods {
		pos := r.pos()
		pkg, sym := r.selector()
		tpkg = pkg
		mtyp := r.signature(pkg, types.FakeRecv())
		methods[i] = types.NewField(pos, sym, mtyp)
	}
	for i := range embeddeds {
		embeddeds[i] = types.NewField(src.NoXPos, nil, r.typ())
	}

	if len(fields) == 0 {
		return types.Types[types.TINTER] // empty interface
	}
	return types.NewInterface(tpkg, fields, false)
}

func (r *reader) structType() *types.Type {
	tpkg := types.LocalPkg // TODO(mdempsky): Remove after iexport is gone.
	fields := make([]*types.Field, r.Len())
	for i := range fields {
		pos := r.pos()
		pkg, sym := r.selector()
		tpkg = pkg
		ftyp := r.typ()
		tag := r.String()
		embedded := r.Bool()

		f := types.NewField(pos, sym, ftyp)
		f.Note = tag
		if embedded {
			f.Embedded = 1
		}
		fields[i] = f
	}
	return types.NewStruct(tpkg, fields)
}

func (r *reader) signature(tpkg *types.Pkg, recv *types.Field) *types.Type {
	r.Sync(pkgbits.SyncSignature)

	params := r.params(&tpkg)
	results := r.params(&tpkg)
	if r.Bool() { // variadic
		params[len(params)-1].SetIsDDD(true)
	}

	return types.NewSignature(tpkg, recv, nil, params, results)
}

func (r *reader) params(tpkg **types.Pkg) []*types.Field {
	r.Sync(pkgbits.SyncParams)
	fields := make([]*types.Field, r.Len())
	for i := range fields {
		*tpkg, fields[i] = r.param()
	}
	return fields
}

func (r *reader) param() (*types.Pkg, *types.Field) {
	r.Sync(pkgbits.SyncParam)

	pos := r.pos()
	pkg, sym := r.localIdent()
	typ := r.typ()

	return pkg, types.NewField(pos, sym, typ)
}

// @@@ Objects

var objReader = map[*types.Sym]pkgReaderIndex{}

func (r *reader) obj() ir.Node {
	r.Sync(pkgbits.SyncObject)

	if r.Bool() {
		idx := r.Len()
		obj := r.dict.funcsObj[idx]
		if obj == nil {
			fn := r.dict.funcs[idx]
			targs := make([]*types.Type, len(fn.explicits))
			for i, targ := range fn.explicits {
				targs[i] = r.p.typIdx(targ, r.dict, true)
			}

			obj = r.p.objIdx(fn.idx, nil, targs)
			assert(r.dict.funcsObj[idx] == nil)
			r.dict.funcsObj[idx] = obj
		}
		return obj
	}

	idx := r.Reloc(pkgbits.RelocObj)

	explicits := make([]*types.Type, r.Len())
	for i := range explicits {
		explicits[i] = r.typ()
	}

	var implicits []*types.Type
	if r.dict != nil {
		implicits = r.dict.targs
	}

	return r.p.objIdx(idx, implicits, explicits)
}

func (pr *pkgReader) objIdx(idx int, implicits, explicits []*types.Type) ir.Node {
	rname := pr.newReader(pkgbits.RelocName, idx, pkgbits.SyncObject1)
	_, sym := rname.qualifiedIdent()
	tag := pkgbits.CodeObj(rname.Code(pkgbits.SyncCodeObj))

	if tag == pkgbits.ObjStub {
		assert(!sym.IsBlank())
		switch sym.Pkg {
		case types.BuiltinPkg, types.UnsafePkg:
			return sym.Def.(ir.Node)
		}
		if pri, ok := objReader[sym]; ok {
			return pri.pr.objIdx(pri.idx, nil, explicits)
		}
		base.Fatalf("unresolved stub: %v", sym)
	}

	dict := pr.objDictIdx(sym, idx, implicits, explicits)

	r := pr.newReader(pkgbits.RelocObj, idx, pkgbits.SyncObject1)
	rext := pr.newReader(pkgbits.RelocObjExt, idx, pkgbits.SyncObject1)

	r.dict = dict
	rext.dict = dict

	sym = r.mangle(sym)
	if !sym.IsBlank() && sym.Def != nil {
		return sym.Def.(*ir.Name)
	}

	do := func(op ir.Op, hasTParams bool) *ir.Name {
		pos := r.pos()
		if hasTParams {
			r.typeParamNames()
		}

		name := ir.NewDeclNameAt(pos, op, sym)
		name.Class = ir.PEXTERN // may be overridden later
		if !sym.IsBlank() {
			if sym.Def != nil {
				base.FatalfAt(name.Pos(), "already have a definition for %v", name)
			}
			assert(sym.Def == nil)
			sym.Def = name
		}
		return name
	}

	switch tag {
	default:
		panic("unexpected object")

	case pkgbits.ObjAlias:
		name := do(ir.OTYPE, false)
		setType(name, r.typ())
		name.SetAlias(true)
		return name

	case pkgbits.ObjConst:
		name := do(ir.OLITERAL, false)
		typ := r.typ()
		val := FixValue(typ, r.Value())
		setType(name, typ)
		setValue(name, val)
		return name

	case pkgbits.ObjFunc:
		if sym.Name == "init" {
			sym = renameinit()
		}
		name := do(ir.ONAME, true)
		setType(name, r.signature(sym.Pkg, nil))

		name.Func = ir.NewFunc(r.pos())
		name.Func.Nname = name

		if r.hasTypeParams() {
			name.Func.SetDupok(true)
		}

		rext.funcExt(name)
		return name

	case pkgbits.ObjType:
		name := do(ir.OTYPE, true)
		typ := types.NewNamed(name)
		setType(name, typ)

		// Important: We need to do this before SetUnderlying.
		rext.typeExt(name)

		// We need to defer CheckSize until we've called SetUnderlying to
		// handle recursive types.
		types.DeferCheckSize()
		typ.SetUnderlying(r.typWrapped(false))
		types.ResumeCheckSize()

		methods := make([]*types.Field, r.Len())
		for i := range methods {
			methods[i] = r.method(rext)
		}
		if len(methods) != 0 {
			typ.Methods().Set(methods)
		}

		r.needWrapper(typ)

		return name

	case pkgbits.ObjVar:
		name := do(ir.ONAME, false)
		setType(name, r.typ())
		rext.varExt(name)
		return name
	}
}

func (r *reader) mangle(sym *types.Sym) *types.Sym {
	if !r.hasTypeParams() {
		return sym
	}

	var buf bytes.Buffer
	buf.WriteString(sym.Name)
	buf.WriteByte('[')
	for i, targ := range r.dict.targs {
		if i > 0 {
			if i == r.dict.implicits {
				buf.WriteByte(';')
			} else {
				buf.WriteByte(',')
			}
		}
		buf.WriteString(targ.LinkString())
	}
	buf.WriteByte(']')
	return sym.Pkg.Lookup(buf.String())
}

func (pr *pkgReader) objDictIdx(sym *types.Sym, idx int, implicits, explicits []*types.Type) *readerDict {
	r := pr.newReader(pkgbits.RelocObjDict, idx, pkgbits.SyncObject1)

	var dict readerDict

	nimplicits := r.Len()
	nexplicits := r.Len()

	if nimplicits > len(implicits) || nexplicits != len(explicits) {
		base.Fatalf("%v has %v+%v params, but instantiated with %v+%v args", sym, nimplicits, nexplicits, len(implicits), len(explicits))
	}

	dict.targs = append(implicits[:nimplicits:nimplicits], explicits...)
	dict.implicits = nimplicits

	// For stenciling, we can just skip over the type parameters.
	for range dict.targs[dict.implicits:] {
		// Skip past bounds without actually evaluating them.
		r.Sync(pkgbits.SyncType)
		if r.Bool() {
			r.Len()
		} else {
			r.Reloc(pkgbits.RelocType)
		}
	}

	dict.derived = make([]derivedInfo, r.Len())
	dict.derivedTypes = make([]*types.Type, len(dict.derived))
	for i := range dict.derived {
		dict.derived[i] = derivedInfo{r.Reloc(pkgbits.RelocType), r.Bool()}
	}

	dict.funcs = make([]objInfo, r.Len())
	dict.funcsObj = make([]ir.Node, len(dict.funcs))
	for i := range dict.funcs {
		objIdx := r.Reloc(pkgbits.RelocObj)
		targs := make([]typeInfo, r.Len())
		for j := range targs {
			targs[j] = r.typInfo()
		}
		dict.funcs[i] = objInfo{idx: objIdx, explicits: targs}
	}

	dict.itabs = make([]itabInfo2, r.Len())
	for i := range dict.itabs {
		typ := pr.typIdx(typeInfo{idx: r.Len(), derived: true}, &dict, true)
		ifaceInfo := r.typInfo()

		var lsym *obj.LSym
		if typ.IsInterface() {
			lsym = reflectdata.TypeLinksym(typ)
		} else {
			iface := pr.typIdx(ifaceInfo, &dict, true)
			lsym = reflectdata.ITabLsym(typ, iface)
		}

		dict.itabs[i] = itabInfo2{typ: typ, lsym: lsym}
	}

	return &dict
}

func (r *reader) typeParamNames() {
	r.Sync(pkgbits.SyncTypeParamNames)

	for range r.dict.targs[r.dict.implicits:] {
		r.pos()
		r.localIdent()
	}
}

func (r *reader) method(rext *reader) *types.Field {
	r.Sync(pkgbits.SyncMethod)
	pos := r.pos()
	pkg, sym := r.selector()
	r.typeParamNames()
	_, recv := r.param()
	typ := r.signature(pkg, recv)

	fnsym := sym
	fnsym = ir.MethodSym(recv.Type, fnsym)
	name := ir.NewNameAt(pos, fnsym)
	setType(name, typ)

	name.Func = ir.NewFunc(r.pos())
	name.Func.Nname = name

	if r.hasTypeParams() {
		name.Func.SetDupok(true)
	}

	rext.funcExt(name)

	meth := types.NewField(name.Func.Pos(), sym, typ)
	meth.Nname = name
	meth.SetNointerface(name.Func.Pragma&ir.Nointerface != 0)

	return meth
}

func (r *reader) qualifiedIdent() (pkg *types.Pkg, sym *types.Sym) {
	r.Sync(pkgbits.SyncSym)
	pkg = r.pkg()
	if name := r.String(); name != "" {
		sym = pkg.Lookup(name)
	}
	return
}

func (r *reader) localIdent() (pkg *types.Pkg, sym *types.Sym) {
	r.Sync(pkgbits.SyncLocalIdent)
	pkg = r.pkg()
	if name := r.String(); name != "" {
		sym = pkg.Lookup(name)
	}
	return
}

func (r *reader) selector() (origPkg *types.Pkg, sym *types.Sym) {
	r.Sync(pkgbits.SyncSelector)
	origPkg = r.pkg()
	name := r.String()
	pkg := origPkg
	if types.IsExported(name) {
		pkg = types.LocalPkg
	}
	sym = pkg.Lookup(name)
	return
}

func (r *reader) hasTypeParams() bool {
	return r.dict.hasTypeParams()
}

func (dict *readerDict) hasTypeParams() bool {
	return dict != nil && len(dict.targs) != 0
}

// @@@ Compiler extensions

func (r *reader) funcExt(name *ir.Name) {
	r.Sync(pkgbits.SyncFuncExt)

	name.Class = 0 // so MarkFunc doesn't complain
	ir.MarkFunc(name)

	fn := name.Func

	// XXX: Workaround because linker doesn't know how to copy Pos.
	if !fn.Pos().IsKnown() {
		fn.SetPos(name.Pos())
	}

	// Normally, we only compile local functions, which saves redundant compilation work.
	// n.Defn is not nil for local functions, and is nil for imported function. But for
	// generic functions, we might have an instantiation that no other package has seen before.
	// So we need to be conservative and compile it again.
	//
	// That's why name.Defn is set here, so ir.VisitFuncsBottomUp can analyze function.
	// TODO(mdempsky,cuonglm): find a cleaner way to handle this.
	if name.Sym().Pkg == types.LocalPkg || r.hasTypeParams() {
		name.Defn = fn
	}

	fn.Pragma = r.pragmaFlag()
	r.linkname(name)

	typecheck.Func(fn)

	if r.Bool() {
		fn.ABI = obj.ABI(r.Uint64())

		// Escape analysis.
		for _, fs := range &types.RecvsParams {
			for _, f := range fs(name.Type()).FieldSlice() {
				f.Note = r.String()
			}
		}

		if r.Bool() {
			fn.Inl = &ir.Inline{
				Cost:            int32(r.Len()),
				CanDelayResults: r.Bool(),
			}
			r.addBody(name.Func)
		}
	} else {
		r.addBody(name.Func)
	}
	r.Sync(pkgbits.SyncEOF)
}

func (r *reader) typeExt(name *ir.Name) {
	r.Sync(pkgbits.SyncTypeExt)

	typ := name.Type()

	if r.hasTypeParams() {
		// Set "RParams" (really type arguments here, not parameters) so
		// this type is treated as "fully instantiated". This ensures the
		// type descriptor is written out as DUPOK and method wrappers are
		// generated even for imported types.
		var targs []*types.Type
		targs = append(targs, r.dict.targs...)
		typ.SetRParams(targs)
	}

	name.SetPragma(r.pragmaFlag())
	if name.Pragma()&ir.NotInHeap != 0 {
		typ.SetNotInHeap(true)
	}

	typecheck.SetBaseTypeIndex(typ, r.Int64(), r.Int64())
}

func (r *reader) varExt(name *ir.Name) {
	r.Sync(pkgbits.SyncVarExt)
	r.linkname(name)
}

func (r *reader) linkname(name *ir.Name) {
	assert(name.Op() == ir.ONAME)
	r.Sync(pkgbits.SyncLinkname)

	if idx := r.Int64(); idx >= 0 {
		lsym := name.Linksym()
		lsym.SymIdx = int32(idx)
		lsym.Set(obj.AttrIndexed, true)
	} else {
		name.Sym().Linkname = r.String()
	}
}

func (r *reader) pragmaFlag() ir.PragmaFlag {
	r.Sync(pkgbits.SyncPragma)
	return ir.PragmaFlag(r.Int())
}

// @@@ Function bodies

// bodyReader tracks where the serialized IR for a function's body can
// be found.
var bodyReader = map[*ir.Func]pkgReaderIndex{}

// todoBodies holds the list of function bodies that still need to be
// constructed.
var todoBodies []*ir.Func

func (r *reader) addBody(fn *ir.Func) {
	pri := pkgReaderIndex{r.p, r.Reloc(pkgbits.RelocBody), r.dict}
	bodyReader[fn] = pri

	if fn.Nname.Defn == nil {
		// Don't read in function body for imported functions.
		// See comment in funcExt.
		return
	}

	if r.curfn == nil {
		todoBodies = append(todoBodies, fn)
		return
	}

	pri.funcBody(fn)
}

func (pri pkgReaderIndex) funcBody(fn *ir.Func) {
	r := pri.asReader(pkgbits.RelocBody, pkgbits.SyncFuncBody)
	r.funcBody(fn)
}

func (r *reader) funcBody(fn *ir.Func) {
	r.curfn = fn
	r.closureVars = fn.ClosureVars

	ir.WithFunc(fn, func() {
		r.funcargs(fn)

		if !r.Bool() {
			return
		}

		body := r.stmts()
		if body == nil {
			body = []ir.Node{typecheck.Stmt(ir.NewBlockStmt(src.NoXPos, nil))}
		}
		fn.Body = body
		fn.Endlineno = r.pos()
	})

	r.marker.WriteTo(fn)
}

func (r *reader) funcargs(fn *ir.Func) {
	sig := fn.Nname.Type()

	if recv := sig.Recv(); recv != nil {
		r.funcarg(recv, recv.Sym, ir.PPARAM)
	}
	for _, param := range sig.Params().FieldSlice() {
		r.funcarg(param, param.Sym, ir.PPARAM)
	}

	for i, param := range sig.Results().FieldSlice() {
		sym := types.OrigSym(param.Sym)

		if sym == nil || sym.IsBlank() {
			prefix := "~r"
			if r.inlCall != nil {
				prefix = "~R"
			} else if sym != nil {
				prefix = "~b"
			}
			sym = typecheck.LookupNum(prefix, i)
		}

		r.funcarg(param, sym, ir.PPARAMOUT)
	}
}

func (r *reader) funcarg(param *types.Field, sym *types.Sym, ctxt ir.Class) {
	if sym == nil {
		assert(ctxt == ir.PPARAM)
		if r.inlCall != nil {
			r.inlvars.Append(ir.BlankNode)
		}
		return
	}

	name := ir.NewNameAt(r.updatePos(param.Pos), sym)
	setType(name, param.Type)
	r.addLocal(name, ctxt)

	if r.inlCall == nil {
		if !r.funarghack {
			param.Sym = sym
			param.Nname = name
		}
	} else {
		if ctxt == ir.PPARAMOUT {
			r.retvars.Append(name)
		} else {
			r.inlvars.Append(name)
		}
	}
}

func (r *reader) addLocal(name *ir.Name, ctxt ir.Class) {
	assert(ctxt == ir.PAUTO || ctxt == ir.PPARAM || ctxt == ir.PPARAMOUT)

	r.Sync(pkgbits.SyncAddLocal)
	if pkgbits.EnableSync {
		want := r.Int()
		if have := len(r.locals); have != want {
			base.FatalfAt(name.Pos(), "locals table has desynced")
		}
	}

	name.SetUsed(true)
	r.locals = append(r.locals, name)

	// TODO(mdempsky): Move earlier.
	if ir.IsBlank(name) {
		return
	}

	if r.inlCall != nil {
		if ctxt == ir.PAUTO {
			name.SetInlLocal(true)
		} else {
			name.SetInlFormal(true)
			ctxt = ir.PAUTO
		}

		// TODO(mdempsky): Rethink this hack.
		if strings.HasPrefix(name.Sym().Name, "~") || base.Flag.GenDwarfInl == 0 {
			name.SetPos(r.inlCall.Pos())
			name.SetInlFormal(false)
			name.SetInlLocal(false)
		}
	}

	name.Class = ctxt
	name.Curfn = r.curfn

	r.curfn.Dcl = append(r.curfn.Dcl, name)

	if ctxt == ir.PAUTO {
		name.SetFrameOffset(0)
	}
}

func (r *reader) useLocal() *ir.Name {
	r.Sync(pkgbits.SyncUseObjLocal)
	if r.Bool() {
		return r.locals[r.Len()]
	}
	return r.closureVars[r.Len()]
}

func (r *reader) openScope() {
	r.Sync(pkgbits.SyncOpenScope)
	pos := r.pos()

	if base.Flag.Dwarf {
		r.scopeVars = append(r.scopeVars, len(r.curfn.Dcl))
		r.marker.Push(pos)
	}
}

func (r *reader) closeScope() {
	r.Sync(pkgbits.SyncCloseScope)
	r.lastCloseScopePos = r.pos()

	r.closeAnotherScope()
}

// closeAnotherScope is like closeScope, but it reuses the same mark
// position as the last closeScope call. This is useful for "for" and
// "if" statements, as their implicit blocks always end at the same
// position as an explicit block.
func (r *reader) closeAnotherScope() {
	r.Sync(pkgbits.SyncCloseAnotherScope)

	if base.Flag.Dwarf {
		scopeVars := r.scopeVars[len(r.scopeVars)-1]
		r.scopeVars = r.scopeVars[:len(r.scopeVars)-1]

		// Quirkish: noder decides which scopes to keep before
		// typechecking, whereas incremental typechecking during IR
		// construction can result in new autotemps being allocated. To
		// produce identical output, we ignore autotemps here for the
		// purpose of deciding whether to retract the scope.
		//
		// This is important for net/http/fcgi, because it contains:
		//
		//	var body io.ReadCloser
		//	if len(content) > 0 {
		//		body, req.pw = io.Pipe()
		//	} else { … }
		//
		// Notably, io.Pipe is inlinable, and inlining it introduces a ~R0
		// variable at the call site.
		//
		// Noder does not preserve the scope where the io.Pipe() call
		// resides, because it doesn't contain any declared variables in
		// source. So the ~R0 variable ends up being assigned to the
		// enclosing scope instead.
		//
		// However, typechecking this assignment also introduces
		// autotemps, because io.Pipe's results need conversion before
		// they can be assigned to their respective destination variables.
		//
		// TODO(mdempsky): We should probably just keep all scopes, and
		// let dwarfgen take care of pruning them instead.
		retract := true
		for _, n := range r.curfn.Dcl[scopeVars:] {
			if !n.AutoTemp() {
				retract = false
				break
			}
		}

		if retract {
			// no variables were declared in this scope, so we can retract it.
			r.marker.Unpush()
		} else {
			r.marker.Pop(r.lastCloseScopePos)
		}
	}
}

// @@@ Statements

func (r *reader) stmt() ir.Node {
	switch stmts := r.stmts(); len(stmts) {
	case 0:
		return nil
	case 1:
		return stmts[0]
	default:
		return ir.NewBlockStmt(stmts[0].Pos(), stmts)
	}
}

func (r *reader) stmts() []ir.Node {
	assert(ir.CurFunc == r.curfn)
	var res ir.Nodes

	r.Sync(pkgbits.SyncStmts)
	for {
		tag := codeStmt(r.Code(pkgbits.SyncStmt1))
		if tag == stmtEnd {
			r.Sync(pkgbits.SyncStmtsEnd)
			return res
		}

		if n := r.stmt1(tag, &res); n != nil {
			res.Append(typecheck.Stmt(n))
		}
	}
}

func (r *reader) stmt1(tag codeStmt, out *ir.Nodes) ir.Node {
	var label *types.Sym
	if n := len(*out); n > 0 {
		if ls, ok := (*out)[n-1].(*ir.LabelStmt); ok {
			label = ls.Label
		}
	}

	switch tag {
	default:
		panic("unexpected statement")

	case stmtAssign:
		pos := r.pos()

		// TODO(mdempsky): After quirks mode is gone, swap these
		// statements so we visit LHS before RHS again.
		rhs := r.exprList()
		names, lhs := r.assignList()

		if len(rhs) == 0 {
			for _, name := range names {
				as := ir.NewAssignStmt(pos, name, nil)
				as.PtrInit().Append(ir.NewDecl(pos, ir.ODCL, name))
				out.Append(typecheck.Stmt(as))
			}
			return nil
		}

		if len(lhs) == 1 && len(rhs) == 1 {
			n := ir.NewAssignStmt(pos, lhs[0], rhs[0])
			n.Def = r.initDefn(n, names)
			return n
		}

		n := ir.NewAssignListStmt(pos, ir.OAS2, lhs, rhs)
		n.Def = r.initDefn(n, names)
		return n

	case stmtAssignOp:
		op := r.op()
		lhs := r.expr()
		pos := r.pos()
		rhs := r.expr()
		return ir.NewAssignOpStmt(pos, op, lhs, rhs)

	case stmtIncDec:
		op := r.op()
		lhs := r.expr()
		pos := r.pos()
		n := ir.NewAssignOpStmt(pos, op, lhs, ir.NewBasicLit(pos, one))
		n.IncDec = true
		return n

	case stmtBlock:
		out.Append(r.blockStmt()...)
		return nil

	case stmtBranch:
		pos := r.pos()
		op := r.op()
		sym := r.optLabel()
		return ir.NewBranchStmt(pos, op, sym)

	case stmtCall:
		pos := r.pos()
		op := r.op()
		call := r.expr()
		return ir.NewGoDeferStmt(pos, op, call)

	case stmtExpr:
		return r.expr()

	case stmtFor:
		return r.forStmt(label)

	case stmtIf:
		return r.ifStmt()

	case stmtLabel:
		pos := r.pos()
		sym := r.label()
		return ir.NewLabelStmt(pos, sym)

	case stmtReturn:
		pos := r.pos()
		results := r.exprList()
		return ir.NewReturnStmt(pos, results)

	case stmtSelect:
		return r.selectStmt(label)

	case stmtSend:
		pos := r.pos()
		ch := r.expr()
		value := r.expr()
		return ir.NewSendStmt(pos, ch, value)

	case stmtSwitch:
		return r.switchStmt(label)
	}
}

func (r *reader) assignList() ([]*ir.Name, []ir.Node) {
	lhs := make([]ir.Node, r.Len())
	var names []*ir.Name

	for i := range lhs {
		if r.Bool() {
			pos := r.pos()
			_, sym := r.localIdent()
			typ := r.typ()

			name := ir.NewNameAt(pos, sym)
			lhs[i] = name
			names = append(names, name)
			setType(name, typ)
			r.addLocal(name, ir.PAUTO)
			continue
		}

		lhs[i] = r.expr()
	}

	return names, lhs
}

func (r *reader) blockStmt() []ir.Node {
	r.Sync(pkgbits.SyncBlockStmt)
	r.openScope()
	stmts := r.stmts()
	r.closeScope()
	return stmts
}

func (r *reader) forStmt(label *types.Sym) ir.Node {
	r.Sync(pkgbits.SyncForStmt)

	r.openScope()

	if r.Bool() {
		pos := r.pos()

		// TODO(mdempsky): After quirks mode is gone, swap these
		// statements so we read LHS before X again.
		x := r.expr()
		names, lhs := r.assignList()

		body := r.blockStmt()
		r.closeAnotherScope()

		rang := ir.NewRangeStmt(pos, nil, nil, x, body)
		if len(lhs) >= 1 {
			rang.Key = lhs[0]
			if len(lhs) >= 2 {
				rang.Value = lhs[1]
			}
		}
		rang.Def = r.initDefn(rang, names)
		rang.Label = label
		return rang
	}

	pos := r.pos()
	init := r.stmt()
	cond := r.expr()
	post := r.stmt()
	body := r.blockStmt()
	r.closeAnotherScope()

	stmt := ir.NewForStmt(pos, init, cond, post, body)
	stmt.Label = label
	return stmt
}

func (r *reader) ifStmt() ir.Node {
	r.Sync(pkgbits.SyncIfStmt)
	r.openScope()
	pos := r.pos()
	init := r.stmts()
	cond := r.expr()
	then := r.blockStmt()
	els := r.stmts()
	n := ir.NewIfStmt(pos, cond, then, els)
	n.SetInit(init)
	r.closeAnotherScope()
	return n
}

func (r *reader) selectStmt(label *types.Sym) ir.Node {
	r.Sync(pkgbits.SyncSelectStmt)

	pos := r.pos()
	clauses := make([]*ir.CommClause, r.Len())
	for i := range clauses {
		if i > 0 {
			r.closeScope()
		}
		r.openScope()

		pos := r.pos()
		comm := r.stmt()
		body := r.stmts()

		clauses[i] = ir.NewCommStmt(pos, comm, body)
	}
	if len(clauses) > 0 {
		r.closeScope()
	}
	n := ir.NewSelectStmt(pos, clauses)
	n.Label = label
	return n
}

func (r *reader) switchStmt(label *types.Sym) ir.Node {
	r.Sync(pkgbits.SyncSwitchStmt)

	r.openScope()
	pos := r.pos()
	init := r.stmt()

	var tag ir.Node
	var ident *ir.Ident
	var iface *types.Type
	if r.Bool() {
		pos := r.pos()
		if r.Bool() {
			pos := r.pos()
			sym := typecheck.Lookup(r.String())
			ident = ir.NewIdent(pos, sym)
		}
		x := r.expr()
		iface = x.Type()
		tag = ir.NewTypeSwitchGuard(pos, ident, x)
	} else {
		tag = r.expr()
	}

	clauses := make([]*ir.CaseClause, r.Len())
	for i := range clauses {
		if i > 0 {
			r.closeScope()
		}
		r.openScope()

		pos := r.pos()
		var cases []ir.Node
		if iface != nil {
			cases = make([]ir.Node, r.Len())
			if len(cases) == 0 {
				cases = nil // TODO(mdempsky): Unclear if this matters.
			}
			for i := range cases {
				cases[i] = r.exprType(true)
			}
		} else {
			cases = r.exprList()
		}

		clause := ir.NewCaseStmt(pos, cases, nil)

		if ident != nil {
			pos := r.pos()
			typ := r.typ()

			name := ir.NewNameAt(pos, ident.Sym())
			setType(name, typ)
			r.addLocal(name, ir.PAUTO)
			clause.Var = name
			name.Defn = tag
		}

		clause.Body = r.stmts()
		clauses[i] = clause
	}
	if len(clauses) > 0 {
		r.closeScope()
	}
	r.closeScope()

	n := ir.NewSwitchStmt(pos, tag, clauses)
	n.Label = label
	if init != nil {
		n.SetInit([]ir.Node{init})
	}
	return n
}

func (r *reader) label() *types.Sym {
	r.Sync(pkgbits.SyncLabel)
	name := r.String()
	if r.inlCall != nil {
		name = fmt.Sprintf("~%s·%d", name, inlgen)
	}
	return typecheck.Lookup(name)
}

func (r *reader) optLabel() *types.Sym {
	r.Sync(pkgbits.SyncOptLabel)
	if r.Bool() {
		return r.label()
	}
	return nil
}

// initDefn marks the given names as declared by defn and populates
// its Init field with ODCL nodes. It then reports whether any names
// were so declared, which can be used to initialize defn.Def.
func (r *reader) initDefn(defn ir.InitNode, names []*ir.Name) bool {
	if len(names) == 0 {
		return false
	}

	init := make([]ir.Node, len(names))
	for i, name := range names {
		name.Defn = defn
		init[i] = ir.NewDecl(name.Pos(), ir.ODCL, name)
	}
	defn.SetInit(init)
	return true
}

// @@@ Expressions

// expr reads and returns a typechecked expression.
func (r *reader) expr() (res ir.Node) {
	defer func() {
		if res != nil && res.Typecheck() == 0 {
			base.FatalfAt(res.Pos(), "%v missed typecheck", res)
		}
	}()

	switch tag := codeExpr(r.Code(pkgbits.SyncExpr)); tag {
	default:
		panic("unhandled expression")

	case exprNone:
		return nil

	case exprBlank:
		// blank only allowed in LHS of assignments
		// TODO(mdempsky): Handle directly in assignList instead?
		return typecheck.AssignExpr(ir.BlankNode)

	case exprLocal:
		return typecheck.Expr(r.useLocal())

	case exprName:
		// Callee instead of Expr allows builtins
		// TODO(mdempsky): Handle builtins directly in exprCall, like method calls?
		return typecheck.Callee(r.obj())

	case exprType:
		return r.exprType(false)

	case exprConst:
		pos := r.pos()
		typ := r.typ()
		val := FixValue(typ, r.Value())
		op := r.op()
		orig := r.String()
		return typecheck.Expr(OrigConst(pos, typ, val, op, orig))

	case exprCompLit:
		return r.compLit()

	case exprFuncLit:
		return r.funcLit()

	case exprSelector:
		x := r.expr()
		pos := r.pos()
		_, sym := r.selector()

		// Method expression with derived receiver type.
		if x.Op() == ir.ODYNAMICTYPE {
			// TODO(mdempsky): Handle with runtime dictionary lookup.
			n := ir.TypeNode(x.Type())
			n.SetTypecheck(1)
			x = n
		}

		n := typecheck.Expr(ir.NewSelectorExpr(pos, ir.OXDOT, x, sym)).(*ir.SelectorExpr)
		if n.Op() == ir.OMETHVALUE {
			wrapper := methodValueWrapper{
				rcvr:   n.X.Type(),
				method: n.Selection,
			}
			if r.importedDef() {
				haveMethodValueWrappers = append(haveMethodValueWrappers, wrapper)
			} else {
				needMethodValueWrappers = append(needMethodValueWrappers, wrapper)
			}
		}
		return n

	case exprIndex:
		x := r.expr()
		pos := r.pos()
		index := r.expr()
		return typecheck.Expr(ir.NewIndexExpr(pos, x, index))

	case exprSlice:
		x := r.expr()
		pos := r.pos()
		var index [3]ir.Node
		for i := range index {
			index[i] = r.expr()
		}
		op := ir.OSLICE
		if index[2] != nil {
			op = ir.OSLICE3
		}
		return typecheck.Expr(ir.NewSliceExpr(pos, op, x, index[0], index[1], index[2]))

	case exprAssert:
		x := r.expr()
		pos := r.pos()
		typ := r.exprType(false)

		if typ, ok := typ.(*ir.DynamicType); ok && typ.Op() == ir.ODYNAMICTYPE {
			return typed(typ.Type(), ir.NewDynamicTypeAssertExpr(pos, ir.ODYNAMICDOTTYPE, x, typ.X))
		}
		return typecheck.Expr(ir.NewTypeAssertExpr(pos, x, typ.(ir.Ntype)))

	case exprUnaryOp:
		op := r.op()
		pos := r.pos()
		x := r.expr()

		switch op {
		case ir.OADDR:
			return typecheck.Expr(typecheck.NodAddrAt(pos, x))
		case ir.ODEREF:
			return typecheck.Expr(ir.NewStarExpr(pos, x))
		}
		return typecheck.Expr(ir.NewUnaryExpr(pos, op, x))

	case exprBinaryOp:
		op := r.op()
		x := r.expr()
		pos := r.pos()
		y := r.expr()

		switch op {
		case ir.OANDAND, ir.OOROR:
			return typecheck.Expr(ir.NewLogicalExpr(pos, op, x, y))
		}
		return typecheck.Expr(ir.NewBinaryExpr(pos, op, x, y))

	case exprCall:
		fun := r.expr()
		if r.Bool() { // method call
			pos := r.pos()
			_, sym := r.selector()
			fun = typecheck.Callee(ir.NewSelectorExpr(pos, ir.OXDOT, fun, sym))
		}
		pos := r.pos()
		args := r.exprs()
		dots := r.Bool()
		return typecheck.Call(pos, fun, args, dots)

	case exprConvert:
		typ := r.typ()
		pos := r.pos()
		x := r.expr()

		// TODO(mdempsky): Stop constructing expressions of untyped type.
		x = typecheck.DefaultLit(x, typ)

		if op, why := typecheck.Convertop(x.Op() == ir.OLITERAL, x.Type(), typ); op == ir.OXXX {
			// types2 ensured that x is convertable to typ under standard Go
			// semantics, but cmd/compile also disallows some conversions
			// involving //go:notinheap.
			//
			// TODO(mdempsky): This can be removed after #46731 is implemented.
			base.ErrorfAt(pos, "cannot convert %L to type %v%v", x, typ, why)
			base.ErrorExit() // harsh, but prevents constructing invalid IR
		}

		return typecheck.Expr(ir.NewConvExpr(pos, ir.OCONV, typ, x))
	}
}

func (r *reader) compLit() ir.Node {
	r.Sync(pkgbits.SyncCompLit)
	pos := r.pos()
	typ0 := r.typ()

	typ := typ0
	if typ.IsPtr() {
		typ = typ.Elem()
	}
	if typ.Kind() == types.TFORW {
		base.FatalfAt(pos, "unresolved composite literal type: %v", typ)
	}
	isStruct := typ.Kind() == types.TSTRUCT

	elems := make([]ir.Node, r.Len())
	for i := range elems {
		elemp := &elems[i]

		if isStruct {
			sk := ir.NewStructKeyExpr(r.pos(), typ.Field(r.Len()), nil)
			*elemp, elemp = sk, &sk.Value
		} else if r.Bool() {
			kv := ir.NewKeyExpr(r.pos(), r.expr(), nil)
			*elemp, elemp = kv, &kv.Value
		}

		*elemp = wrapName(r.pos(), r.expr())
	}

	lit := typecheck.Expr(ir.NewCompLitExpr(pos, ir.OCOMPLIT, ir.TypeNode(typ), elems))
	if typ0.IsPtr() {
		lit = typecheck.Expr(typecheck.NodAddrAt(pos, lit))
		lit.SetType(typ0)
	}
	return lit
}

func wrapName(pos src.XPos, x ir.Node) ir.Node {
	// These nodes do not carry line numbers.
	// Introduce a wrapper node to give them the correct line.
	switch ir.Orig(x).Op() {
	case ir.OTYPE, ir.OLITERAL:
		if x.Sym() == nil {
			break
		}
		fallthrough
	case ir.ONAME, ir.ONONAME, ir.ONIL:
		p := ir.NewParenExpr(pos, x)
		p.SetImplicit(true)
		return p
	}
	return x
}

func (r *reader) funcLit() ir.Node {
	r.Sync(pkgbits.SyncFuncLit)

	pos := r.pos()
	xtype2 := r.signature(types.LocalPkg, nil)

	opos := pos

	fn := ir.NewClosureFunc(opos, r.curfn != nil)
	clo := fn.OClosure
	ir.NameClosure(clo, r.curfn)

	setType(fn.Nname, xtype2)
	typecheck.Func(fn)
	setType(clo, fn.Type())

	fn.ClosureVars = make([]*ir.Name, 0, r.Len())
	for len(fn.ClosureVars) < cap(fn.ClosureVars) {
		ir.NewClosureVar(r.pos(), fn, r.useLocal())
	}

	r.addBody(fn)

	// TODO(mdempsky): Remove hard-coding of typecheck.Target.
	return ir.UseClosure(clo, typecheck.Target)
}

func (r *reader) exprList() []ir.Node {
	r.Sync(pkgbits.SyncExprList)
	return r.exprs()
}

func (r *reader) exprs() []ir.Node {
	r.Sync(pkgbits.SyncExprs)
	nodes := make([]ir.Node, r.Len())
	if len(nodes) == 0 {
		return nil // TODO(mdempsky): Unclear if this matters.
	}
	for i := range nodes {
		nodes[i] = r.expr()
	}
	return nodes
}

func (r *reader) exprType(nilOK bool) ir.Node {
	r.Sync(pkgbits.SyncExprType)

	if nilOK && r.Bool() {
		return typecheck.Expr(types.BuiltinPkg.Lookup("nil").Def.(*ir.NilExpr))
	}

	pos := r.pos()

	var typ *types.Type
	var lsym *obj.LSym

	if r.Bool() {
		itab := r.dict.itabs[r.Len()]
		typ, lsym = itab.typ, itab.lsym
	} else {
		info := r.typInfo()
		typ = r.p.typIdx(info, r.dict, true)

		if !info.derived {
			// TODO(mdempsky): ir.TypeNode should probably return a typecheck'd node.
			n := ir.TypeNode(typ)
			n.SetTypecheck(1)
			return n
		}

		lsym = reflectdata.TypeLinksym(typ)
	}

	ptr := typecheck.Expr(typecheck.NodAddr(ir.NewLinksymExpr(pos, lsym, types.Types[types.TUINT8])))
	return typed(typ, ir.NewDynamicType(pos, ptr))
}

func (r *reader) op() ir.Op {
	r.Sync(pkgbits.SyncOp)
	return ir.Op(r.Len())
}

// @@@ Package initialization

func (r *reader) pkgInit(self *types.Pkg, target *ir.Package) {
	cgoPragmas := make([][]string, r.Len())
	for i := range cgoPragmas {
		cgoPragmas[i] = r.Strings()
	}
	target.CgoPragmas = cgoPragmas

	r.pkgDecls(target)

	r.Sync(pkgbits.SyncEOF)
}

func (r *reader) pkgDecls(target *ir.Package) {
	r.Sync(pkgbits.SyncDecls)
	for {
		switch code := codeDecl(r.Code(pkgbits.SyncDecl)); code {
		default:
			panic(fmt.Sprintf("unhandled decl: %v", code))

		case declEnd:
			return

		case declFunc:
			names := r.pkgObjs(target)
			assert(len(names) == 1)
			target.Decls = append(target.Decls, names[0].Func)

		case declMethod:
			typ := r.typ()
			_, sym := r.selector()

			method := typecheck.Lookdot1(nil, sym, typ, typ.Methods(), 0)
			target.Decls = append(target.Decls, method.Nname.(*ir.Name).Func)

		case declVar:
			pos := r.pos()
			names := r.pkgObjs(target)
			values := r.exprList()

			if len(names) > 1 && len(values) == 1 {
				as := ir.NewAssignListStmt(pos, ir.OAS2, nil, values)
				for _, name := range names {
					as.Lhs.Append(name)
					name.Defn = as
				}
				target.Decls = append(target.Decls, as)
			} else {
				for i, name := range names {
					as := ir.NewAssignStmt(pos, name, nil)
					if i < len(values) {
						as.Y = values[i]
					}
					name.Defn = as
					target.Decls = append(target.Decls, as)
				}
			}

			if n := r.Len(); n > 0 {
				assert(len(names) == 1)
				embeds := make([]ir.Embed, n)
				for i := range embeds {
					embeds[i] = ir.Embed{Pos: r.pos(), Patterns: r.Strings()}
				}
				names[0].Embed = &embeds
				target.Embeds = append(target.Embeds, names[0])
			}

		case declOther:
			r.pkgObjs(target)
		}
	}
}

func (r *reader) pkgObjs(target *ir.Package) []*ir.Name {
	r.Sync(pkgbits.SyncDeclNames)
	nodes := make([]*ir.Name, r.Len())
	for i := range nodes {
		r.Sync(pkgbits.SyncDeclName)

		name := r.obj().(*ir.Name)
		nodes[i] = name

		sym := name.Sym()
		if sym.IsBlank() {
			continue
		}

		switch name.Class {
		default:
			base.FatalfAt(name.Pos(), "unexpected class: %v", name.Class)

		case ir.PEXTERN:
			target.Externs = append(target.Externs, name)

		case ir.PFUNC:
			assert(name.Type().Recv() == nil)

			// TODO(mdempsky): Cleaner way to recognize init?
			if strings.HasPrefix(sym.Name, "init.") {
				target.Inits = append(target.Inits, name.Func)
			}
		}

		if types.IsExported(sym.Name) {
			assert(!sym.OnExportList())
			target.Exports = append(target.Exports, name)
			sym.SetOnExportList(true)
		}

		if base.Flag.AsmHdr != "" {
			assert(!sym.Asm())
			target.Asms = append(target.Asms, name)
			sym.SetAsm(true)
		}
	}

	return nodes
}

// @@@ Inlining

var inlgen = 0

func InlineCall(call *ir.CallExpr, fn *ir.Func, inlIndex int) *ir.InlinedCallExpr {
	// TODO(mdempsky): Turn callerfn into an explicit parameter.
	callerfn := ir.CurFunc

	pri, ok := bodyReader[fn]
	if !ok {
		base.FatalfAt(call.Pos(), "missing function body for call to %v", fn)
	}

	if fn.Inl.Body == nil {
		expandInline(fn, pri)
	}

	r := pri.asReader(pkgbits.RelocBody, pkgbits.SyncFuncBody)

	// TODO(mdempsky): This still feels clumsy. Can we do better?
	tmpfn := ir.NewFunc(fn.Pos())
	tmpfn.Nname = ir.NewNameAt(fn.Nname.Pos(), callerfn.Sym())
	tmpfn.Closgen = callerfn.Closgen
	defer func() { callerfn.Closgen = tmpfn.Closgen }()

	setType(tmpfn.Nname, fn.Type())
	r.curfn = tmpfn

	r.inlCaller = callerfn
	r.inlCall = call
	r.inlFunc = fn
	r.inlTreeIndex = inlIndex
	r.inlPosBases = make(map[*src.PosBase]*src.PosBase)

	r.closureVars = make([]*ir.Name, len(r.inlFunc.ClosureVars))
	for i, cv := range r.inlFunc.ClosureVars {
		r.closureVars[i] = cv.Outer
	}

	r.funcargs(fn)

	assert(r.Bool()) // have body
	r.delayResults = fn.Inl.CanDelayResults

	r.retlabel = typecheck.AutoLabel(".i")
	inlgen++

	init := ir.TakeInit(call)

	// For normal function calls, the function callee expression
	// may contain side effects. Make sure to preserve these,
	// if necessary (#42703).
	if call.Op() == ir.OCALLFUNC {
		inline.CalleeEffects(&init, call.X)
	}

	var args ir.Nodes
	if call.Op() == ir.OCALLMETH {
		base.FatalfAt(call.Pos(), "OCALLMETH missed by typecheck")
	}
	args.Append(call.Args...)

	// Create assignment to declare and initialize inlvars.
	as2 := ir.NewAssignListStmt(call.Pos(), ir.OAS2, r.inlvars, args)
	as2.Def = true
	var as2init ir.Nodes
	for _, name := range r.inlvars {
		if ir.IsBlank(name) {
			continue
		}
		// TODO(mdempsky): Use inlined position of name.Pos() instead?
		name := name.(*ir.Name)
		as2init.Append(ir.NewDecl(call.Pos(), ir.ODCL, name))
		name.Defn = as2
	}
	as2.SetInit(as2init)
	init.Append(typecheck.Stmt(as2))

	if !r.delayResults {
		// If not delaying retvars, declare and zero initialize the
		// result variables now.
		for _, name := range r.retvars {
			// TODO(mdempsky): Use inlined position of name.Pos() instead?
			name := name.(*ir.Name)
			init.Append(ir.NewDecl(call.Pos(), ir.ODCL, name))
			ras := ir.NewAssignStmt(call.Pos(), name, nil)
			init.Append(typecheck.Stmt(ras))
		}
	}

	// Add an inline mark just before the inlined body.
	// This mark is inline in the code so that it's a reasonable spot
	// to put a breakpoint. Not sure if that's really necessary or not
	// (in which case it could go at the end of the function instead).
	// Note issue 28603.
	init.Append(ir.NewInlineMarkStmt(call.Pos().WithIsStmt(), int64(r.inlTreeIndex)))

	nparams := len(r.curfn.Dcl)

	ir.WithFunc(r.curfn, func() {
		r.curfn.Body = r.stmts()
		r.curfn.Endlineno = r.pos()

		// TODO(mdempsky): This shouldn't be necessary. Inlining might
		// read in new function/method declarations, which could
		// potentially be recursively inlined themselves; but we shouldn't
		// need to read in the non-inlined bodies for the declarations
		// themselves. But currently it's an easy fix to #50552.
		readBodies(typecheck.Target)

		deadcode.Func(r.curfn)

		// Replace any "return" statements within the function body.
		var edit func(ir.Node) ir.Node
		edit = func(n ir.Node) ir.Node {
			if ret, ok := n.(*ir.ReturnStmt); ok {
				n = typecheck.Stmt(r.inlReturn(ret))
			}
			ir.EditChildren(n, edit)
			return n
		}
		edit(r.curfn)
	})

	body := ir.Nodes(r.curfn.Body)

	// Quirkish: We need to eagerly prune variables added during
	// inlining, but removed by deadcode.FuncBody above. Unused
	// variables will get removed during stack frame layout anyway, but
	// len(fn.Dcl) ends up influencing things like autotmp naming.

	used := usedLocals(body)

	for i, name := range r.curfn.Dcl {
		if i < nparams || used.Has(name) {
			name.Curfn = callerfn
			callerfn.Dcl = append(callerfn.Dcl, name)

			// Quirkish. TODO(mdempsky): Document why.
			if name.AutoTemp() {
				name.SetEsc(ir.EscUnknown)

				if base.Flag.GenDwarfInl != 0 {
					name.SetInlLocal(true)
				} else {
					name.SetPos(r.inlCall.Pos())
				}
			}
		}
	}

	body.Append(ir.NewLabelStmt(call.Pos(), r.retlabel))

	res := ir.NewInlinedCallExpr(call.Pos(), body, append([]ir.Node(nil), r.retvars...))
	res.SetInit(init)
	res.SetType(call.Type())
	res.SetTypecheck(1)

	// Inlining shouldn't add any functions to todoBodies.
	assert(len(todoBodies) == 0)

	return res
}

// inlReturn returns a statement that can substitute for the given
// return statement when inlining.
func (r *reader) inlReturn(ret *ir.ReturnStmt) *ir.BlockStmt {
	pos := r.inlCall.Pos()

	block := ir.TakeInit(ret)

	if results := ret.Results; len(results) != 0 {
		assert(len(r.retvars) == len(results))

		as2 := ir.NewAssignListStmt(pos, ir.OAS2, append([]ir.Node(nil), r.retvars...), ret.Results)

		if r.delayResults {
			for _, name := range r.retvars {
				// TODO(mdempsky): Use inlined position of name.Pos() instead?
				name := name.(*ir.Name)
				block.Append(ir.NewDecl(pos, ir.ODCL, name))
				name.Defn = as2
			}
		}

		block.Append(as2)
	}

	block.Append(ir.NewBranchStmt(pos, ir.OGOTO, r.retlabel))
	return ir.NewBlockStmt(pos, block)
}

// expandInline reads in an extra copy of IR to populate
// fn.Inl.{Dcl,Body}.
func expandInline(fn *ir.Func, pri pkgReaderIndex) {
	// TODO(mdempsky): Remove this function. It's currently needed by
	// dwarfgen/dwarf.go:preInliningDcls, which requires fn.Inl.Dcl to
	// create abstract function DIEs. But we should be able to provide it
	// with the same information some other way.

	fndcls := len(fn.Dcl)
	topdcls := len(typecheck.Target.Decls)

	tmpfn := ir.NewFunc(fn.Pos())
	tmpfn.Nname = ir.NewNameAt(fn.Nname.Pos(), fn.Sym())
	tmpfn.ClosureVars = fn.ClosureVars

	{
		r := pri.asReader(pkgbits.RelocBody, pkgbits.SyncFuncBody)
		setType(tmpfn.Nname, fn.Type())

		// Don't change parameter's Sym/Nname fields.
		r.funarghack = true

		r.funcBody(tmpfn)

		ir.WithFunc(tmpfn, func() {
			deadcode.Func(tmpfn)
		})
	}

	used := usedLocals(tmpfn.Body)

	for _, name := range tmpfn.Dcl {
		if name.Class != ir.PAUTO || used.Has(name) {
			name.Curfn = fn
			fn.Inl.Dcl = append(fn.Inl.Dcl, name)
		}
	}
	fn.Inl.Body = tmpfn.Body

	// Double check that we didn't change fn.Dcl by accident.
	assert(fndcls == len(fn.Dcl))

	// typecheck.Stmts may have added function literals to
	// typecheck.Target.Decls. Remove them again so we don't risk trying
	// to compile them multiple times.
	typecheck.Target.Decls = typecheck.Target.Decls[:topdcls]
}

// usedLocals returns a set of local variables that are used within body.
func usedLocals(body []ir.Node) ir.NameSet {
	var used ir.NameSet
	ir.VisitList(body, func(n ir.Node) {
		if n, ok := n.(*ir.Name); ok && n.Op() == ir.ONAME && n.Class == ir.PAUTO {
			used.Add(n)
		}
	})
	return used
}

// @@@ Method wrappers

// needWrapperTypes lists types for which we may need to generate
// method wrappers.
var needWrapperTypes []*types.Type

// haveWrapperTypes lists types for which we know we already have
// method wrappers, because we found the type in an imported package.
var haveWrapperTypes []*types.Type

// needMethodValueWrappers lists methods for which we may need to
// generate method value wrappers.
var needMethodValueWrappers []methodValueWrapper

// haveMethodValueWrappers lists methods for which we know we already
// have method value wrappers, because we found it in an imported
// package.
var haveMethodValueWrappers []methodValueWrapper

type methodValueWrapper struct {
	rcvr   *types.Type
	method *types.Field
}

func (r *reader) needWrapper(typ *types.Type) {
	if typ.IsPtr() {
		return
	}

	// If a type was found in an imported package, then we can assume
	// that package (or one of its transitive dependencies) already
	// generated method wrappers for it.
	if r.importedDef() {
		haveWrapperTypes = append(haveWrapperTypes, typ)
	} else {
		needWrapperTypes = append(needWrapperTypes, typ)
	}
}

func (r *reader) importedDef() bool {
	// If a type was found in an imported package, then we can assume
	// that package (or one of its transitive dependencies) already
	// generated method wrappers for it.
	//
	// Exception: If we're instantiating an imported generic type or
	// function, we might be instantiating it with type arguments not
	// previously seen before.
	//
	// TODO(mdempsky): Distinguish when a generic function or type was
	// instantiated in an imported package so that we can add types to
	// haveWrapperTypes instead.
	return r.p != localPkgReader && !r.hasTypeParams()
}

func MakeWrappers(target *ir.Package) {
	// Only unified IR emits its own wrappers.
	if base.Debug.Unified == 0 {
		return
	}

	// always generate a wrapper for error.Error (#29304)
	needWrapperTypes = append(needWrapperTypes, types.ErrorType)

	seen := make(map[string]*types.Type)

	for _, typ := range haveWrapperTypes {
		wrapType(typ, target, seen, false)
	}
	haveWrapperTypes = nil

	for _, typ := range needWrapperTypes {
		wrapType(typ, target, seen, true)
	}
	needWrapperTypes = nil

	for _, wrapper := range haveMethodValueWrappers {
		wrapMethodValue(wrapper.rcvr, wrapper.method, target, false)
	}
	haveMethodValueWrappers = nil

	for _, wrapper := range needMethodValueWrappers {
		wrapMethodValue(wrapper.rcvr, wrapper.method, target, true)
	}
	needMethodValueWrappers = nil
}

func wrapType(typ *types.Type, target *ir.Package, seen map[string]*types.Type, needed bool) {
	key := typ.LinkString()
	if prev := seen[key]; prev != nil {
		if !types.Identical(typ, prev) {
			base.Fatalf("collision: types %v and %v have link string %q", typ, prev, key)
		}
		return
	}
	seen[key] = typ

	if !needed {
		// Only called to add to 'seen'.
		return
	}

	if !typ.IsInterface() {
		typecheck.CalcMethods(typ)
	}
	for _, meth := range typ.AllMethods().Slice() {
		if meth.Sym.IsBlank() || !meth.IsMethod() {
			base.FatalfAt(meth.Pos, "invalid method: %v", meth)
		}

		methodWrapper(0, typ, meth, target)

		// For non-interface types, we also want *T wrappers.
		if !typ.IsInterface() {
			methodWrapper(1, typ, meth, target)

			// For not-in-heap types, *T is a scalar, not pointer shaped,
			// so the interface wrappers use **T.
			if typ.NotInHeap() {
				methodWrapper(2, typ, meth, target)
			}
		}
	}
}

func methodWrapper(derefs int, tbase *types.Type, method *types.Field, target *ir.Package) {
	wrapper := tbase
	for i := 0; i < derefs; i++ {
		wrapper = types.NewPtr(wrapper)
	}

	sym := ir.MethodSym(wrapper, method.Sym)
	base.Assertf(!sym.Siggen(), "already generated wrapper %v", sym)
	sym.SetSiggen(true)

	wrappee := method.Type.Recv().Type
	if types.Identical(wrapper, wrappee) ||
		!types.IsMethodApplicable(wrapper, method) ||
		!reflectdata.NeedEmit(tbase) {
		return
	}

	// TODO(mdempsky): Use method.Pos instead?
	pos := base.AutogeneratedPos

	fn := newWrapperFunc(pos, sym, wrapper, method)

	var recv ir.Node = fn.Nname.Type().Recv().Nname.(*ir.Name)

	// For simple *T wrappers around T methods, panicwrap produces a
	// nicer panic message.
	if wrapper.IsPtr() && types.Identical(wrapper.Elem(), wrappee) {
		cond := ir.NewBinaryExpr(pos, ir.OEQ, recv, types.BuiltinPkg.Lookup("nil").Def.(ir.Node))
		then := []ir.Node{ir.NewCallExpr(pos, ir.OCALL, typecheck.LookupRuntime("panicwrap"), nil)}
		fn.Body.Append(ir.NewIfStmt(pos, cond, then, nil))
	}

	// typecheck will add one implicit deref, if necessary,
	// but not-in-heap types require more for their **T wrappers.
	for i := 1; i < derefs; i++ {
		recv = Implicit(ir.NewStarExpr(pos, recv))
	}

	addTailCall(pos, fn, recv, method)

	finishWrapperFunc(fn, target)
}

func wrapMethodValue(recvType *types.Type, method *types.Field, target *ir.Package, needed bool) {
	sym := ir.MethodSymSuffix(recvType, method.Sym, "-fm")
	if sym.Uniq() {
		return
	}
	sym.SetUniq(true)

	// TODO(mdempsky): Use method.Pos instead?
	pos := base.AutogeneratedPos

	fn := newWrapperFunc(pos, sym, nil, method)
	sym.Def = fn.Nname

	// Declare and initialize variable holding receiver.
	recv := ir.NewHiddenParam(pos, fn, typecheck.Lookup(".this"), recvType)

	if !needed {
		typecheck.Func(fn)
		return
	}

	addTailCall(pos, fn, recv, method)

	finishWrapperFunc(fn, target)
}

func newWrapperFunc(pos src.XPos, sym *types.Sym, wrapper *types.Type, method *types.Field) *ir.Func {
	fn := ir.NewFunc(pos)
	fn.SetDupok(true) // TODO(mdempsky): Leave unset for local, non-generic wrappers?

	name := ir.NewNameAt(pos, sym)
	ir.MarkFunc(name)
	name.Func = fn
	name.Defn = fn
	fn.Nname = name

	sig := newWrapperType(wrapper, method)
	setType(name, sig)

	// TODO(mdempsky): De-duplicate with similar logic in funcargs.
	defParams := func(class ir.Class, params *types.Type) {
		for _, param := range params.FieldSlice() {
			name := ir.NewNameAt(param.Pos, param.Sym)
			name.Class = class
			setType(name, param.Type)

			name.Curfn = fn
			fn.Dcl = append(fn.Dcl, name)

			param.Nname = name
		}
	}

	defParams(ir.PPARAM, sig.Recvs())
	defParams(ir.PPARAM, sig.Params())
	defParams(ir.PPARAMOUT, sig.Results())

	return fn
}

func finishWrapperFunc(fn *ir.Func, target *ir.Package) {
	typecheck.Func(fn)

	ir.WithFunc(fn, func() {
		typecheck.Stmts(fn.Body)
	})

	// We generate wrappers after the global inlining pass,
	// so we're responsible for applying inlining ourselves here.
	inline.InlineCalls(fn)

	target.Decls = append(target.Decls, fn)
}

// newWrapperType returns a copy of the given signature type, but with
// the receiver parameter type substituted with recvType.
// If recvType is nil, newWrapperType returns a signature
// without a receiver parameter.
func newWrapperType(recvType *types.Type, method *types.Field) *types.Type {
	clone := func(params []*types.Field) []*types.Field {
		res := make([]*types.Field, len(params))
		for i, param := range params {
			sym := param.Sym
			if sym == nil || sym.Name == "_" {
				sym = typecheck.LookupNum(".anon", i)
			}
			res[i] = types.NewField(param.Pos, sym, param.Type)
			res[i].SetIsDDD(param.IsDDD())
		}
		return res
	}

	sig := method.Type

	var recv *types.Field
	if recvType != nil {
		recv = types.NewField(sig.Recv().Pos, typecheck.Lookup(".this"), recvType)
	}
	params := clone(sig.Params().FieldSlice())
	results := clone(sig.Results().FieldSlice())

	return types.NewSignature(types.NoPkg, recv, nil, params, results)
}

func addTailCall(pos src.XPos, fn *ir.Func, recv ir.Node, method *types.Field) {
	sig := fn.Nname.Type()
	args := make([]ir.Node, sig.NumParams())
	for i, param := range sig.Params().FieldSlice() {
		args[i] = param.Nname.(*ir.Name)
	}

	// TODO(mdempsky): Support creating OTAILCALL, when possible. See reflectdata.methodWrapper.
	// Not urgent though, because tail calls are currently incompatible with regabi anyway.

	fn.SetWrapper(true) // TODO(mdempsky): Leave unset for tail calls?

	dot := ir.NewSelectorExpr(pos, ir.OXDOT, recv, method.Sym)
	call := typecheck.Call(pos, dot, args, method.Type.IsVariadic()).(*ir.CallExpr)

	if method.Type.NumResults() == 0 {
		fn.Body.Append(call)
		return
	}

	ret := ir.NewReturnStmt(pos, nil)
	ret.Results = []ir.Node{call}
	fn.Body.Append(ret)
}
