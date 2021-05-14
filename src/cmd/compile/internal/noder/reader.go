// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"fmt"
	"go/constant"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/deadcode"
	"cmd/compile/internal/dwarfgen"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// TODO(mdempsky): Suppress duplicate type/const errors that can arise
// during typecheck due to naive type substitution (e.g., see #42758).
// I anticipate these will be handled as a consequence of adding
// dictionaries support, so it's probably not important to focus on
// this until after that's done.

type pkgReader struct {
	pkgDecoder

	posBases []*src.PosBase
	pkgs     []*types.Pkg
	typs     []*types.Type

	// offset for rewriting the given index into the output,
	// but bitwise inverted so we can detect if we're missing the entry or not.
	newindex []int
}

func newPkgReader(pr pkgDecoder) *pkgReader {
	return &pkgReader{
		pkgDecoder: pr,

		posBases: make([]*src.PosBase, pr.numElems(relocPosBase)),
		pkgs:     make([]*types.Pkg, pr.numElems(relocPkg)),
		typs:     make([]*types.Type, pr.numElems(relocType)),

		newindex: make([]int, pr.totalElems()),
	}
}

type pkgReaderIndex struct {
	pr        *pkgReader
	idx       int
	implicits []*types.Type
}

func (pri pkgReaderIndex) asReader(k reloc, marker syncMarker) *reader {
	r := pri.pr.newReader(k, pri.idx, marker)
	r.implicits = pri.implicits
	return r
}

func (pr *pkgReader) newReader(k reloc, idx int, marker syncMarker) *reader {
	return &reader{
		decoder: pr.newDecoder(k, idx, marker),
		p:       pr,
	}
}

type reader struct {
	decoder

	p *pkgReader

	// Implicit and explicit type arguments in use for reading the
	// current object. For example:
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
	implicits []*types.Type
	explicits []*types.Type

	ext *reader

	// TODO(mdempsky): The state below is all specific to reading
	// function bodies. It probably makes sense to split it out
	// separately so that it doesn't take up space in every reader
	// instance.

	curfn  *ir.Func
	locals []*ir.Name

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

func (r *reader) setType(n ir.Node, typ *types.Type) {
	n.SetType(typ)
	n.SetTypecheck(1)

	if name, ok := n.(*ir.Name); ok {
		name.SetWalkdef(1)
		name.Ntype = ir.TypeNode(name.Type())
	}
}

func (r *reader) setValue(name *ir.Name, val constant.Value) {
	name.SetVal(val)
	name.Defn = nil
}

// @@@ Positions

func (r *reader) pos() src.XPos {
	return base.Ctxt.PosTable.XPos(r.pos0())
}

func (r *reader) pos0() src.Pos {
	r.sync(syncPos)
	if !r.bool() {
		return src.NoPos
	}

	posBase := r.posBase()
	line := r.uint()
	col := r.uint()
	return src.MakePos(posBase, line, col)
}

func (r *reader) posBase() *src.PosBase {
	return r.inlPosBase(r.p.posBaseIdx(r.reloc(relocPosBase)))
}

func (pr *pkgReader) posBaseIdx(idx int) *src.PosBase {
	if b := pr.posBases[idx]; b != nil {
		return b
	}

	r := pr.newReader(relocPosBase, idx, syncPosBase)
	var b *src.PosBase

	fn := r.string()
	absfn := r.string()

	if r.bool() {
		b = src.NewFileBase(fn, absfn)
	} else {
		pos := r.pos0()
		line := r.uint()
		col := r.uint()
		b = src.NewLinePragmaBase(pos, fn, absfn, line, col)
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
	r.sync(syncPkg)
	return r.p.pkgIdx(r.reloc(relocPkg))
}

func (pr *pkgReader) pkgIdx(idx int) *types.Pkg {
	if pkg := pr.pkgs[idx]; pkg != nil {
		return pkg
	}

	pkg := pr.newReader(relocPkg, idx, syncPkgDef).doPkg()
	pr.pkgs[idx] = pkg
	return pkg
}

func (r *reader) doPkg() *types.Pkg {
	path := r.string()
	if path == "builtin" {
		return types.BuiltinPkg
	}
	if path == "" {
		path = r.p.pkgPath
	}

	name := r.string()
	height := r.len()

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
	r.sync(syncType)
	return r.p.typIdx(r.reloc(relocType), r.implicits, r.explicits)
}

func (pr *pkgReader) typIdx(idx int, implicits, explicits []*types.Type) *types.Type {
	if typ := pr.typs[idx]; typ != nil {
		return typ
	}

	r := pr.newReader(relocType, idx, syncTypeIdx)
	r.implicits = implicits
	r.explicits = explicits
	typ := r.doTyp()
	assert(typ != nil)

	if typ := pr.typs[idx]; typ != nil {
		// This happens in fixedbugs/issue27232.go.
		// TODO(mdempsky): Explain why/how this happens.
		return typ
	}

	// If we have type parameters, the type might refer to them, and it
	// wouldn't be safe to reuse those in other contexts. So we
	// conservatively avoid caching them in that case.
	//
	// TODO(mdempsky): If we're clever, we should be able to still cache
	// types by tracking which type parameters are used. However, in my
	// attempts so far, I haven't yet succeeded in being clever enough.
	if len(implicits)+len(explicits) == 0 {
		pr.typs[idx] = typ
	}

	if !typ.IsUntyped() {
		types.CheckSize(typ)
	}

	return typ
}

func (r *reader) doTyp() *types.Type {
	switch tag := codeType(r.code(syncType)); tag {
	default:
		panic(fmt.Sprintf("unexpected type: %v", tag))

	case typeBasic:
		return *basics[r.len()]

	case typeNamed:
		obj := r.obj()
		assert(obj.Op() == ir.OTYPE)
		return obj.Type()

	case typeTypeParam:
		idx := r.len()
		if idx < len(r.implicits) {
			return r.implicits[idx]
		}
		return r.explicits[idx-len(r.implicits)]

	case typeArray:
		len := int64(r.uint64())
		return types.NewArray(r.typ(), len)
	case typeChan:
		dir := dirs[r.len()]
		return types.NewChan(r.typ(), dir)
	case typeMap:
		return types.NewMap(r.typ(), r.typ())
	case typePointer:
		return types.NewPtr(r.typ())
	case typeSignature:
		return r.signature(types.LocalPkg, nil)
	case typeSlice:
		return types.NewSlice(r.typ())
	case typeStruct:
		return r.structType()
	case typeInterface:
		return r.interfaceType()
	}
}

func (r *reader) interfaceType() *types.Type {
	tpkg := types.LocalPkg // TODO(mdempsky): Remove after iexport is gone.

	nmethods, nembeddeds := r.len(), r.len()

	fields := make([]*types.Field, nmethods+nembeddeds)
	methods, embeddeds := fields[:nmethods], fields[nmethods:]

	for i := range methods {
		pos := r.pos()
		pkg, sym := r.selector()
		tpkg = pkg
		mtyp := r.signature(pkg, typecheck.FakeRecv())
		methods[i] = types.NewField(pos, sym, mtyp)
	}
	for i := range embeddeds {
		embeddeds[i] = types.NewField(src.NoXPos, nil, r.typ())
	}

	if len(fields) == 0 {
		return types.Types[types.TINTER] // empty interface
	}
	return types.NewInterface(tpkg, fields)
}

func (r *reader) structType() *types.Type {
	tpkg := types.LocalPkg // TODO(mdempsky): Remove after iexport is gone.
	fields := make([]*types.Field, r.len())
	for i := range fields {
		pos := r.pos()
		pkg, sym := r.selector()
		tpkg = pkg
		ftyp := r.typ()
		tag := r.string()
		embedded := r.bool()

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
	r.sync(syncSignature)

	params := r.params(&tpkg)
	results := r.params(&tpkg)
	if r.bool() { // variadic
		params[len(params)-1].SetIsDDD(true)
	}

	return types.NewSignature(tpkg, recv, nil, params, results)
}

func (r *reader) params(tpkg **types.Pkg) []*types.Field {
	r.sync(syncParams)
	fields := make([]*types.Field, r.len())
	for i := range fields {
		*tpkg, fields[i] = r.param()
	}
	return fields
}

func (r *reader) param() (*types.Pkg, *types.Field) {
	r.sync(syncParam)

	pos := r.pos()
	pkg, sym := r.localIdent()
	typ := r.typ()

	return pkg, types.NewField(pos, sym, typ)
}

// @@@ Objects

var objReader = map[*types.Sym]pkgReaderIndex{}

func (r *reader) obj() ir.Node {
	r.sync(syncObject)

	idx := r.reloc(relocObj)

	explicits := make([]*types.Type, r.len())
	for i := range explicits {
		explicits[i] = r.typ()
	}

	return r.p.objIdx(idx, r.implicits, explicits)
}

func (pr *pkgReader) objIdx(idx int, implicits, explicits []*types.Type) ir.Node {
	r := pr.newReader(relocObj, idx, syncObject1)
	r.ext = pr.newReader(relocObjExt, idx, syncObject1)

	_, sym := r.qualifiedIdent()

	// Middle dot indicates local defined type; see writer.sym.
	// TODO(mdempsky): Come up with a better way to handle this.
	if strings.Contains(sym.Name, "·") {
		r.implicits = implicits
		r.ext.implicits = implicits
	}
	r.explicits = explicits
	r.ext.explicits = explicits

	origSym := sym

	sym = r.mangle(sym)
	if !sym.IsBlank() && sym.Def != nil {
		return sym.Def.(ir.Node)
	}

	r.typeParamBounds(origSym)
	tag := codeObj(r.code(syncCodeObj))

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

	case objStub:
		if pri, ok := objReader[origSym]; ok {
			return pri.pr.objIdx(pri.idx, pri.implicits, r.explicits)
		}
		if haveLegacyImports {
			assert(len(r.implicits)+len(r.explicits) == 0)
			return typecheck.Resolve(ir.NewIdent(src.NoXPos, origSym))
		}
		base.Fatalf("unresolved stub: %v", origSym)
		panic("unreachable")

	case objAlias:
		name := do(ir.OTYPE, false)
		r.setType(name, r.typ())
		name.SetAlias(true)
		return name

	case objConst:
		name := do(ir.OLITERAL, false)
		typ, val := r.value()
		r.setType(name, typ)
		r.setValue(name, val)
		return name

	case objFunc:
		if sym.Name == "init" {
			sym = renameinit()
		}
		name := do(ir.ONAME, true)
		r.setType(name, r.signature(sym.Pkg, nil))

		name.Func = ir.NewFunc(r.pos())
		name.Func.Nname = name

		r.ext.funcExt(name)
		return name

	case objType:
		name := do(ir.OTYPE, true)
		typ := types.NewNamed(name)
		r.setType(name, typ)

		// Important: We need to do this before SetUnderlying.
		r.ext.typeExt(name)

		// We need to defer CheckSize until we've called SetUnderlying to
		// handle recursive types.
		types.DeferCheckSize()
		typ.SetUnderlying(r.typ())
		types.ResumeCheckSize()

		methods := make([]*types.Field, r.len())
		for i := range methods {
			methods[i] = r.method()
		}
		if len(methods) != 0 {
			typ.Methods().Set(methods)
		}

		return name

	case objVar:
		name := do(ir.ONAME, false)
		r.setType(name, r.typ())
		r.ext.varExt(name)
		return name
	}
}

func (r *reader) mangle(sym *types.Sym) *types.Sym {
	if len(r.implicits)+len(r.explicits) == 0 {
		return sym
	}

	var buf bytes.Buffer
	buf.WriteString(sym.Name)
	buf.WriteByte('[')
	for i, targs := range [2][]*types.Type{r.implicits, r.explicits} {
		if i > 0 && len(r.implicits) != 0 && len(r.explicits) != 0 {
			buf.WriteByte(';')
		}
		for j, targ := range targs {
			if j > 0 {
				buf.WriteByte(',')
			}
			// TODO(mdempsky): We need the linker to replace "" in the symbol
			// names here.
			buf.WriteString(targ.ShortString())
		}
	}
	buf.WriteByte(']')
	return sym.Pkg.Lookup(buf.String())
}

func (r *reader) typeParamBounds(sym *types.Sym) {
	r.sync(syncTypeParamBounds)

	nimplicits := r.len()
	nexplicits := r.len()

	if len(r.implicits) != nimplicits || len(r.explicits) != nexplicits {
		base.Fatalf("%v has %v+%v params, but instantiated with %v+%v args", sym, nimplicits, nexplicits, len(r.implicits), len(r.explicits))
	}

	// For stenciling, we can just skip over the type parameters.

	for range r.explicits {
		// Skip past bounds without actually evaluating them.
		r.sync(syncType)
		r.reloc(relocType)
	}
}

func (r *reader) typeParamNames() {
	r.sync(syncTypeParamNames)

	for range r.explicits {
		r.pos()
		r.localIdent()
	}
}

func (r *reader) value() (*types.Type, constant.Value) {
	r.sync(syncValue)
	typ := r.typ()
	return typ, FixValue(typ, r.rawValue())
}

func (r *reader) method() *types.Field {
	r.sync(syncMethod)
	pos := r.pos()
	pkg, sym := r.selector()
	r.typeParamNames()
	_, recv := r.param()
	typ := r.signature(pkg, recv)

	fnsym := sym
	fnsym = ir.MethodSym(recv.Type, fnsym)
	name := ir.NewNameAt(pos, fnsym)
	r.setType(name, typ)

	name.Func = ir.NewFunc(r.pos())
	name.Func.Nname = name

	// TODO(mdempsky): Make sure we're handling //go:nointerface
	// correctly. I don't think this is exercised within the Go repo.

	r.ext.funcExt(name)

	meth := types.NewField(name.Func.Pos(), sym, typ)
	meth.Nname = name
	return meth
}

func (r *reader) qualifiedIdent() (pkg *types.Pkg, sym *types.Sym) {
	r.sync(syncSym)
	pkg = r.pkg()
	if name := r.string(); name != "" {
		sym = pkg.Lookup(name)
	}
	return
}

func (r *reader) localIdent() (pkg *types.Pkg, sym *types.Sym) {
	r.sync(syncLocalIdent)
	pkg = r.pkg()
	if name := r.string(); name != "" {
		sym = pkg.Lookup(name)
	}
	return
}

func (r *reader) selector() (origPkg *types.Pkg, sym *types.Sym) {
	r.sync(syncSelector)
	origPkg = r.pkg()
	name := r.string()
	pkg := origPkg
	if types.IsExported(name) {
		pkg = types.LocalPkg
	}
	sym = pkg.Lookup(name)
	return
}

// @@@ Compiler extensions

func (r *reader) funcExt(name *ir.Name) {
	r.sync(syncFuncExt)

	name.Class = 0 // so MarkFunc doesn't complain
	ir.MarkFunc(name)

	fn := name.Func

	// XXX: Workaround because linker doesn't know how to copy Pos.
	if !fn.Pos().IsKnown() {
		fn.SetPos(name.Pos())
	}

	// TODO(mdempsky): Remember why I wrote this code. I think it has to
	// do with how ir.VisitFuncsBottomUp works?
	if name.Sym().Pkg == types.LocalPkg || len(r.implicits)+len(r.explicits) != 0 {
		name.Defn = fn
	}

	fn.Pragma = r.pragmaFlag()
	r.linkname(name)

	if r.bool() {
		fn.ABI = obj.ABI(r.uint64())

		// Escape analysis.
		for _, fs := range &types.RecvsParams {
			for _, f := range fs(name.Type()).FieldSlice() {
				f.Note = r.string()
			}
		}

		if r.bool() {
			fn.Inl = &ir.Inline{
				Cost:            int32(r.len()),
				CanDelayResults: r.bool(),
			}
			r.addBody(name.Func)
		}
	} else {
		r.addBody(name.Func)
	}
	r.sync(syncEOF)
}

func (r *reader) typeExt(name *ir.Name) {
	r.sync(syncTypeExt)

	typ := name.Type()

	if len(r.implicits)+len(r.explicits) != 0 {
		// Set "RParams" (really type arguments here, not parameters) so
		// this type is treated as "fully instantiated". This ensures the
		// type descriptor is written out as DUPOK and method wrappers are
		// generated even for imported types.
		var targs []*types.Type
		targs = append(targs, r.implicits...)
		targs = append(targs, r.explicits...)
		typ.SetRParams(targs)
	}

	name.SetPragma(r.pragmaFlag())
	if name.Pragma()&ir.NotInHeap != 0 {
		typ.SetNotInHeap(true)
	}

	typecheck.SetBaseTypeIndex(typ, r.int64(), r.int64())
}

func (r *reader) varExt(name *ir.Name) {
	r.sync(syncVarExt)
	r.linkname(name)
}

func (r *reader) linkname(name *ir.Name) {
	assert(name.Op() == ir.ONAME)
	r.sync(syncLinkname)

	if idx := r.int64(); idx >= 0 {
		lsym := name.Linksym()
		lsym.SymIdx = int32(idx)
		lsym.Set(obj.AttrIndexed, true)
	} else {
		name.Sym().Linkname = r.string()
	}
}

func (r *reader) pragmaFlag() ir.PragmaFlag {
	r.sync(syncPragma)
	return ir.PragmaFlag(r.int())
}

// @@@ Function bodies

// bodyReader tracks where the serialized IR for a function's body can
// be found.
var bodyReader = map[*ir.Func]pkgReaderIndex{}

// todoBodies holds the list of function bodies that still need to be
// constructed.
var todoBodies []*ir.Func

func (r *reader) addBody(fn *ir.Func) {
	r.sync(syncAddBody)

	// See commont in writer.addBody for why r.implicits and r.explicits
	// should never both be non-empty.
	implicits := r.implicits
	if len(implicits) == 0 {
		implicits = r.explicits
	} else {
		assert(len(r.explicits) == 0)
	}

	pri := pkgReaderIndex{r.p, r.reloc(relocBody), implicits}
	bodyReader[fn] = pri

	if r.curfn == nil {
		todoBodies = append(todoBodies, fn)
		return
	}

	pri.funcBody(fn)
}

func (pri pkgReaderIndex) funcBody(fn *ir.Func) {
	r := pri.asReader(relocBody, syncFuncBody)
	r.funcBody(fn)
}

func (r *reader) funcBody(fn *ir.Func) {
	r.curfn = fn
	r.locals = fn.ClosureVars

	// TODO(mdempsky): Get rid of uses of typecheck.NodAddrAt so we
	// don't have to set ir.CurFunc.
	outerCurFunc := ir.CurFunc
	ir.CurFunc = fn

	r.funcargs(fn)

	if r.bool() {
		body := r.stmts()
		if body == nil {
			pos := src.NoXPos
			if quirksMode() {
				pos = funcParamsEndPos(fn)
			}
			body = []ir.Node{ir.NewBlockStmt(pos, nil)}
		}
		fn.Body = body
		fn.Endlineno = r.pos()
	}

	ir.CurFunc = outerCurFunc
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
	r.setType(name, param.Type)
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

	r.sync(syncAddLocal)
	if debug {
		want := r.int()
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
	r.sync(syncUseObjLocal)
	return r.locals[r.len()]
}

func (r *reader) openScope() {
	r.sync(syncOpenScope)
	pos := r.pos()

	if base.Flag.Dwarf {
		r.scopeVars = append(r.scopeVars, len(r.curfn.Dcl))
		r.marker.Push(pos)
	}
}

func (r *reader) closeScope() {
	r.sync(syncCloseScope)
	r.lastCloseScopePos = r.pos()

	r.closeAnotherScope()
}

// closeAnotherScope is like closeScope, but it reuses the same mark
// position as the last closeScope call. This is useful for "for" and
// "if" statements, as their implicit blocks always end at the same
// position as an explicit block.
func (r *reader) closeAnotherScope() {
	r.sync(syncCloseAnotherScope)

	if base.Flag.Dwarf {
		scopeVars := r.scopeVars[len(r.scopeVars)-1]
		r.scopeVars = r.scopeVars[:len(r.scopeVars)-1]

		if scopeVars == len(r.curfn.Dcl) {
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
	var res ir.Nodes

	r.sync(syncStmts)
	for {
		tag := codeStmt(r.code(syncStmt1))
		if tag == stmtEnd {
			r.sync(syncStmtsEnd)
			return res
		}

		if n := r.stmt1(tag, &res); n != nil {
			res.Append(n)
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
		names, lhs := r.assignList()
		rhs := r.exprList()

		if len(rhs) == 0 {
			for _, name := range names {
				as := ir.NewAssignStmt(pos, name, nil)
				as.PtrInit().Append(ir.NewDecl(pos, ir.ODCL, name))
				out.Append(as)
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

	case stmtTypeDeclHack:
		// fake "type _ = int" declaration to prevent inlining in quirks mode.
		assert(quirksMode())

		name := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, ir.BlankNode.Sym())
		name.SetAlias(true)
		r.setType(name, types.Types[types.TINT])

		n := ir.NewDecl(src.NoXPos, ir.ODCLTYPE, name)
		n.SetTypecheck(1)
		return n
	}
}

func (r *reader) assignList() ([]*ir.Name, []ir.Node) {
	lhs := make([]ir.Node, r.len())
	var names []*ir.Name

	for i := range lhs {
		if r.bool() {
			pos := r.pos()
			_, sym := r.localIdent()
			typ := r.typ()

			name := ir.NewNameAt(pos, sym)
			lhs[i] = name
			names = append(names, name)
			r.setType(name, typ)
			r.addLocal(name, ir.PAUTO)
			continue
		}

		lhs[i] = r.expr()
	}

	return names, lhs
}

func (r *reader) blockStmt() []ir.Node {
	r.sync(syncBlockStmt)
	r.openScope()
	stmts := r.stmts()
	r.closeScope()
	return stmts
}

func (r *reader) forStmt(label *types.Sym) ir.Node {
	r.sync(syncForStmt)

	r.openScope()

	if r.bool() {
		pos := r.pos()
		names, lhs := r.assignList()
		x := r.expr()
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
	r.sync(syncIfStmt)
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
	r.sync(syncSelectStmt)

	pos := r.pos()
	clauses := make([]*ir.CommClause, r.len())
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
	r.sync(syncSwitchStmt)

	r.openScope()
	pos := r.pos()
	init := r.stmt()
	tag := r.expr()

	tswitch, ok := tag.(*ir.TypeSwitchGuard)
	if ok && tswitch.Tag == nil {
		tswitch = nil
	}

	clauses := make([]*ir.CaseClause, r.len())
	for i := range clauses {
		if i > 0 {
			r.closeScope()
		}
		r.openScope()

		pos := r.pos()
		cases := r.exprList()

		clause := ir.NewCaseStmt(pos, cases, nil)
		if tswitch != nil {
			pos := r.pos()
			typ := r.typ()

			name := ir.NewNameAt(pos, tswitch.Tag.Sym())
			r.setType(name, typ)
			r.addLocal(name, ir.PAUTO)
			clause.Var = name
			name.Defn = tswitch
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
	r.sync(syncLabel)
	name := r.string()
	if r.inlCall != nil {
		name = fmt.Sprintf("~%s·%d", name, inlgen)
	}
	return typecheck.Lookup(name)
}

func (r *reader) optLabel() *types.Sym {
	r.sync(syncOptLabel)
	if r.bool() {
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

func (r *reader) expr() ir.Node {
	switch tag := codeExpr(r.code(syncExpr)); tag {
	default:
		panic("unhandled expression")

	case exprNone:
		return nil

	case exprBlank:
		return ir.BlankNode

	case exprLocal:
		return r.useLocal()

	case exprName:
		return r.obj()

	case exprType:
		return ir.TypeNode(r.typ())

	case exprConst:
		pos := r.pos()
		typ, val := r.value()
		op := r.op()
		orig := r.string()
		return OrigConst(pos, typ, val, op, orig)

	case exprCompLit:
		return r.compLit()

	case exprFuncLit:
		return r.funcLit()

	case exprSelector:
		x := r.expr()
		pos := r.pos()
		_, sym := r.selector()
		return ir.NewSelectorExpr(pos, ir.OXDOT, x, sym)

	case exprIndex:
		x := r.expr()
		pos := r.pos()
		index := r.expr()
		return ir.NewIndexExpr(pos, x, index)

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
		return ir.NewSliceExpr(pos, op, x, index[0], index[1], index[2])

	case exprAssert:
		x := r.expr()
		pos := r.pos()
		typ := r.expr().(ir.Ntype)
		return ir.NewTypeAssertExpr(pos, x, typ)

	case exprUnaryOp:
		op := r.op()
		pos := r.pos()
		x := r.expr()

		switch op {
		case ir.OADDR:
			return typecheck.NodAddrAt(pos, x)
		case ir.ODEREF:
			return ir.NewStarExpr(pos, x)
		}
		return ir.NewUnaryExpr(pos, op, x)

	case exprBinaryOp:
		op := r.op()
		x := r.expr()
		pos := r.pos()
		y := r.expr()

		switch op {
		case ir.OANDAND, ir.OOROR:
			return ir.NewLogicalExpr(pos, op, x, y)
		}
		return ir.NewBinaryExpr(pos, op, x, y)

	case exprCall:
		fun := r.expr()
		pos := r.pos()
		args := r.exprs()
		dots := r.bool()
		n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
		n.IsDDD = dots
		return n

	case exprTypeSwitchGuard:
		pos := r.pos()
		var tag *ir.Ident
		if r.bool() {
			pos := r.pos()
			sym := typecheck.Lookup(r.string())
			tag = ir.NewIdent(pos, sym)
		}
		x := r.expr()
		return ir.NewTypeSwitchGuard(pos, tag, x)
	}
}

func (r *reader) compLit() ir.Node {
	r.sync(syncCompLit)
	pos := r.pos()
	typ := r.typ()

	isPtrLit := typ.IsPtr()
	if isPtrLit {
		typ = typ.Elem()
	}
	if typ.Kind() == types.TFORW {
		base.FatalfAt(pos, "unresolved composite literal type: %v", typ)
	}
	isStruct := typ.Kind() == types.TSTRUCT

	elems := make([]ir.Node, r.len())
	for i := range elems {
		elemp := &elems[i]

		if isStruct {
			sk := ir.NewStructKeyExpr(r.pos(), typ.Field(r.len()), nil)
			*elemp, elemp = sk, &sk.Value
		} else if r.bool() {
			kv := ir.NewKeyExpr(r.pos(), r.expr(), nil)
			*elemp, elemp = kv, &kv.Value
		}

		*elemp = wrapName(r.pos(), r.expr())
	}

	lit := ir.NewCompLitExpr(pos, ir.OCOMPLIT, ir.TypeNode(typ), elems)
	if isPtrLit {
		return typecheck.NodAddrAt(pos, lit)
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
	case ir.ONAME, ir.ONONAME, ir.OPACK, ir.ONIL:
		p := ir.NewParenExpr(pos, x)
		p.SetImplicit(true)
		return p
	}
	return x
}

func (r *reader) funcLit() ir.Node {
	r.sync(syncFuncLit)

	pos := r.pos()
	typPos := r.pos()
	xtype2 := r.signature(types.LocalPkg, nil)

	opos := pos
	if quirksMode() {
		opos = r.origPos(pos)
	}

	fn := ir.NewClosureFunc(opos, r.curfn != nil)

	r.setType(fn.Nname, xtype2)
	if quirksMode() {
		fn.Nname.Ntype = ir.TypeNodeAt(typPos, xtype2)
	}

	fn.ClosureVars = make([]*ir.Name, r.len())
	for i := range fn.ClosureVars {
		pos := r.pos()
		outer := r.useLocal()

		cv := ir.NewNameAt(pos, outer.Sym())
		r.setType(cv, outer.Type())
		cv.Curfn = fn
		cv.Class = ir.PAUTOHEAP
		cv.SetIsClosureVar(true)
		cv.Defn = outer.Canonical()
		cv.Outer = outer

		fn.ClosureVars[i] = cv
	}

	r.addBody(fn)

	return fn.OClosure
}

func (r *reader) exprList() []ir.Node {
	r.sync(syncExprList)
	return r.exprs()
}

func (r *reader) exprs() []ir.Node {
	r.sync(syncExprs)
	nodes := make([]ir.Node, r.len())
	if len(nodes) == 0 {
		return nil // TODO(mdempsky): Unclear if this matters.
	}
	for i := range nodes {
		nodes[i] = r.expr()
	}
	return nodes
}

func (r *reader) op() ir.Op {
	r.sync(syncOp)
	return ir.Op(r.len())
}

// @@@ Package initialization

func (r *reader) pkgInit(self *types.Pkg, target *ir.Package) {
	if quirksMode() {
		for i, n := 0, r.len(); i < n; i++ {
			// Eagerly register position bases, so their filenames are
			// assigned stable indices.
			posBase := r.posBase()
			_ = base.Ctxt.PosTable.XPos(src.MakePos(posBase, 0, 0))
		}

		for i, n := 0, r.len(); i < n; i++ {
			// Eagerly resolve imported objects, so any filenames registered
			// in the process are assigned stable indices too.
			_, sym := r.qualifiedIdent()
			typecheck.Resolve(ir.NewIdent(src.NoXPos, sym))
			assert(sym.Def != nil)
		}
	}

	cgoPragmas := make([][]string, r.len())
	for i := range cgoPragmas {
		cgoPragmas[i] = r.strings()
	}
	target.CgoPragmas = cgoPragmas

	r.pkgDecls(target)

	r.sync(syncEOF)
}

func (r *reader) pkgDecls(target *ir.Package) {
	r.sync(syncDecls)
	for {
		switch code := codeDecl(r.code(syncDecl)); code {
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

			if n := r.len(); n > 0 {
				assert(len(names) == 1)
				embeds := make([]ir.Embed, n)
				for i := range embeds {
					embeds[i] = ir.Embed{Pos: r.pos(), Patterns: r.strings()}
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
	r.sync(syncDeclNames)
	nodes := make([]*ir.Name, r.len())
	for i := range nodes {
		r.sync(syncDeclName)

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
		// Assume it's an imported function or something that we don't
		// have access to in quirks mode.
		if haveLegacyImports {
			return nil
		}

		base.FatalfAt(call.Pos(), "missing function body for call to %v", fn)
	}

	if fn.Inl.Body == nil {
		expandInline(fn, pri)
	}

	r := pri.asReader(relocBody, syncFuncBody)

	// TODO(mdempsky): This still feels clumsy. Can we do better?
	tmpfn := ir.NewFunc(fn.Pos())
	tmpfn.Nname = ir.NewNameAt(fn.Nname.Pos(), callerfn.Sym())
	tmpfn.Closgen = callerfn.Closgen
	defer func() { callerfn.Closgen = tmpfn.Closgen }()

	r.setType(tmpfn.Nname, fn.Type())
	r.curfn = tmpfn

	r.inlCaller = ir.CurFunc
	r.inlCall = call
	r.inlFunc = fn
	r.inlTreeIndex = inlIndex
	r.inlPosBases = make(map[*src.PosBase]*src.PosBase)

	for _, cv := range r.inlFunc.ClosureVars {
		r.locals = append(r.locals, cv.Outer)
	}

	r.funcargs(fn)

	assert(r.bool()) // have body
	r.delayResults = fn.Inl.CanDelayResults

	r.retlabel = typecheck.AutoLabel(".i")
	inlgen++

	init := ir.TakeInit(call)

	// For normal function calls, the function callee expression
	// may contain side effects (e.g., added by addinit during
	// inlconv2expr or inlconv2list). Make sure to preserve these,
	// if necessary (#42703).
	if call.Op() == ir.OCALLFUNC {
		callee := call.X
		for callee.Op() == ir.OCONVNOP {
			conv := callee.(*ir.ConvExpr)
			init.Append(ir.TakeInit(conv)...)
			callee = conv.X
		}

		switch callee.Op() {
		case ir.ONAME, ir.OCLOSURE, ir.OMETHEXPR:
			// ok
		default:
			base.Fatalf("unexpected callee expression: %v", callee)
		}
	}

	var args ir.Nodes
	if call.Op() == ir.OCALLMETH {
		assert(call.X.Op() == ir.ODOTMETH)
		args.Append(call.X.(*ir.SelectorExpr).X)
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

	oldcurfn := ir.CurFunc
	ir.CurFunc = r.curfn

	r.curfn.Body = r.stmts()
	r.curfn.Endlineno = r.pos()

	typecheck.Stmts(r.curfn.Body)
	deadcode.Func(r.curfn)

	// Replace any "return" statements within the function body.
	{
		var edit func(ir.Node) ir.Node
		edit = func(n ir.Node) ir.Node {
			if ret, ok := n.(*ir.ReturnStmt); ok {
				n = typecheck.Stmt(r.inlReturn(ret))
			}
			ir.EditChildren(n, edit)
			return n
		}
		edit(r.curfn)
	}

	ir.CurFunc = oldcurfn

	body := ir.Nodes(r.curfn.Body)

	// Quirk: If deadcode elimination turned a non-empty function into
	// an empty one, we need to set the position for the empty block
	// left behind to the the inlined position for src.NoXPos, so that
	// an empty string gets added into the DWARF file name listing at
	// the appropriate index.
	if quirksMode() && len(body) == 1 {
		if block, ok := body[0].(*ir.BlockStmt); ok && len(block.List) == 0 {
			block.SetPos(r.updatePos(src.NoXPos))
		}
	}

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
	// TODO(mdempsky): Remove this function. It's currently needed for
	// dwarfgen for some reason, but we should be able to provide it
	// with the same information some other way.

	fndcls := len(fn.Dcl)
	topdcls := len(typecheck.Target.Decls)

	tmpfn := ir.NewFunc(fn.Pos())
	tmpfn.Nname = ir.NewNameAt(fn.Nname.Pos(), fn.Sym())
	tmpfn.ClosureVars = fn.ClosureVars

	{
		r := pri.asReader(relocBody, syncFuncBody)
		r.setType(tmpfn.Nname, fn.Type())

		// Don't change parameter's Sym/Nname fields.
		r.funarghack = true

		r.funcBody(tmpfn)
	}

	oldcurfn := ir.CurFunc
	ir.CurFunc = tmpfn

	typecheck.Stmts(tmpfn.Body)
	deadcode.Func(tmpfn)

	ir.CurFunc = oldcurfn

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
