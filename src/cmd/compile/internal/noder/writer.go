// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/pkgbits"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type pkgWriter struct {
	pkgbits.PkgEncoder

	m      posMap
	curpkg *types2.Package
	info   *types2.Info

	posBasesIdx map[*syntax.PosBase]int
	pkgsIdx     map[*types2.Package]int
	typsIdx     map[types2.Type]int
	globalsIdx  map[types2.Object]int

	funDecls map[*types2.Func]*syntax.FuncDecl
	typDecls map[*types2.TypeName]typeDeclGen

	linknames  map[types2.Object]string
	cgoPragmas [][]string
}

func newPkgWriter(m posMap, pkg *types2.Package, info *types2.Info) *pkgWriter {
	return &pkgWriter{
		PkgEncoder: pkgbits.NewPkgEncoder(base.Debug.SyncFrames),

		m:      m,
		curpkg: pkg,
		info:   info,

		pkgsIdx:    make(map[*types2.Package]int),
		globalsIdx: make(map[types2.Object]int),
		typsIdx:    make(map[types2.Type]int),

		posBasesIdx: make(map[*syntax.PosBase]int),

		funDecls: make(map[*types2.Func]*syntax.FuncDecl),
		typDecls: make(map[*types2.TypeName]typeDeclGen),

		linknames: make(map[types2.Object]string),
	}
}

func (pw *pkgWriter) errorf(p poser, msg string, args ...interface{}) {
	base.ErrorfAt(pw.m.pos(p), msg, args...)
}

func (pw *pkgWriter) fatalf(p poser, msg string, args ...interface{}) {
	base.FatalfAt(pw.m.pos(p), msg, args...)
}

func (pw *pkgWriter) unexpected(what string, p poser) {
	pw.fatalf(p, "unexpected %s: %v (%T)", what, p, p)
}

type writer struct {
	p *pkgWriter

	pkgbits.Encoder

	// TODO(mdempsky): We should be able to prune localsIdx whenever a
	// scope closes, and then maybe we can just use the same map for
	// storing the TypeParams too (as their TypeName instead).

	// variables declared within this function
	localsIdx map[*types2.Var]int

	closureVars    []posObj
	closureVarsIdx map[*types2.Var]int

	dict    *writerDict
	derived bool
}

// A writerDict tracks types and objects that are used by a declaration.
type writerDict struct {
	implicits []*types2.TypeName

	// derived is a slice of type indices for computing derived types
	// (i.e., types that depend on the declaration's type parameters).
	derived []derivedInfo

	// derivedIdx maps a Type to its corresponding index within the
	// derived slice, if present.
	derivedIdx map[types2.Type]int

	// funcs lists references to generic functions that were
	// instantiated with derived types (i.e., that require
	// sub-dictionaries when called at run time).
	funcs []objInfo

	// itabs lists itabs that are needed for dynamic type assertions
	// (including type switches).
	itabs []itabInfo
}

type derivedInfo struct {
	idx    int
	needed bool
}

type typeInfo struct {
	idx     int
	derived bool
}

type objInfo struct {
	idx       int        // index for the generic function declaration
	explicits []typeInfo // info for the type arguments
}

type itabInfo struct {
	typIdx int      // always a derived type index
	iface  typeInfo // always a non-empty interface type
}

func (info objInfo) anyDerived() bool {
	for _, explicit := range info.explicits {
		if explicit.derived {
			return true
		}
	}
	return false
}

func (info objInfo) equals(other objInfo) bool {
	if info.idx != other.idx {
		return false
	}
	assert(len(info.explicits) == len(other.explicits))
	for i, targ := range info.explicits {
		if targ != other.explicits[i] {
			return false
		}
	}
	return true
}

func (pw *pkgWriter) newWriter(k pkgbits.RelocKind, marker pkgbits.SyncMarker) *writer {
	return &writer{
		Encoder: pw.NewEncoder(k, marker),
		p:       pw,
	}
}

// @@@ Positions

func (w *writer) pos(p poser) {
	w.Sync(pkgbits.SyncPos)
	pos := p.Pos()

	// TODO(mdempsky): Track down the remaining cases here and fix them.
	if !w.Bool(pos.IsKnown()) {
		return
	}

	// TODO(mdempsky): Delta encoding. Also, if there's a b-side, update
	// its position base too (but not vice versa!).
	w.posBase(pos.Base())
	w.Uint(pos.Line())
	w.Uint(pos.Col())
}

func (w *writer) posBase(b *syntax.PosBase) {
	w.Reloc(pkgbits.RelocPosBase, w.p.posBaseIdx(b))
}

func (pw *pkgWriter) posBaseIdx(b *syntax.PosBase) int {
	if idx, ok := pw.posBasesIdx[b]; ok {
		return idx
	}

	w := pw.newWriter(pkgbits.RelocPosBase, pkgbits.SyncPosBase)
	w.p.posBasesIdx[b] = w.Idx

	w.String(trimFilename(b))

	if !w.Bool(b.IsFileBase()) {
		w.pos(b)
		w.Uint(b.Line())
		w.Uint(b.Col())
	}

	return w.Flush()
}

// @@@ Packages

func (w *writer) pkg(pkg *types2.Package) {
	w.Sync(pkgbits.SyncPkg)
	w.Reloc(pkgbits.RelocPkg, w.p.pkgIdx(pkg))
}

func (pw *pkgWriter) pkgIdx(pkg *types2.Package) int {
	if idx, ok := pw.pkgsIdx[pkg]; ok {
		return idx
	}

	w := pw.newWriter(pkgbits.RelocPkg, pkgbits.SyncPkgDef)
	pw.pkgsIdx[pkg] = w.Idx

	if pkg == nil {
		w.String("builtin")
	} else {
		var path string
		if pkg != w.p.curpkg {
			path = pkg.Path()
		}
		w.String(path)
		w.String(pkg.Name())
		w.Len(pkg.Height())

		w.Len(len(pkg.Imports()))
		for _, imp := range pkg.Imports() {
			w.pkg(imp)
		}
	}

	return w.Flush()
}

// @@@ Types

var anyTypeName = types2.Universe.Lookup("any").(*types2.TypeName)

func (w *writer) typ(typ types2.Type) {
	w.typInfo(w.p.typIdx(typ, w.dict))
}

func (w *writer) typInfo(info typeInfo) {
	w.Sync(pkgbits.SyncType)
	if w.Bool(info.derived) {
		w.Len(info.idx)
		w.derived = true
	} else {
		w.Reloc(pkgbits.RelocType, info.idx)
	}
}

// typIdx returns the index where the export data description of type
// can be read back in. If no such index exists yet, it's created.
//
// typIdx also reports whether typ is a derived type; that is, whether
// its identity depends on type parameters.
func (pw *pkgWriter) typIdx(typ types2.Type, dict *writerDict) typeInfo {
	if idx, ok := pw.typsIdx[typ]; ok {
		return typeInfo{idx: idx, derived: false}
	}
	if dict != nil {
		if idx, ok := dict.derivedIdx[typ]; ok {
			return typeInfo{idx: idx, derived: true}
		}
	}

	w := pw.newWriter(pkgbits.RelocType, pkgbits.SyncTypeIdx)
	w.dict = dict

	switch typ := typ.(type) {
	default:
		base.Fatalf("unexpected type: %v (%T)", typ, typ)

	case *types2.Basic:
		switch kind := typ.Kind(); {
		case kind == types2.Invalid:
			base.Fatalf("unexpected types2.Invalid")

		case types2.Typ[kind] == typ:
			w.Code(pkgbits.TypeBasic)
			w.Len(int(kind))

		default:
			// Handle "byte" and "rune" as references to their TypeName.
			obj := types2.Universe.Lookup(typ.Name())
			assert(obj.Type() == typ)

			w.Code(pkgbits.TypeNamed)
			w.obj(obj, nil)
		}

	case *types2.Named:
		assert(typ.TypeParams().Len() == typ.TypeArgs().Len())

		// TODO(mdempsky): Why do we need to loop here?
		orig := typ
		for orig.TypeArgs() != nil {
			orig = orig.Origin()
		}

		w.Code(pkgbits.TypeNamed)
		w.obj(orig.Obj(), typ.TypeArgs())

	case *types2.TypeParam:
		index := func() int {
			for idx, name := range w.dict.implicits {
				if name.Type().(*types2.TypeParam) == typ {
					return idx
				}
			}

			return len(w.dict.implicits) + typ.Index()
		}()

		w.derived = true
		w.Code(pkgbits.TypeTypeParam)
		w.Len(index)

	case *types2.Array:
		w.Code(pkgbits.TypeArray)
		w.Uint64(uint64(typ.Len()))
		w.typ(typ.Elem())

	case *types2.Chan:
		w.Code(pkgbits.TypeChan)
		w.Len(int(typ.Dir()))
		w.typ(typ.Elem())

	case *types2.Map:
		w.Code(pkgbits.TypeMap)
		w.typ(typ.Key())
		w.typ(typ.Elem())

	case *types2.Pointer:
		w.Code(pkgbits.TypePointer)
		w.typ(typ.Elem())

	case *types2.Signature:
		base.Assertf(typ.TypeParams() == nil, "unexpected type params: %v", typ)
		w.Code(pkgbits.TypeSignature)
		w.signature(typ)

	case *types2.Slice:
		w.Code(pkgbits.TypeSlice)
		w.typ(typ.Elem())

	case *types2.Struct:
		w.Code(pkgbits.TypeStruct)
		w.structType(typ)

	case *types2.Interface:
		if typ == anyTypeName.Type() {
			w.Code(pkgbits.TypeNamed)
			w.obj(anyTypeName, nil)
			break
		}

		w.Code(pkgbits.TypeInterface)
		w.interfaceType(typ)

	case *types2.Union:
		w.Code(pkgbits.TypeUnion)
		w.unionType(typ)
	}

	if w.derived {
		idx := len(dict.derived)
		dict.derived = append(dict.derived, derivedInfo{idx: w.Flush()})
		dict.derivedIdx[typ] = idx
		return typeInfo{idx: idx, derived: true}
	}

	pw.typsIdx[typ] = w.Idx
	return typeInfo{idx: w.Flush(), derived: false}
}

func (w *writer) structType(typ *types2.Struct) {
	w.Len(typ.NumFields())
	for i := 0; i < typ.NumFields(); i++ {
		f := typ.Field(i)
		w.pos(f)
		w.selector(f)
		w.typ(f.Type())
		w.String(typ.Tag(i))
		w.Bool(f.Embedded())
	}
}

func (w *writer) unionType(typ *types2.Union) {
	w.Len(typ.Len())
	for i := 0; i < typ.Len(); i++ {
		t := typ.Term(i)
		w.Bool(t.Tilde())
		w.typ(t.Type())
	}
}

func (w *writer) interfaceType(typ *types2.Interface) {
	w.Len(typ.NumExplicitMethods())
	w.Len(typ.NumEmbeddeds())

	if typ.NumExplicitMethods() == 0 && typ.NumEmbeddeds() == 1 {
		w.Bool(typ.IsImplicit())
	} else {
		// Implicit interfaces always have 0 explicit methods and 1
		// embedded type, so we skip writing out the implicit flag
		// otherwise as a space optimization.
		assert(!typ.IsImplicit())
	}

	for i := 0; i < typ.NumExplicitMethods(); i++ {
		m := typ.ExplicitMethod(i)
		sig := m.Type().(*types2.Signature)
		assert(sig.TypeParams() == nil)

		w.pos(m)
		w.selector(m)
		w.signature(sig)
	}

	for i := 0; i < typ.NumEmbeddeds(); i++ {
		w.typ(typ.EmbeddedType(i))
	}
}

func (w *writer) signature(sig *types2.Signature) {
	w.Sync(pkgbits.SyncSignature)
	w.params(sig.Params())
	w.params(sig.Results())
	w.Bool(sig.Variadic())
}

func (w *writer) params(typ *types2.Tuple) {
	w.Sync(pkgbits.SyncParams)
	w.Len(typ.Len())
	for i := 0; i < typ.Len(); i++ {
		w.param(typ.At(i))
	}
}

func (w *writer) param(param *types2.Var) {
	w.Sync(pkgbits.SyncParam)
	w.pos(param)
	w.localIdent(param)
	w.typ(param.Type())
}

// @@@ Objects

func (w *writer) obj(obj types2.Object, explicits *types2.TypeList) {
	explicitInfos := make([]typeInfo, explicits.Len())
	for i := range explicitInfos {
		explicitInfos[i] = w.p.typIdx(explicits.At(i), w.dict)
	}
	info := objInfo{idx: w.p.objIdx(obj), explicits: explicitInfos}

	if _, ok := obj.(*types2.Func); ok && info.anyDerived() {
		idx := -1
		for i, prev := range w.dict.funcs {
			if prev.equals(info) {
				idx = i
			}
		}
		if idx < 0 {
			idx = len(w.dict.funcs)
			w.dict.funcs = append(w.dict.funcs, info)
		}

		// TODO(mdempsky): Push up into expr; this shouldn't appear
		// outside of expression context.
		w.Sync(pkgbits.SyncObject)
		w.Bool(true)
		w.Len(idx)
		return
	}

	// TODO(mdempsky): Push up into typIdx; this shouldn't be needed
	// except while writing out types.
	if isDefinedType(obj) && obj.Pkg() == w.p.curpkg {
		decl, ok := w.p.typDecls[obj.(*types2.TypeName)]
		assert(ok)
		if len(decl.implicits) != 0 {
			w.derived = true
		}
	}

	w.Sync(pkgbits.SyncObject)
	w.Bool(false)
	w.Reloc(pkgbits.RelocObj, info.idx)

	w.Len(len(info.explicits))
	for _, info := range info.explicits {
		w.typInfo(info)
	}
}

func (pw *pkgWriter) objIdx(obj types2.Object) int {
	if idx, ok := pw.globalsIdx[obj]; ok {
		return idx
	}

	dict := &writerDict{
		derivedIdx: make(map[types2.Type]int),
	}

	if isDefinedType(obj) && obj.Pkg() == pw.curpkg {
		decl, ok := pw.typDecls[obj.(*types2.TypeName)]
		assert(ok)
		dict.implicits = decl.implicits
	}

	w := pw.newWriter(pkgbits.RelocObj, pkgbits.SyncObject1)
	wext := pw.newWriter(pkgbits.RelocObjExt, pkgbits.SyncObject1)
	wname := pw.newWriter(pkgbits.RelocName, pkgbits.SyncObject1)
	wdict := pw.newWriter(pkgbits.RelocObjDict, pkgbits.SyncObject1)

	pw.globalsIdx[obj] = w.Idx // break cycles
	assert(wext.Idx == w.Idx)
	assert(wname.Idx == w.Idx)
	assert(wdict.Idx == w.Idx)

	w.dict = dict
	wext.dict = dict

	code := w.doObj(wext, obj)
	w.Flush()
	wext.Flush()

	wname.qualifiedIdent(obj)
	wname.Code(code)
	wname.Flush()

	wdict.objDict(obj, w.dict)
	wdict.Flush()

	return w.Idx
}

func (w *writer) doObj(wext *writer, obj types2.Object) pkgbits.CodeObj {
	if obj.Pkg() != w.p.curpkg {
		return pkgbits.ObjStub
	}

	switch obj := obj.(type) {
	default:
		w.p.unexpected("object", obj)
		panic("unreachable")

	case *types2.Const:
		w.pos(obj)
		w.typ(obj.Type())
		w.Value(obj.Val())
		return pkgbits.ObjConst

	case *types2.Func:
		decl, ok := w.p.funDecls[obj]
		assert(ok)
		sig := obj.Type().(*types2.Signature)

		w.pos(obj)
		w.typeParamNames(sig.TypeParams())
		w.signature(sig)
		w.pos(decl)
		wext.funcExt(obj)
		return pkgbits.ObjFunc

	case *types2.TypeName:
		decl, ok := w.p.typDecls[obj]
		assert(ok)

		if obj.IsAlias() {
			w.pos(obj)
			w.typ(obj.Type())
			return pkgbits.ObjAlias
		}

		named := obj.Type().(*types2.Named)
		assert(named.TypeArgs() == nil)

		w.pos(obj)
		w.typeParamNames(named.TypeParams())
		wext.typeExt(obj)
		w.typExpr(decl.Type)

		w.Len(named.NumMethods())
		for i := 0; i < named.NumMethods(); i++ {
			w.method(wext, named.Method(i))
		}

		return pkgbits.ObjType

	case *types2.Var:
		w.pos(obj)
		w.typ(obj.Type())
		wext.varExt(obj)
		return pkgbits.ObjVar
	}
}

// typExpr writes the type represented by the given expression.
func (w *writer) typExpr(expr syntax.Expr) {
	tv, ok := w.p.info.Types[expr]
	assert(ok)
	assert(tv.IsType())
	w.typ(tv.Type)
}

// objDict writes the dictionary needed for reading the given object.
func (w *writer) objDict(obj types2.Object, dict *writerDict) {
	// TODO(mdempsky): Split objDict into multiple entries? reader.go
	// doesn't care about the type parameter bounds, and reader2.go
	// doesn't care about referenced functions.

	w.dict = dict // TODO(mdempsky): This is a bit sketchy.

	w.Len(len(dict.implicits))

	tparams := objTypeParams(obj)
	ntparams := tparams.Len()
	w.Len(ntparams)
	for i := 0; i < ntparams; i++ {
		w.typ(tparams.At(i).Constraint())
	}

	nderived := len(dict.derived)
	w.Len(nderived)
	for _, typ := range dict.derived {
		w.Reloc(pkgbits.RelocType, typ.idx)
		w.Bool(typ.needed)
	}

	nfuncs := len(dict.funcs)
	w.Len(nfuncs)
	for _, fn := range dict.funcs {
		w.Reloc(pkgbits.RelocObj, fn.idx)
		w.Len(len(fn.explicits))
		for _, targ := range fn.explicits {
			w.typInfo(targ)
		}
	}

	nitabs := len(dict.itabs)
	w.Len(nitabs)
	for _, itab := range dict.itabs {
		w.Len(itab.typIdx)
		w.typInfo(itab.iface)
	}

	assert(len(dict.derived) == nderived)
	assert(len(dict.funcs) == nfuncs)
}

func (w *writer) typeParamNames(tparams *types2.TypeParamList) {
	w.Sync(pkgbits.SyncTypeParamNames)

	ntparams := tparams.Len()
	for i := 0; i < ntparams; i++ {
		tparam := tparams.At(i).Obj()
		w.pos(tparam)
		w.localIdent(tparam)
	}
}

func (w *writer) method(wext *writer, meth *types2.Func) {
	decl, ok := w.p.funDecls[meth]
	assert(ok)
	sig := meth.Type().(*types2.Signature)

	w.Sync(pkgbits.SyncMethod)
	w.pos(meth)
	w.selector(meth)
	w.typeParamNames(sig.RecvTypeParams())
	w.param(sig.Recv())
	w.signature(sig)

	w.pos(decl) // XXX: Hack to workaround linker limitations.
	wext.funcExt(meth)
}

// qualifiedIdent writes out the name of an object declared at package
// scope. (For now, it's also used to refer to local defined types.)
func (w *writer) qualifiedIdent(obj types2.Object) {
	w.Sync(pkgbits.SyncSym)

	name := obj.Name()
	if isDefinedType(obj) && obj.Pkg() == w.p.curpkg {
		decl, ok := w.p.typDecls[obj.(*types2.TypeName)]
		assert(ok)
		if decl.gen != 0 {
			// TODO(mdempsky): Find a better solution than embedding middle
			// dot in the symbol name; this is terrible.
			name = fmt.Sprintf("%s·%v", name, decl.gen)
		}
	}

	w.pkg(obj.Pkg())
	w.String(name)
}

// TODO(mdempsky): We should be able to omit pkg from both localIdent
// and selector, because they should always be known from context.
// However, past frustrations with this optimization in iexport make
// me a little nervous to try it again.

// localIdent writes the name of a locally declared object (i.e.,
// objects that can only be accessed by name, within the context of a
// particular function).
func (w *writer) localIdent(obj types2.Object) {
	assert(!isGlobal(obj))
	w.Sync(pkgbits.SyncLocalIdent)
	w.pkg(obj.Pkg())
	w.String(obj.Name())
}

// selector writes the name of a field or method (i.e., objects that
// can only be accessed using selector expressions).
func (w *writer) selector(obj types2.Object) {
	w.Sync(pkgbits.SyncSelector)
	w.pkg(obj.Pkg())
	w.String(obj.Name())
}

// @@@ Compiler extensions

func (w *writer) funcExt(obj *types2.Func) {
	decl, ok := w.p.funDecls[obj]
	assert(ok)

	// TODO(mdempsky): Extend these pragma validation flags to account
	// for generics. E.g., linkname probably doesn't make sense at
	// least.

	pragma := asPragmaFlag(decl.Pragma)
	if pragma&ir.Systemstack != 0 && pragma&ir.Nosplit != 0 {
		w.p.errorf(decl, "go:nosplit and go:systemstack cannot be combined")
	}

	if decl.Body != nil {
		if pragma&ir.Noescape != 0 {
			w.p.errorf(decl, "can only use //go:noescape with external func implementations")
		}
	} else {
		if base.Flag.Complete || decl.Name.Value == "init" {
			// Linknamed functions are allowed to have no body. Hopefully
			// the linkname target has a body. See issue 23311.
			if _, ok := w.p.linknames[obj]; !ok {
				w.p.errorf(decl, "missing function body")
			}
		}
	}

	sig, block := obj.Type().(*types2.Signature), decl.Body
	body, closureVars := w.p.bodyIdx(w.p.curpkg, sig, block, w.dict)
	assert(len(closureVars) == 0)

	w.Sync(pkgbits.SyncFuncExt)
	w.pragmaFlag(pragma)
	w.linkname(obj)
	w.Bool(false) // stub extension
	w.Reloc(pkgbits.RelocBody, body)
	w.Sync(pkgbits.SyncEOF)
}

func (w *writer) typeExt(obj *types2.TypeName) {
	decl, ok := w.p.typDecls[obj]
	assert(ok)

	w.Sync(pkgbits.SyncTypeExt)

	w.pragmaFlag(asPragmaFlag(decl.Pragma))

	// No LSym.SymIdx info yet.
	w.Int64(-1)
	w.Int64(-1)
}

func (w *writer) varExt(obj *types2.Var) {
	w.Sync(pkgbits.SyncVarExt)
	w.linkname(obj)
}

func (w *writer) linkname(obj types2.Object) {
	w.Sync(pkgbits.SyncLinkname)
	w.Int64(-1)
	w.String(w.p.linknames[obj])
}

func (w *writer) pragmaFlag(p ir.PragmaFlag) {
	w.Sync(pkgbits.SyncPragma)
	w.Int(int(p))
}

// @@@ Function bodies

func (pw *pkgWriter) bodyIdx(pkg *types2.Package, sig *types2.Signature, block *syntax.BlockStmt, dict *writerDict) (idx int, closureVars []posObj) {
	w := pw.newWriter(pkgbits.RelocBody, pkgbits.SyncFuncBody)
	w.dict = dict

	w.funcargs(sig)
	if w.Bool(block != nil) {
		w.stmts(block.List)
		w.pos(block.Rbrace)
	}

	return w.Flush(), w.closureVars
}

func (w *writer) funcargs(sig *types2.Signature) {
	do := func(params *types2.Tuple, result bool) {
		for i := 0; i < params.Len(); i++ {
			w.funcarg(params.At(i), result)
		}
	}

	if recv := sig.Recv(); recv != nil {
		w.funcarg(recv, false)
	}
	do(sig.Params(), false)
	do(sig.Results(), true)
}

func (w *writer) funcarg(param *types2.Var, result bool) {
	if param.Name() != "" || result {
		w.addLocal(param)
	}
}

func (w *writer) addLocal(obj *types2.Var) {
	w.Sync(pkgbits.SyncAddLocal)
	idx := len(w.localsIdx)
	if pkgbits.EnableSync {
		w.Int(idx)
	}
	if w.localsIdx == nil {
		w.localsIdx = make(map[*types2.Var]int)
	}
	w.localsIdx[obj] = idx
}

func (w *writer) useLocal(pos syntax.Pos, obj *types2.Var) {
	w.Sync(pkgbits.SyncUseObjLocal)

	if idx, ok := w.localsIdx[obj]; w.Bool(ok) {
		w.Len(idx)
		return
	}

	idx, ok := w.closureVarsIdx[obj]
	if !ok {
		if w.closureVarsIdx == nil {
			w.closureVarsIdx = make(map[*types2.Var]int)
		}
		idx = len(w.closureVars)
		w.closureVars = append(w.closureVars, posObj{pos, obj})
		w.closureVarsIdx[obj] = idx
	}
	w.Len(idx)
}

func (w *writer) openScope(pos syntax.Pos) {
	w.Sync(pkgbits.SyncOpenScope)
	w.pos(pos)
}

func (w *writer) closeScope(pos syntax.Pos) {
	w.Sync(pkgbits.SyncCloseScope)
	w.pos(pos)
	w.closeAnotherScope()
}

func (w *writer) closeAnotherScope() {
	w.Sync(pkgbits.SyncCloseAnotherScope)
}

// @@@ Statements

func (w *writer) stmt(stmt syntax.Stmt) {
	var stmts []syntax.Stmt
	if stmt != nil {
		stmts = []syntax.Stmt{stmt}
	}
	w.stmts(stmts)
}

func (w *writer) stmts(stmts []syntax.Stmt) {
	w.Sync(pkgbits.SyncStmts)
	for _, stmt := range stmts {
		w.stmt1(stmt)
	}
	w.Code(stmtEnd)
	w.Sync(pkgbits.SyncStmtsEnd)
}

func (w *writer) stmt1(stmt syntax.Stmt) {
	switch stmt := stmt.(type) {
	default:
		w.p.unexpected("statement", stmt)

	case nil, *syntax.EmptyStmt:
		return

	case *syntax.AssignStmt:
		switch {
		case stmt.Rhs == nil:
			w.Code(stmtIncDec)
			w.op(binOps[stmt.Op])
			w.expr(stmt.Lhs)
			w.pos(stmt)

		case stmt.Op != 0 && stmt.Op != syntax.Def:
			w.Code(stmtAssignOp)
			w.op(binOps[stmt.Op])
			w.expr(stmt.Lhs)
			w.pos(stmt)
			w.expr(stmt.Rhs)

		default:
			w.Code(stmtAssign)
			w.pos(stmt)
			w.exprList(stmt.Rhs)
			w.assignList(stmt.Lhs)
		}

	case *syntax.BlockStmt:
		w.Code(stmtBlock)
		w.blockStmt(stmt)

	case *syntax.BranchStmt:
		w.Code(stmtBranch)
		w.pos(stmt)
		w.op(branchOps[stmt.Tok])
		w.optLabel(stmt.Label)

	case *syntax.CallStmt:
		w.Code(stmtCall)
		w.pos(stmt)
		w.op(callOps[stmt.Tok])
		w.expr(stmt.Call)

	case *syntax.DeclStmt:
		for _, decl := range stmt.DeclList {
			w.declStmt(decl)
		}

	case *syntax.ExprStmt:
		w.Code(stmtExpr)
		w.expr(stmt.X)

	case *syntax.ForStmt:
		w.Code(stmtFor)
		w.forStmt(stmt)

	case *syntax.IfStmt:
		w.Code(stmtIf)
		w.ifStmt(stmt)

	case *syntax.LabeledStmt:
		w.Code(stmtLabel)
		w.pos(stmt)
		w.label(stmt.Label)
		w.stmt1(stmt.Stmt)

	case *syntax.ReturnStmt:
		w.Code(stmtReturn)
		w.pos(stmt)
		w.exprList(stmt.Results)

	case *syntax.SelectStmt:
		w.Code(stmtSelect)
		w.selectStmt(stmt)

	case *syntax.SendStmt:
		w.Code(stmtSend)
		w.pos(stmt)
		w.expr(stmt.Chan)
		w.expr(stmt.Value)

	case *syntax.SwitchStmt:
		w.Code(stmtSwitch)
		w.switchStmt(stmt)
	}
}

func (w *writer) assignList(expr syntax.Expr) {
	exprs := unpackListExpr(expr)
	w.Len(len(exprs))

	for _, expr := range exprs {
		if name, ok := expr.(*syntax.Name); ok && name.Value != "_" {
			if obj, ok := w.p.info.Defs[name]; ok {
				obj := obj.(*types2.Var)

				w.Bool(true)
				w.pos(obj)
				w.localIdent(obj)
				w.typ(obj.Type())

				// TODO(mdempsky): Minimize locals index size by deferring
				// this until the variables actually come into scope.
				w.addLocal(obj)
				continue
			}
		}

		w.Bool(false)
		w.expr(expr)
	}
}

func (w *writer) declStmt(decl syntax.Decl) {
	switch decl := decl.(type) {
	default:
		w.p.unexpected("declaration", decl)

	case *syntax.ConstDecl, *syntax.TypeDecl:

	case *syntax.VarDecl:
		w.Code(stmtAssign)
		w.pos(decl)
		w.exprList(decl.Values)
		w.assignList(namesAsExpr(decl.NameList))
	}
}

func (w *writer) blockStmt(stmt *syntax.BlockStmt) {
	w.Sync(pkgbits.SyncBlockStmt)
	w.openScope(stmt.Pos())
	w.stmts(stmt.List)
	w.closeScope(stmt.Rbrace)
}

func (w *writer) forStmt(stmt *syntax.ForStmt) {
	w.Sync(pkgbits.SyncForStmt)
	w.openScope(stmt.Pos())

	if rang, ok := stmt.Init.(*syntax.RangeClause); w.Bool(ok) {
		w.pos(rang)
		w.expr(rang.X)
		w.assignList(rang.Lhs)
	} else {
		w.pos(stmt)
		w.stmt(stmt.Init)
		w.expr(stmt.Cond)
		w.stmt(stmt.Post)
	}

	w.blockStmt(stmt.Body)
	w.closeAnotherScope()
}

func (w *writer) ifStmt(stmt *syntax.IfStmt) {
	w.Sync(pkgbits.SyncIfStmt)
	w.openScope(stmt.Pos())
	w.pos(stmt)
	w.stmt(stmt.Init)
	w.expr(stmt.Cond)
	w.blockStmt(stmt.Then)
	w.stmt(stmt.Else)
	w.closeAnotherScope()
}

func (w *writer) selectStmt(stmt *syntax.SelectStmt) {
	w.Sync(pkgbits.SyncSelectStmt)

	w.pos(stmt)
	w.Len(len(stmt.Body))
	for i, clause := range stmt.Body {
		if i > 0 {
			w.closeScope(clause.Pos())
		}
		w.openScope(clause.Pos())

		w.pos(clause)
		w.stmt(clause.Comm)
		w.stmts(clause.Body)
	}
	if len(stmt.Body) > 0 {
		w.closeScope(stmt.Rbrace)
	}
}

func (w *writer) switchStmt(stmt *syntax.SwitchStmt) {
	w.Sync(pkgbits.SyncSwitchStmt)

	w.openScope(stmt.Pos())
	w.pos(stmt)
	w.stmt(stmt.Init)

	var iface types2.Type
	if guard, ok := stmt.Tag.(*syntax.TypeSwitchGuard); w.Bool(ok) {
		tv, ok := w.p.info.Types[guard.X]
		assert(ok && tv.IsValue())
		iface = tv.Type

		w.pos(guard)
		if tag := guard.Lhs; w.Bool(tag != nil) {
			w.pos(tag)
			w.String(tag.Value)
		}
		w.expr(guard.X)
	} else {
		w.expr(stmt.Tag)
	}

	w.Len(len(stmt.Body))
	for i, clause := range stmt.Body {
		if i > 0 {
			w.closeScope(clause.Pos())
		}
		w.openScope(clause.Pos())

		w.pos(clause)

		if iface != nil {
			cases := unpackListExpr(clause.Cases)
			w.Len(len(cases))
			for _, cas := range cases {
				w.exprType(iface, cas, true)
			}
		} else {
			w.exprList(clause.Cases)
		}

		if obj, ok := w.p.info.Implicits[clause]; ok {
			// TODO(mdempsky): These pos details are quirkish, but also
			// necessary so the variable's position is correct for DWARF
			// scope assignment later. It would probably be better for us to
			// instead just set the variable's DWARF scoping info earlier so
			// we can give it the correct position information.
			pos := clause.Pos()
			if typs := unpackListExpr(clause.Cases); len(typs) != 0 {
				pos = typeExprEndPos(typs[len(typs)-1])
			}
			w.pos(pos)

			obj := obj.(*types2.Var)
			w.typ(obj.Type())
			w.addLocal(obj)
		}

		w.stmts(clause.Body)
	}
	if len(stmt.Body) > 0 {
		w.closeScope(stmt.Rbrace)
	}

	w.closeScope(stmt.Rbrace)
}

func (w *writer) label(label *syntax.Name) {
	w.Sync(pkgbits.SyncLabel)

	// TODO(mdempsky): Replace label strings with dense indices.
	w.String(label.Value)
}

func (w *writer) optLabel(label *syntax.Name) {
	w.Sync(pkgbits.SyncOptLabel)
	if w.Bool(label != nil) {
		w.label(label)
	}
}

// @@@ Expressions

func (w *writer) expr(expr syntax.Expr) {
	expr = unparen(expr) // skip parens; unneeded after typecheck

	obj, inst := lookupObj(w.p.info, expr)
	targs := inst.TypeArgs

	if tv, ok := w.p.info.Types[expr]; ok {
		// TODO(mdempsky): Be more judicious about which types are marked as "needed".
		if inst.Type != nil {
			w.needType(inst.Type)
		} else {
			w.needType(tv.Type)
		}

		if tv.IsType() {
			w.Code(exprType)
			w.exprType(nil, expr, false)
			return
		}

		if tv.Value != nil {
			w.Code(exprConst)
			w.pos(expr)
			w.typ(tv.Type)
			w.Value(tv.Value)

			// TODO(mdempsky): These details are only important for backend
			// diagnostics. Explore writing them out separately.
			w.op(constExprOp(expr))
			w.String(syntax.String(expr))
			return
		}
	}

	if obj != nil {
		if isGlobal(obj) {
			w.Code(exprName)
			w.obj(obj, targs)
			return
		}

		obj := obj.(*types2.Var)
		assert(!obj.IsField())
		assert(targs.Len() == 0)

		w.Code(exprLocal)
		w.useLocal(expr.Pos(), obj)
		return
	}

	switch expr := expr.(type) {
	default:
		w.p.unexpected("expression", expr)

	case nil: // absent slice index, for condition, or switch tag
		w.Code(exprNone)

	case *syntax.Name:
		assert(expr.Value == "_")
		w.Code(exprBlank)

	case *syntax.CompositeLit:
		w.Code(exprCompLit)
		w.compLit(expr)

	case *syntax.FuncLit:
		w.Code(exprFuncLit)
		w.funcLit(expr)

	case *syntax.SelectorExpr:
		sel, ok := w.p.info.Selections[expr]
		assert(ok)

		w.Code(exprSelector)
		w.expr(expr.X)
		w.pos(expr)
		w.selector(sel.Obj())

	case *syntax.IndexExpr:
		tv, ok := w.p.info.Types[expr.Index]
		assert(ok && tv.IsValue())

		w.Code(exprIndex)
		w.expr(expr.X)
		w.pos(expr)
		w.expr(expr.Index)

	case *syntax.SliceExpr:
		w.Code(exprSlice)
		w.expr(expr.X)
		w.pos(expr)
		for _, n := range &expr.Index {
			w.expr(n)
		}

	case *syntax.AssertExpr:
		tv, ok := w.p.info.Types[expr.X]
		assert(ok && tv.IsValue())

		w.Code(exprAssert)
		w.expr(expr.X)
		w.pos(expr)
		w.exprType(tv.Type, expr.Type, false)

	case *syntax.Operation:
		if expr.Y == nil {
			w.Code(exprUnaryOp)
			w.op(unOps[expr.Op])
			w.pos(expr)
			w.expr(expr.X)
			break
		}

		w.Code(exprBinaryOp)
		w.op(binOps[expr.Op])
		w.expr(expr.X)
		w.pos(expr)
		w.expr(expr.Y)

	case *syntax.CallExpr:
		tv, ok := w.p.info.Types[expr.Fun]
		assert(ok)
		if tv.IsType() {
			assert(len(expr.ArgList) == 1)
			assert(!expr.HasDots)

			w.Code(exprConvert)
			w.typ(tv.Type)
			w.pos(expr)
			w.expr(expr.ArgList[0])
			break
		}

		writeFunExpr := func() {
			if selector, ok := unparen(expr.Fun).(*syntax.SelectorExpr); ok {
				if sel, ok := w.p.info.Selections[selector]; ok && sel.Kind() == types2.MethodVal {
					w.expr(selector.X)
					w.Bool(true) // method call
					w.pos(selector)
					w.selector(sel.Obj())
					return
				}
			}

			w.expr(expr.Fun)
			w.Bool(false) // not a method call (i.e., normal function call)
		}

		w.Code(exprCall)
		writeFunExpr()
		w.pos(expr)
		w.exprs(expr.ArgList)
		w.Bool(expr.HasDots)
	}
}

func (w *writer) compLit(lit *syntax.CompositeLit) {
	tv, ok := w.p.info.Types[lit]
	assert(ok)

	w.Sync(pkgbits.SyncCompLit)
	w.pos(lit)
	w.typ(tv.Type)

	typ := tv.Type
	if ptr, ok := types2.CoreType(typ).(*types2.Pointer); ok {
		typ = ptr.Elem()
	}
	str, isStruct := types2.CoreType(typ).(*types2.Struct)

	w.Len(len(lit.ElemList))
	for i, elem := range lit.ElemList {
		if isStruct {
			if kv, ok := elem.(*syntax.KeyValueExpr); ok {
				// use position of expr.Key rather than of elem (which has position of ':')
				w.pos(kv.Key)
				w.Len(fieldIndex(w.p.info, str, kv.Key.(*syntax.Name)))
				elem = kv.Value
			} else {
				w.pos(elem)
				w.Len(i)
			}
		} else {
			if kv, ok := elem.(*syntax.KeyValueExpr); w.Bool(ok) {
				// use position of expr.Key rather than of elem (which has position of ':')
				w.pos(kv.Key)
				w.expr(kv.Key)
				elem = kv.Value
			}
		}
		w.pos(elem)
		w.expr(elem)
	}
}

func (w *writer) funcLit(expr *syntax.FuncLit) {
	tv, ok := w.p.info.Types[expr]
	assert(ok)
	sig := tv.Type.(*types2.Signature)

	body, closureVars := w.p.bodyIdx(w.p.curpkg, sig, expr.Body, w.dict)

	w.Sync(pkgbits.SyncFuncLit)
	w.pos(expr)
	w.signature(sig)

	w.Len(len(closureVars))
	for _, cv := range closureVars {
		w.pos(cv.pos)
		w.useLocal(cv.pos, cv.obj)
	}

	w.Reloc(pkgbits.RelocBody, body)
}

type posObj struct {
	pos syntax.Pos
	obj *types2.Var
}

func (w *writer) exprList(expr syntax.Expr) {
	w.Sync(pkgbits.SyncExprList)
	w.exprs(unpackListExpr(expr))
}

func (w *writer) exprs(exprs []syntax.Expr) {
	if len(exprs) == 0 {
		assert(exprs == nil)
	}

	w.Sync(pkgbits.SyncExprs)
	w.Len(len(exprs))
	for _, expr := range exprs {
		w.expr(expr)
	}
}

func (w *writer) exprType(iface types2.Type, typ syntax.Expr, nilOK bool) {
	base.Assertf(iface == nil || isInterface(iface), "%v must be nil or an interface type", iface)

	tv, ok := w.p.info.Types[typ]
	assert(ok)

	w.Sync(pkgbits.SyncExprType)

	if nilOK && w.Bool(tv.IsNil()) {
		return
	}

	assert(tv.IsType())
	info := w.p.typIdx(tv.Type, w.dict)

	w.pos(typ)

	if w.Bool(info.derived && iface != nil && !iface.Underlying().(*types2.Interface).Empty()) {
		ifaceInfo := w.p.typIdx(iface, w.dict)

		idx := -1
		for i, itab := range w.dict.itabs {
			if itab.typIdx == info.idx && itab.iface == ifaceInfo {
				idx = i
			}
		}
		if idx < 0 {
			idx = len(w.dict.itabs)
			w.dict.itabs = append(w.dict.itabs, itabInfo{typIdx: info.idx, iface: ifaceInfo})
		}
		w.Len(idx)
		return
	}

	w.typInfo(info)
}

func isInterface(typ types2.Type) bool {
	if _, ok := typ.(*types2.TypeParam); ok {
		// typ is a type parameter and may be instantiated as either a
		// concrete or interface type, so the writer can't depend on
		// knowing this.
		base.Fatalf("%v is a type parameter", typ)
	}

	_, ok := typ.Underlying().(*types2.Interface)
	return ok
}

func (w *writer) op(op ir.Op) {
	// TODO(mdempsky): Remove in favor of explicit codes? Would make
	// export data more stable against internal refactorings, but low
	// priority at the moment.
	assert(op != 0)
	w.Sync(pkgbits.SyncOp)
	w.Len(int(op))
}

func (w *writer) needType(typ types2.Type) {
	// Decompose tuple into component element types.
	if typ, ok := typ.(*types2.Tuple); ok {
		for i := 0; i < typ.Len(); i++ {
			w.needType(typ.At(i).Type())
		}
		return
	}

	if info := w.p.typIdx(typ, w.dict); info.derived {
		w.dict.derived[info.idx].needed = true
	}
}

// @@@ Package initialization

// Caution: This code is still clumsy, because toolstash -cmp is
// particularly sensitive to it.

type typeDeclGen struct {
	*syntax.TypeDecl
	gen int

	// Implicit type parameters in scope at this type declaration.
	implicits []*types2.TypeName
}

type fileImports struct {
	importedEmbed, importedUnsafe bool
}

type declCollector struct {
	pw         *pkgWriter
	typegen    *int
	file       *fileImports
	withinFunc bool
	implicits  []*types2.TypeName
}

func (c *declCollector) withTParams(obj types2.Object) *declCollector {
	tparams := objTypeParams(obj)
	n := tparams.Len()
	if n == 0 {
		return c
	}

	copy := *c
	copy.implicits = copy.implicits[:len(copy.implicits):len(copy.implicits)]
	for i := 0; i < n; i++ {
		copy.implicits = append(copy.implicits, tparams.At(i).Obj())
	}
	return &copy
}

func (c *declCollector) Visit(n syntax.Node) syntax.Visitor {
	pw := c.pw

	switch n := n.(type) {
	case *syntax.File:
		pw.checkPragmas(n.Pragma, ir.GoBuildPragma, false)

	case *syntax.ImportDecl:
		pw.checkPragmas(n.Pragma, 0, false)

		switch pkgNameOf(pw.info, n).Imported().Path() {
		case "embed":
			c.file.importedEmbed = true
		case "unsafe":
			c.file.importedUnsafe = true
		}

	case *syntax.ConstDecl:
		pw.checkPragmas(n.Pragma, 0, false)

	case *syntax.FuncDecl:
		pw.checkPragmas(n.Pragma, funcPragmas, false)

		obj := pw.info.Defs[n.Name].(*types2.Func)
		pw.funDecls[obj] = n

		return c.withTParams(obj)

	case *syntax.TypeDecl:
		obj := pw.info.Defs[n.Name].(*types2.TypeName)
		d := typeDeclGen{TypeDecl: n, implicits: c.implicits}

		if n.Alias {
			pw.checkPragmas(n.Pragma, 0, false)
		} else {
			pw.checkPragmas(n.Pragma, typePragmas, false)

			// Assign a unique ID to function-scoped defined types.
			if c.withinFunc {
				*c.typegen++
				d.gen = *c.typegen
			}
		}

		pw.typDecls[obj] = d

		// TODO(mdempsky): Omit? Not strictly necessary; only matters for
		// type declarations within function literals within parameterized
		// type declarations, but types2 the function literals will be
		// constant folded away.
		return c.withTParams(obj)

	case *syntax.VarDecl:
		pw.checkPragmas(n.Pragma, 0, true)

		if p, ok := n.Pragma.(*pragmas); ok && len(p.Embeds) > 0 {
			if err := checkEmbed(n, c.file.importedEmbed, c.withinFunc); err != nil {
				pw.errorf(p.Embeds[0].Pos, "%s", err)
			}
		}

	case *syntax.BlockStmt:
		if !c.withinFunc {
			copy := *c
			copy.withinFunc = true
			return &copy
		}
	}

	return c
}

func (pw *pkgWriter) collectDecls(noders []*noder) {
	var typegen int
	for _, p := range noders {
		var file fileImports

		syntax.Walk(p.file, &declCollector{
			pw:      pw,
			typegen: &typegen,
			file:    &file,
		})

		pw.cgoPragmas = append(pw.cgoPragmas, p.pragcgobuf...)

		for _, l := range p.linknames {
			if !file.importedUnsafe {
				pw.errorf(l.pos, "//go:linkname only allowed in Go files that import \"unsafe\"")
				continue
			}

			switch obj := pw.curpkg.Scope().Lookup(l.local).(type) {
			case *types2.Func, *types2.Var:
				if _, ok := pw.linknames[obj]; !ok {
					pw.linknames[obj] = l.remote
				} else {
					pw.errorf(l.pos, "duplicate //go:linkname for %s", l.local)
				}

			default:
				// TODO(mdempsky): Enable after #42938 is fixed.
				if false {
					pw.errorf(l.pos, "//go:linkname must refer to declared function or variable")
				}
			}
		}
	}
}

func (pw *pkgWriter) checkPragmas(p syntax.Pragma, allowed ir.PragmaFlag, embedOK bool) {
	if p == nil {
		return
	}
	pragma := p.(*pragmas)

	for _, pos := range pragma.Pos {
		if pos.Flag&^allowed != 0 {
			pw.errorf(pos.Pos, "misplaced compiler directive")
		}
	}

	if !embedOK {
		for _, e := range pragma.Embeds {
			pw.errorf(e.Pos, "misplaced go:embed directive")
		}
	}
}

func (w *writer) pkgInit(noders []*noder) {
	w.Len(len(w.p.cgoPragmas))
	for _, cgoPragma := range w.p.cgoPragmas {
		w.Strings(cgoPragma)
	}

	w.Sync(pkgbits.SyncDecls)
	for _, p := range noders {
		for _, decl := range p.file.DeclList {
			w.pkgDecl(decl)
		}
	}
	w.Code(declEnd)

	w.Sync(pkgbits.SyncEOF)
}

func (w *writer) pkgDecl(decl syntax.Decl) {
	switch decl := decl.(type) {
	default:
		w.p.unexpected("declaration", decl)

	case *syntax.ImportDecl:

	case *syntax.ConstDecl:
		w.Code(declOther)
		w.pkgObjs(decl.NameList...)

	case *syntax.FuncDecl:
		if decl.Name.Value == "_" {
			break // skip blank functions
		}

		obj := w.p.info.Defs[decl.Name].(*types2.Func)
		sig := obj.Type().(*types2.Signature)

		if sig.RecvTypeParams() != nil || sig.TypeParams() != nil {
			break // skip generic functions
		}

		if recv := sig.Recv(); recv != nil {
			w.Code(declMethod)
			w.typ(recvBase(recv))
			w.selector(obj)
			break
		}

		w.Code(declFunc)
		w.pkgObjs(decl.Name)

	case *syntax.TypeDecl:
		if len(decl.TParamList) != 0 {
			break // skip generic type decls
		}

		if decl.Name.Value == "_" {
			break // skip blank type decls
		}

		name := w.p.info.Defs[decl.Name].(*types2.TypeName)
		// Skip type declarations for interfaces that are only usable as
		// type parameter bounds.
		if iface, ok := name.Type().Underlying().(*types2.Interface); ok && !iface.IsMethodSet() {
			break
		}

		w.Code(declOther)
		w.pkgObjs(decl.Name)

	case *syntax.VarDecl:
		w.Code(declVar)
		w.pos(decl)
		w.pkgObjs(decl.NameList...)
		w.exprList(decl.Values)

		var embeds []pragmaEmbed
		if p, ok := decl.Pragma.(*pragmas); ok {
			embeds = p.Embeds
		}
		w.Len(len(embeds))
		for _, embed := range embeds {
			w.pos(embed.Pos)
			w.Strings(embed.Patterns)
		}
	}
}

func (w *writer) pkgObjs(names ...*syntax.Name) {
	w.Sync(pkgbits.SyncDeclNames)
	w.Len(len(names))

	for _, name := range names {
		obj, ok := w.p.info.Defs[name]
		assert(ok)

		w.Sync(pkgbits.SyncDeclName)
		w.obj(obj, nil)
	}
}

// @@@ Helpers

// isDefinedType reports whether obj is a defined type.
func isDefinedType(obj types2.Object) bool {
	if obj, ok := obj.(*types2.TypeName); ok {
		return !obj.IsAlias()
	}
	return false
}

// isGlobal reports whether obj was declared at package scope.
//
// Caveat: blank objects are not declared.
func isGlobal(obj types2.Object) bool {
	return obj.Parent() == obj.Pkg().Scope()
}

// lookupObj returns the object that expr refers to, if any. If expr
// is an explicit instantiation of a generic object, then the instance
// object is returned as well.
func lookupObj(info *types2.Info, expr syntax.Expr) (obj types2.Object, inst types2.Instance) {
	if index, ok := expr.(*syntax.IndexExpr); ok {
		args := unpackListExpr(index.Index)
		if len(args) == 1 {
			tv, ok := info.Types[args[0]]
			assert(ok)
			if tv.IsValue() {
				return // normal index expression
			}
		}

		expr = index.X
	}

	// Strip package qualifier, if present.
	if sel, ok := expr.(*syntax.SelectorExpr); ok {
		if !isPkgQual(info, sel) {
			return // normal selector expression
		}
		expr = sel.Sel
	}

	if name, ok := expr.(*syntax.Name); ok {
		obj = info.Uses[name]
		inst = info.Instances[name]
	}
	return
}

// isPkgQual reports whether the given selector expression is a
// package-qualified identifier.
func isPkgQual(info *types2.Info, sel *syntax.SelectorExpr) bool {
	if name, ok := sel.X.(*syntax.Name); ok {
		_, isPkgName := info.Uses[name].(*types2.PkgName)
		return isPkgName
	}
	return false
}

// recvBase returns the base type for the given receiver parameter.
func recvBase(recv *types2.Var) *types2.Named {
	typ := recv.Type()
	if ptr, ok := typ.(*types2.Pointer); ok {
		typ = ptr.Elem()
	}
	return typ.(*types2.Named)
}

// namesAsExpr returns a list of names as a syntax.Expr.
func namesAsExpr(names []*syntax.Name) syntax.Expr {
	if len(names) == 1 {
		return names[0]
	}

	exprs := make([]syntax.Expr, len(names))
	for i, name := range names {
		exprs[i] = name
	}
	return &syntax.ListExpr{ElemList: exprs}
}

// fieldIndex returns the index of the struct field named by key.
func fieldIndex(info *types2.Info, str *types2.Struct, key *syntax.Name) int {
	field := info.Uses[key].(*types2.Var)

	for i := 0; i < str.NumFields(); i++ {
		if str.Field(i) == field {
			return i
		}
	}

	panic(fmt.Sprintf("%s: %v is not a field of %v", key.Pos(), field, str))
}

// objTypeParams returns the type parameters on the given object.
func objTypeParams(obj types2.Object) *types2.TypeParamList {
	switch obj := obj.(type) {
	case *types2.Func:
		sig := obj.Type().(*types2.Signature)
		if sig.Recv() != nil {
			return sig.RecvTypeParams()
		}
		return sig.TypeParams()
	case *types2.TypeName:
		if !obj.IsAlias() {
			return obj.Type().(*types2.Named).TypeParams()
		}
	}
	return nil
}

func asPragmaFlag(p syntax.Pragma) ir.PragmaFlag {
	if p == nil {
		return 0
	}
	return p.(*pragmas).Flag
}
