// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type pkgWriter struct {
	pkgEncoder

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

	dups dupTypes
}

func newPkgWriter(m posMap, pkg *types2.Package, info *types2.Info) *pkgWriter {
	return &pkgWriter{
		pkgEncoder: newPkgEncoder(),

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

	encoder

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

func (pw *pkgWriter) newWriter(k reloc, marker syncMarker) *writer {
	return &writer{
		encoder: pw.newEncoder(k, marker),
		p:       pw,
	}
}

// @@@ Positions

func (w *writer) pos(p poser) {
	w.sync(syncPos)
	pos := p.Pos()

	// TODO(mdempsky): Track down the remaining cases here and fix them.
	if !w.bool(pos.IsKnown()) {
		return
	}

	// TODO(mdempsky): Delta encoding. Also, if there's a b-side, update
	// its position base too (but not vice versa!).
	w.posBase(pos.Base())
	w.uint(pos.Line())
	w.uint(pos.Col())
}

func (w *writer) posBase(b *syntax.PosBase) {
	w.reloc(relocPosBase, w.p.posBaseIdx(b))
}

func (pw *pkgWriter) posBaseIdx(b *syntax.PosBase) int {
	if idx, ok := pw.posBasesIdx[b]; ok {
		return idx
	}

	w := pw.newWriter(relocPosBase, syncPosBase)
	w.p.posBasesIdx[b] = w.idx

	w.string(trimFilename(b))

	if !w.bool(b.IsFileBase()) {
		w.pos(b)
		w.uint(b.Line())
		w.uint(b.Col())
	}

	return w.flush()
}

// @@@ Packages

func (w *writer) pkg(pkg *types2.Package) {
	w.sync(syncPkg)
	w.reloc(relocPkg, w.p.pkgIdx(pkg))
}

func (pw *pkgWriter) pkgIdx(pkg *types2.Package) int {
	if idx, ok := pw.pkgsIdx[pkg]; ok {
		return idx
	}

	w := pw.newWriter(relocPkg, syncPkgDef)
	pw.pkgsIdx[pkg] = w.idx

	if pkg == nil {
		w.string("builtin")
	} else {
		var path string
		if pkg != w.p.curpkg {
			path = pkg.Path()
		}
		w.string(path)
		w.string(pkg.Name())
		w.len(pkg.Height())

		w.len(len(pkg.Imports()))
		for _, imp := range pkg.Imports() {
			w.pkg(imp)
		}
	}

	return w.flush()
}

// @@@ Types

var anyTypeName = types2.Universe.Lookup("any").(*types2.TypeName)

func (w *writer) typ(typ types2.Type) {
	w.typInfo(w.p.typIdx(typ, w.dict))
}

func (w *writer) typInfo(info typeInfo) {
	w.sync(syncType)
	if w.bool(info.derived) {
		w.len(info.idx)
		w.derived = true
	} else {
		w.reloc(relocType, info.idx)
	}
}

// typIdx returns the index where the export data description of type
// can be read back in. If no such index exists yet, it's created.
//
// typIdx also reports whether typ is a derived type; that is, whether
// its identity depends on type parameters.
func (pw *pkgWriter) typIdx(typ types2.Type, dict *writerDict) typeInfo {
	if quirksMode() {
		typ = pw.dups.orig(typ)
	}

	if idx, ok := pw.typsIdx[typ]; ok {
		return typeInfo{idx: idx, derived: false}
	}
	if dict != nil {
		if idx, ok := dict.derivedIdx[typ]; ok {
			return typeInfo{idx: idx, derived: true}
		}
	}

	w := pw.newWriter(relocType, syncTypeIdx)
	w.dict = dict

	switch typ := typ.(type) {
	default:
		base.Fatalf("unexpected type: %v (%T)", typ, typ)

	case *types2.Basic:
		switch kind := typ.Kind(); {
		case kind == types2.Invalid:
			base.Fatalf("unexpected types2.Invalid")

		case types2.Typ[kind] == typ:
			w.code(typeBasic)
			w.len(int(kind))

		default:
			// Handle "byte" and "rune" as references to their TypeName.
			obj := types2.Universe.Lookup(typ.Name())
			assert(obj.Type() == typ)

			w.code(typeNamed)
			w.obj(obj, nil)
		}

	case *types2.Named:
		// Type aliases can refer to uninstantiated generic types, so we
		// might see len(TParams) != 0 && len(TArgs) == 0 here.
		// TODO(mdempsky): Revisit after #46477 is resolved.
		assert(typ.TypeParams().Len() == typ.TypeArgs().Len() || typ.TypeArgs().Len() == 0)

		// TODO(mdempsky): Why do we need to loop here?
		orig := typ
		for orig.TypeArgs() != nil {
			orig = orig.Origin()
		}

		w.code(typeNamed)
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
		w.code(typeTypeParam)
		w.len(index)

	case *types2.Array:
		w.code(typeArray)
		w.uint64(uint64(typ.Len()))
		w.typ(typ.Elem())

	case *types2.Chan:
		w.code(typeChan)
		w.len(int(typ.Dir()))
		w.typ(typ.Elem())

	case *types2.Map:
		w.code(typeMap)
		w.typ(typ.Key())
		w.typ(typ.Elem())

	case *types2.Pointer:
		w.code(typePointer)
		w.typ(typ.Elem())

	case *types2.Signature:
		base.Assertf(typ.TypeParams() == nil, "unexpected type params: %v", typ)
		w.code(typeSignature)
		w.signature(typ)

	case *types2.Slice:
		w.code(typeSlice)
		w.typ(typ.Elem())

	case *types2.Struct:
		w.code(typeStruct)
		w.structType(typ)

	case *types2.Interface:
		if typ == anyTypeName.Type() {
			w.code(typeNamed)
			w.obj(anyTypeName, nil)
			break
		}

		w.code(typeInterface)
		w.interfaceType(typ)

	case *types2.Union:
		w.code(typeUnion)
		w.unionType(typ)
	}

	if w.derived {
		idx := len(dict.derived)
		dict.derived = append(dict.derived, derivedInfo{idx: w.flush()})
		dict.derivedIdx[typ] = idx
		return typeInfo{idx: idx, derived: true}
	}

	pw.typsIdx[typ] = w.idx
	return typeInfo{idx: w.flush(), derived: false}
}

func (w *writer) structType(typ *types2.Struct) {
	w.len(typ.NumFields())
	for i := 0; i < typ.NumFields(); i++ {
		f := typ.Field(i)
		w.pos(f)
		w.selector(f)
		w.typ(f.Type())
		w.string(typ.Tag(i))
		w.bool(f.Embedded())
	}
}

func (w *writer) unionType(typ *types2.Union) {
	w.len(typ.Len())
	for i := 0; i < typ.Len(); i++ {
		t := typ.Term(i)
		w.bool(t.Tilde())
		w.typ(t.Type())
	}
}

func (w *writer) interfaceType(typ *types2.Interface) {
	w.len(typ.NumExplicitMethods())
	w.len(typ.NumEmbeddeds())

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
	w.sync(syncSignature)
	w.params(sig.Params())
	w.params(sig.Results())
	w.bool(sig.Variadic())
}

func (w *writer) params(typ *types2.Tuple) {
	w.sync(syncParams)
	w.len(typ.Len())
	for i := 0; i < typ.Len(); i++ {
		w.param(typ.At(i))
	}
}

func (w *writer) param(param *types2.Var) {
	w.sync(syncParam)
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
		w.sync(syncObject)
		w.bool(true)
		w.len(idx)
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

	w.sync(syncObject)
	w.bool(false)
	w.reloc(relocObj, info.idx)

	w.len(len(info.explicits))
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

	w := pw.newWriter(relocObj, syncObject1)
	wext := pw.newWriter(relocObjExt, syncObject1)
	wname := pw.newWriter(relocName, syncObject1)
	wdict := pw.newWriter(relocObjDict, syncObject1)

	pw.globalsIdx[obj] = w.idx // break cycles
	assert(wext.idx == w.idx)
	assert(wname.idx == w.idx)
	assert(wdict.idx == w.idx)

	w.dict = dict
	wext.dict = dict

	code := w.doObj(wext, obj)
	w.flush()
	wext.flush()

	wname.qualifiedIdent(obj)
	wname.code(code)
	wname.flush()

	wdict.objDict(obj, w.dict)
	wdict.flush()

	return w.idx
}

func (w *writer) doObj(wext *writer, obj types2.Object) codeObj {
	if obj.Pkg() != w.p.curpkg {
		return objStub
	}

	switch obj := obj.(type) {
	default:
		w.p.unexpected("object", obj)
		panic("unreachable")

	case *types2.Const:
		w.pos(obj)
		w.typ(obj.Type())
		w.value(obj.Val())
		return objConst

	case *types2.Func:
		decl, ok := w.p.funDecls[obj]
		assert(ok)
		sig := obj.Type().(*types2.Signature)

		w.pos(obj)
		w.typeParamNames(sig.TypeParams())
		w.signature(sig)
		w.pos(decl)
		wext.funcExt(obj)
		return objFunc

	case *types2.TypeName:
		decl, ok := w.p.typDecls[obj]
		assert(ok)

		if obj.IsAlias() {
			w.pos(obj)
			w.typ(obj.Type())
			return objAlias
		}

		named := obj.Type().(*types2.Named)
		assert(named.TypeArgs() == nil)

		w.pos(obj)
		w.typeParamNames(named.TypeParams())
		wext.typeExt(obj)
		w.typExpr(decl.Type)

		w.len(named.NumMethods())
		for i := 0; i < named.NumMethods(); i++ {
			w.method(wext, named.Method(i))
		}

		return objType

	case *types2.Var:
		w.pos(obj)
		w.typ(obj.Type())
		wext.varExt(obj)
		return objVar
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

	w.len(len(dict.implicits))

	tparams := objTypeParams(obj)
	ntparams := tparams.Len()
	w.len(ntparams)
	for i := 0; i < ntparams; i++ {
		w.typ(tparams.At(i).Constraint())
	}

	nderived := len(dict.derived)
	w.len(nderived)
	for _, typ := range dict.derived {
		w.reloc(relocType, typ.idx)
		w.bool(typ.needed)
	}

	nfuncs := len(dict.funcs)
	w.len(nfuncs)
	for _, fn := range dict.funcs {
		w.reloc(relocObj, fn.idx)
		w.len(len(fn.explicits))
		for _, targ := range fn.explicits {
			w.typInfo(targ)
		}
	}

	assert(len(dict.derived) == nderived)
	assert(len(dict.funcs) == nfuncs)
}

func (w *writer) typeParamNames(tparams *types2.TypeParamList) {
	w.sync(syncTypeParamNames)

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

	w.sync(syncMethod)
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
	w.sync(syncSym)

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
	w.string(name)
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
	w.sync(syncLocalIdent)
	w.pkg(obj.Pkg())
	w.string(obj.Name())
}

// selector writes the name of a field or method (i.e., objects that
// can only be accessed using selector expressions).
func (w *writer) selector(obj types2.Object) {
	w.sync(syncSelector)
	w.pkg(obj.Pkg())
	w.string(obj.Name())
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

	w.sync(syncFuncExt)
	w.pragmaFlag(pragma)
	w.linkname(obj)
	w.bool(false) // stub extension
	w.reloc(relocBody, body)
	w.sync(syncEOF)
}

func (w *writer) typeExt(obj *types2.TypeName) {
	decl, ok := w.p.typDecls[obj]
	assert(ok)

	w.sync(syncTypeExt)

	w.pragmaFlag(asPragmaFlag(decl.Pragma))

	// No LSym.SymIdx info yet.
	w.int64(-1)
	w.int64(-1)
}

func (w *writer) varExt(obj *types2.Var) {
	w.sync(syncVarExt)
	w.linkname(obj)
}

func (w *writer) linkname(obj types2.Object) {
	w.sync(syncLinkname)
	w.int64(-1)
	w.string(w.p.linknames[obj])
}

func (w *writer) pragmaFlag(p ir.PragmaFlag) {
	w.sync(syncPragma)
	w.int(int(p))
}

// @@@ Function bodies

func (pw *pkgWriter) bodyIdx(pkg *types2.Package, sig *types2.Signature, block *syntax.BlockStmt, dict *writerDict) (idx int, closureVars []posObj) {
	w := pw.newWriter(relocBody, syncFuncBody)
	w.dict = dict

	w.funcargs(sig)
	if w.bool(block != nil) {
		w.stmts(block.List)
		w.pos(block.Rbrace)
	}

	return w.flush(), w.closureVars
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
	w.sync(syncAddLocal)
	idx := len(w.localsIdx)
	if enableSync {
		w.int(idx)
	}
	if w.localsIdx == nil {
		w.localsIdx = make(map[*types2.Var]int)
	}
	w.localsIdx[obj] = idx
}

func (w *writer) useLocal(pos syntax.Pos, obj *types2.Var) {
	w.sync(syncUseObjLocal)

	if idx, ok := w.localsIdx[obj]; w.bool(ok) {
		w.len(idx)
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
	w.len(idx)
}

func (w *writer) openScope(pos syntax.Pos) {
	w.sync(syncOpenScope)
	w.pos(pos)
}

func (w *writer) closeScope(pos syntax.Pos) {
	w.sync(syncCloseScope)
	w.pos(pos)
	w.closeAnotherScope()
}

func (w *writer) closeAnotherScope() {
	w.sync(syncCloseAnotherScope)
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
	w.sync(syncStmts)
	for _, stmt := range stmts {
		w.stmt1(stmt)
	}
	w.code(stmtEnd)
	w.sync(syncStmtsEnd)
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
			w.code(stmtIncDec)
			w.op(binOps[stmt.Op])
			w.expr(stmt.Lhs)
			w.pos(stmt)

		case stmt.Op != 0 && stmt.Op != syntax.Def:
			w.code(stmtAssignOp)
			w.op(binOps[stmt.Op])
			w.expr(stmt.Lhs)
			w.pos(stmt)
			w.expr(stmt.Rhs)

		default:
			w.code(stmtAssign)
			w.pos(stmt)
			w.exprList(stmt.Rhs)
			w.assignList(stmt.Lhs)
		}

	case *syntax.BlockStmt:
		w.code(stmtBlock)
		w.blockStmt(stmt)

	case *syntax.BranchStmt:
		w.code(stmtBranch)
		w.pos(stmt)
		w.op(branchOps[stmt.Tok])
		w.optLabel(stmt.Label)

	case *syntax.CallStmt:
		w.code(stmtCall)
		w.pos(stmt)
		w.op(callOps[stmt.Tok])
		w.expr(stmt.Call)

	case *syntax.DeclStmt:
		for _, decl := range stmt.DeclList {
			w.declStmt(decl)
		}

	case *syntax.ExprStmt:
		w.code(stmtExpr)
		w.expr(stmt.X)

	case *syntax.ForStmt:
		w.code(stmtFor)
		w.forStmt(stmt)

	case *syntax.IfStmt:
		w.code(stmtIf)
		w.ifStmt(stmt)

	case *syntax.LabeledStmt:
		w.code(stmtLabel)
		w.pos(stmt)
		w.label(stmt.Label)
		w.stmt1(stmt.Stmt)

	case *syntax.ReturnStmt:
		w.code(stmtReturn)
		w.pos(stmt)
		w.exprList(stmt.Results)

	case *syntax.SelectStmt:
		w.code(stmtSelect)
		w.selectStmt(stmt)

	case *syntax.SendStmt:
		w.code(stmtSend)
		w.pos(stmt)
		w.expr(stmt.Chan)
		w.expr(stmt.Value)

	case *syntax.SwitchStmt:
		w.code(stmtSwitch)
		w.switchStmt(stmt)
	}
}

func (w *writer) assignList(expr syntax.Expr) {
	exprs := unpackListExpr(expr)
	w.len(len(exprs))

	for _, expr := range exprs {
		if name, ok := expr.(*syntax.Name); ok && name.Value != "_" {
			if obj, ok := w.p.info.Defs[name]; ok {
				obj := obj.(*types2.Var)

				w.bool(true)
				w.pos(obj)
				w.localIdent(obj)
				w.typ(obj.Type())

				// TODO(mdempsky): Minimize locals index size by deferring
				// this until the variables actually come into scope.
				w.addLocal(obj)
				continue
			}
		}

		w.bool(false)
		w.expr(expr)
	}
}

func (w *writer) declStmt(decl syntax.Decl) {
	switch decl := decl.(type) {
	default:
		w.p.unexpected("declaration", decl)

	case *syntax.ConstDecl:

	case *syntax.TypeDecl:
		// Quirk: The legacy inliner doesn't support inlining functions
		// with type declarations. Unified IR doesn't have any need to
		// write out type declarations explicitly (they're always looked
		// up via global index tables instead), so we just write out a
		// marker so the reader knows to synthesize a fake declaration to
		// prevent inlining.
		if quirksMode() {
			w.code(stmtTypeDeclHack)
		}

	case *syntax.VarDecl:
		values := unpackListExpr(decl.Values)

		// Quirk: When N variables are declared with N initialization
		// values, we need to decompose that into N interleaved
		// declarations+initializations, because it leads to different
		// (albeit semantically equivalent) code generation.
		if quirksMode() && len(decl.NameList) == len(values) {
			for i, name := range decl.NameList {
				w.code(stmtAssign)
				w.pos(decl)
				w.exprList(values[i])
				w.assignList(name)
			}
			break
		}

		w.code(stmtAssign)
		w.pos(decl)
		w.exprList(decl.Values)
		w.assignList(namesAsExpr(decl.NameList))
	}
}

func (w *writer) blockStmt(stmt *syntax.BlockStmt) {
	w.sync(syncBlockStmt)
	w.openScope(stmt.Pos())
	w.stmts(stmt.List)
	w.closeScope(stmt.Rbrace)
}

func (w *writer) forStmt(stmt *syntax.ForStmt) {
	w.sync(syncForStmt)
	w.openScope(stmt.Pos())

	if rang, ok := stmt.Init.(*syntax.RangeClause); w.bool(ok) {
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
	w.sync(syncIfStmt)
	w.openScope(stmt.Pos())
	w.pos(stmt)
	w.stmt(stmt.Init)
	w.expr(stmt.Cond)
	w.blockStmt(stmt.Then)
	w.stmt(stmt.Else)
	w.closeAnotherScope()
}

func (w *writer) selectStmt(stmt *syntax.SelectStmt) {
	w.sync(syncSelectStmt)

	w.pos(stmt)
	w.len(len(stmt.Body))
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
	w.sync(syncSwitchStmt)

	w.openScope(stmt.Pos())
	w.pos(stmt)
	w.stmt(stmt.Init)

	if guard, ok := stmt.Tag.(*syntax.TypeSwitchGuard); w.bool(ok) {
		w.pos(guard)
		if tag := guard.Lhs; w.bool(tag != nil) {
			w.pos(tag)
			w.string(tag.Value)
		}
		w.expr(guard.X)
	} else {
		w.expr(stmt.Tag)
	}

	w.len(len(stmt.Body))
	for i, clause := range stmt.Body {
		if i > 0 {
			w.closeScope(clause.Pos())
		}
		w.openScope(clause.Pos())

		w.pos(clause)
		w.exprList(clause.Cases)

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
	w.sync(syncLabel)

	// TODO(mdempsky): Replace label strings with dense indices.
	w.string(label.Value)
}

func (w *writer) optLabel(label *syntax.Name) {
	w.sync(syncOptLabel)
	if w.bool(label != nil) {
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
			w.code(exprType)
			w.typ(tv.Type)
			return
		}

		if tv.Value != nil {
			pos := expr.Pos()
			if quirksMode() {
				if obj != nil {
					// Quirk: IR (and thus iexport) doesn't track position
					// information for uses of declared objects.
					pos = syntax.Pos{}
				} else if tv.Value.Kind() == constant.String {
					// Quirk: noder.sum picks a particular position for certain
					// string concatenations.
					pos = sumPos(expr)
				}
			}

			w.code(exprConst)
			w.pos(pos)
			w.typ(tv.Type)
			w.value(tv.Value)

			// TODO(mdempsky): These details are only important for backend
			// diagnostics. Explore writing them out separately.
			w.op(constExprOp(expr))
			w.string(syntax.String(expr))
			return
		}
	}

	if obj != nil {
		if isGlobal(obj) {
			w.code(exprName)
			w.obj(obj, targs)
			return
		}

		obj := obj.(*types2.Var)
		assert(!obj.IsField())
		assert(targs.Len() == 0)

		w.code(exprLocal)
		w.useLocal(expr.Pos(), obj)
		return
	}

	switch expr := expr.(type) {
	default:
		w.p.unexpected("expression", expr)

	case nil: // absent slice index, for condition, or switch tag
		w.code(exprNone)

	case *syntax.Name:
		assert(expr.Value == "_")
		w.code(exprBlank)

	case *syntax.CompositeLit:
		w.code(exprCompLit)
		w.compLit(expr)

	case *syntax.FuncLit:
		w.code(exprFuncLit)
		w.funcLit(expr)

	case *syntax.SelectorExpr:
		sel, ok := w.p.info.Selections[expr]
		assert(ok)

		w.code(exprSelector)
		w.expr(expr.X)
		w.pos(expr)
		w.selector(sel.Obj())

	case *syntax.IndexExpr:
		tv, ok := w.p.info.Types[expr.Index]
		assert(ok && tv.IsValue())

		w.code(exprIndex)
		w.expr(expr.X)
		w.pos(expr)
		w.expr(expr.Index)

	case *syntax.SliceExpr:
		w.code(exprSlice)
		w.expr(expr.X)
		w.pos(expr)
		for _, n := range &expr.Index {
			w.expr(n)
		}

	case *syntax.AssertExpr:
		w.code(exprAssert)
		w.expr(expr.X)
		w.pos(expr)
		w.expr(expr.Type)

	case *syntax.Operation:
		if expr.Y == nil {
			w.code(exprUnaryOp)
			w.op(unOps[expr.Op])
			w.pos(expr)
			w.expr(expr.X)
			break
		}

		w.code(exprBinaryOp)
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

			w.code(exprConvert)
			w.typ(tv.Type)
			w.pos(expr)
			w.expr(expr.ArgList[0])
			break
		}

		writeFunExpr := func() {
			if selector, ok := unparen(expr.Fun).(*syntax.SelectorExpr); ok {
				if sel, ok := w.p.info.Selections[selector]; ok && sel.Kind() == types2.MethodVal {
					w.expr(selector.X)
					w.bool(true) // method call
					w.pos(selector)
					w.selector(sel.Obj())
					return
				}
			}

			w.expr(expr.Fun)
			w.bool(false) // not a method call (i.e., normal function call)
		}

		w.code(exprCall)
		writeFunExpr()
		w.pos(expr)
		w.exprs(expr.ArgList)
		w.bool(expr.HasDots)
	}
}

func (w *writer) compLit(lit *syntax.CompositeLit) {
	tv, ok := w.p.info.Types[lit]
	assert(ok)

	w.sync(syncCompLit)
	w.pos(lit)
	w.typ(tv.Type)

	typ := tv.Type
	if ptr, ok := types2.CoreType(typ).(*types2.Pointer); ok {
		typ = ptr.Elem()
	}
	str, isStruct := types2.CoreType(typ).(*types2.Struct)

	w.len(len(lit.ElemList))
	for i, elem := range lit.ElemList {
		if isStruct {
			if kv, ok := elem.(*syntax.KeyValueExpr); ok {
				// use position of expr.Key rather than of elem (which has position of ':')
				w.pos(kv.Key)
				w.len(fieldIndex(w.p.info, str, kv.Key.(*syntax.Name)))
				elem = kv.Value
			} else {
				w.pos(elem)
				w.len(i)
			}
		} else {
			if kv, ok := elem.(*syntax.KeyValueExpr); w.bool(ok) {
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

	w.sync(syncFuncLit)
	w.pos(expr)
	w.pos(expr.Type) // for QuirksMode
	w.signature(sig)

	w.len(len(closureVars))
	for _, cv := range closureVars {
		w.pos(cv.pos)
		if quirksMode() {
			cv.pos = expr.Body.Rbrace
		}
		w.useLocal(cv.pos, cv.obj)
	}

	w.reloc(relocBody, body)
}

type posObj struct {
	pos syntax.Pos
	obj *types2.Var
}

func (w *writer) exprList(expr syntax.Expr) {
	w.sync(syncExprList)
	w.exprs(unpackListExpr(expr))
}

func (w *writer) exprs(exprs []syntax.Expr) {
	if len(exprs) == 0 {
		assert(exprs == nil)
	}

	w.sync(syncExprs)
	w.len(len(exprs))
	for _, expr := range exprs {
		w.expr(expr)
	}
}

func (w *writer) op(op ir.Op) {
	// TODO(mdempsky): Remove in favor of explicit codes? Would make
	// export data more stable against internal refactorings, but low
	// priority at the moment.
	assert(op != 0)
	w.sync(syncOp)
	w.len(int(op))
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

		// Workaround for #46208. For variable declarations that
		// declare multiple variables and have an explicit type
		// expression, the type expression is evaluated multiple
		// times. This affects toolstash -cmp, because iexport is
		// sensitive to *types.Type pointer identity.
		if quirksMode() && n.Type != nil {
			tv, ok := pw.info.Types[n.Type]
			assert(ok)
			assert(tv.IsType())
			for _, name := range n.NameList {
				obj := pw.info.Defs[name].(*types2.Var)
				pw.dups.add(obj.Type(), tv.Type)
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
	if quirksMode() {
		posBases := posBasesOf(noders)
		w.len(len(posBases))
		for _, posBase := range posBases {
			w.posBase(posBase)
		}

		objs := importedObjsOf(w.p.curpkg, w.p.info, noders)
		w.len(len(objs))
		for _, obj := range objs {
			w.qualifiedIdent(obj)
		}
	}

	w.len(len(w.p.cgoPragmas))
	for _, cgoPragma := range w.p.cgoPragmas {
		w.strings(cgoPragma)
	}

	w.sync(syncDecls)
	for _, p := range noders {
		for _, decl := range p.file.DeclList {
			w.pkgDecl(decl)
		}
	}
	w.code(declEnd)

	w.sync(syncEOF)
}

func (w *writer) pkgDecl(decl syntax.Decl) {
	switch decl := decl.(type) {
	default:
		w.p.unexpected("declaration", decl)

	case *syntax.ImportDecl:

	case *syntax.ConstDecl:
		w.code(declOther)
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
			w.code(declMethod)
			w.typ(recvBase(recv))
			w.selector(obj)
			break
		}

		w.code(declFunc)
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

		// Skip aliases to uninstantiated generic types.
		// TODO(mdempsky): Revisit after #46477 is resolved.
		if name.IsAlias() {
			named, ok := name.Type().(*types2.Named)
			if ok && named.TypeParams().Len() != 0 && named.TypeArgs().Len() == 0 {
				break
			}
		}

		w.code(declOther)
		w.pkgObjs(decl.Name)

	case *syntax.VarDecl:
		w.code(declVar)
		w.pos(decl)
		w.pkgObjs(decl.NameList...)
		w.exprList(decl.Values)

		var embeds []pragmaEmbed
		if p, ok := decl.Pragma.(*pragmas); ok {
			embeds = p.Embeds
		}
		w.len(len(embeds))
		for _, embed := range embeds {
			w.pos(embed.Pos)
			w.strings(embed.Patterns)
		}
	}
}

func (w *writer) pkgObjs(names ...*syntax.Name) {
	w.sync(syncDeclNames)
	w.len(len(names))

	for _, name := range names {
		obj, ok := w.p.info.Defs[name]
		assert(ok)

		w.sync(syncDeclName)
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
