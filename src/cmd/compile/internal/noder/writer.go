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

	// For writing out object descriptions, ext points to the extension
	// writer for where we can write the compiler's private extension
	// details for the object.
	//
	// TODO(mdempsky): This is a little hacky, but works easiest with
	// the way things are currently.
	ext *writer

	// TODO(mdempsky): We should be able to prune localsIdx whenever a
	// scope closes, and then maybe we can just use the same map for
	// storing the TypeParams too (as their TypeName instead).

	// type parameters. explicitIdx has the type parameters declared on
	// the current object, while implicitIdx has the type parameters
	// declared on the enclosing object (if any).
	//
	// TODO(mdempsky): Merge these back together, now that I've got them
	// working.
	implicitIdx map[*types2.TypeParam]int
	explicitIdx map[*types2.TypeParam]int

	// variables declared within this function
	localsIdx map[types2.Object]int
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

	// TODO(mdempsky): What exactly does "fileh" do anyway? Is writing
	// out both of these strings really the right thing to do here?
	fn := b.Filename()
	w.string(fn)
	w.string(fileh(fn))

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

func (w *writer) typ(typ types2.Type) {
	w.sync(syncType)

	if quirksMode() {
		typ = w.p.dups.orig(typ)
	}

	w.reloc(relocType, w.p.typIdx(typ, w.implicitIdx, w.explicitIdx))
}

func (pw *pkgWriter) typIdx(typ types2.Type, implicitIdx, explicitIdx map[*types2.TypeParam]int) int {
	if idx, ok := pw.typsIdx[typ]; ok {
		return idx
	}

	w := pw.newWriter(relocType, syncTypeIdx)
	w.implicitIdx = implicitIdx
	w.explicitIdx = explicitIdx

	pw.typsIdx[typ] = w.idx // handle cycles
	w.doTyp(typ)
	return w.flush()
}

func (w *writer) doTyp(typ types2.Type) {
	switch typ := typ.(type) {
	default:
		base.Fatalf("unexpected type: %v (%T)", typ, typ)

	case *types2.Basic:
		if kind := typ.Kind(); types2.Typ[kind] == typ {
			w.code(typeBasic)
			w.len(int(kind))
			break
		}

		// Handle "byte" and "rune" as references to their TypeName.
		obj := types2.Universe.Lookup(typ.Name())
		assert(obj.Type() == typ)

		w.code(typeNamed)
		w.obj(obj, nil)

	case *types2.Named:
		// Type aliases can refer to uninstantiated generic types, so we
		// might see len(TParams) != 0 && len(TArgs) == 0 here.
		// TODO(mdempsky): Revisit after #46477 is resolved.
		assert(len(typ.TParams()) == len(typ.TArgs()) || len(typ.TArgs()) == 0)

		// TODO(mdempsky): Why do we need to loop here?
		orig := typ
		for orig.TArgs() != nil {
			orig = orig.Orig()
		}

		w.code(typeNamed)
		w.obj(orig.Obj(), typ.TArgs())

	case *types2.TypeParam:
		w.code(typeTypeParam)
		if idx, ok := w.implicitIdx[typ]; ok {
			w.len(idx)
		} else if idx, ok := w.explicitIdx[typ]; ok {
			w.len(len(w.implicitIdx) + idx)
		} else {
			w.p.fatalf(typ.Obj(), "%v not in %v or %v", typ, w.implicitIdx, w.explicitIdx)
		}

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
		assert(typ.TParams() == nil)
		w.code(typeSignature)
		w.signature(typ)

	case *types2.Slice:
		w.code(typeSlice)
		w.typ(typ.Elem())

	case *types2.Struct:
		w.code(typeStruct)
		w.structType(typ)

	case *types2.Interface:
		w.code(typeInterface)
		w.interfaceType(typ)

	case *types2.Union:
		w.code(typeUnion)
		w.unionType(typ)
	}
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
	w.len(typ.NumTerms())
	for i := 0; i < typ.NumTerms(); i++ {
		term, tilde := typ.Term(i)
		w.typ(term)
		w.bool(tilde)
	}
}

func (w *writer) interfaceType(typ *types2.Interface) {
	w.len(typ.NumExplicitMethods())
	w.len(typ.NumEmbeddeds())

	for i := 0; i < typ.NumExplicitMethods(); i++ {
		m := typ.ExplicitMethod(i)
		sig := m.Type().(*types2.Signature)
		assert(sig.TParams() == nil)

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

func (w *writer) obj(obj types2.Object, explicits []types2.Type) {
	w.sync(syncObject)

	var implicitIdx map[*types2.TypeParam]int
	if isDefinedType(obj) && !isGlobal(obj) {
		implicitIdx = w.implicitIdx
	}
	w.reloc(relocObj, w.p.objIdx(obj, implicitIdx))

	w.len(len(explicits))
	for _, explicit := range explicits {
		w.typ(explicit)
	}
}

func (pw *pkgWriter) objIdx(obj types2.Object, implicitIdx map[*types2.TypeParam]int) int {
	if idx, ok := pw.globalsIdx[obj]; ok {
		return idx
	}

	w := pw.newWriter(relocObj, syncObject1)
	w.ext = pw.newWriter(relocObjExt, syncObject1)
	assert(w.ext.idx == w.idx)

	pw.globalsIdx[obj] = w.idx

	w.implicitIdx = implicitIdx
	w.ext.implicitIdx = implicitIdx

	w.doObj(obj)

	w.flush()
	w.ext.flush()

	return w.idx
}

func (w *writer) doObj(obj types2.Object) {
	// Ident goes first so importer can avoid unnecessary work if
	// they've already resolved this object.
	w.qualifiedIdent(obj)

	tparams := objTypeParams(obj)
	w.setTypeParams(tparams)
	w.typeParamBounds(tparams)

	if obj.Pkg() != w.p.curpkg {
		w.code(objStub)
		return
	}

	switch obj := obj.(type) {
	default:
		w.p.unexpected("object", obj)

	case *types2.Const:
		w.code(objConst)
		w.pos(obj)
		w.value(obj.Type(), obj.Val())

	case *types2.Func:
		decl, ok := w.p.funDecls[obj]
		assert(ok)
		sig := obj.Type().(*types2.Signature)

		// Rewrite blank methods into blank functions.
		// They aren't included in the receiver type's method set,
		// and we still want to write them out to be compiled
		// for regression tests.
		// TODO(mdempsky): Change regress tests to avoid relying
		// on blank functions/methods, so we can just ignore them
		// altogether.
		if recv := sig.Recv(); recv != nil {
			assert(obj.Name() == "_")
			assert(sig.TParams() == nil)

			params := make([]*types2.Var, 1+sig.Params().Len())
			params[0] = recv
			for i := 0; i < sig.Params().Len(); i++ {
				params[1+i] = sig.Params().At(i)
			}
			sig = types2.NewSignature(nil, types2.NewTuple(params...), sig.Results(), sig.Variadic())
		}

		w.code(objFunc)
		w.pos(obj)
		w.typeParamNames(sig.TParams())
		w.signature(sig)
		w.pos(decl)
		w.ext.funcExt(obj)

	case *types2.TypeName:
		decl, ok := w.p.typDecls[obj]
		assert(ok)

		if obj.IsAlias() {
			w.code(objAlias)
			w.pos(obj)
			w.typ(obj.Type())
			break
		}

		named := obj.Type().(*types2.Named)
		assert(named.TArgs() == nil)

		w.code(objType)
		w.pos(obj)
		w.typeParamNames(named.TParams())
		w.ext.typeExt(obj)
		w.typExpr(decl.Type)

		w.len(named.NumMethods())
		for i := 0; i < named.NumMethods(); i++ {
			w.method(named.Method(i))
		}

	case *types2.Var:
		w.code(objVar)
		w.pos(obj)
		w.typ(obj.Type())
		w.ext.varExt(obj)
	}
}

// typExpr writes the type represented by the given expression.
func (w *writer) typExpr(expr syntax.Expr) {
	tv, ok := w.p.info.Types[expr]
	assert(ok)
	assert(tv.IsType())
	w.typ(tv.Type)
}

func (w *writer) value(typ types2.Type, val constant.Value) {
	w.sync(syncValue)
	w.typ(typ)
	w.rawValue(val)
}

func (w *writer) setTypeParams(tparams []*types2.TypeName) {
	if len(tparams) == 0 {
		return
	}

	explicitIdx := make(map[*types2.TypeParam]int)
	for _, tparam := range tparams {
		explicitIdx[tparam.Type().(*types2.TypeParam)] = len(explicitIdx)
	}

	w.explicitIdx = explicitIdx
	w.ext.explicitIdx = explicitIdx
}

func (w *writer) typeParamBounds(tparams []*types2.TypeName) {
	w.sync(syncTypeParamBounds)

	// TODO(mdempsky): Remove. It's useful for debugging at the moment,
	// but it doesn't belong here.
	w.len(len(w.implicitIdx))
	w.len(len(w.explicitIdx))
	assert(len(w.explicitIdx) == len(tparams))

	for _, tparam := range tparams {
		w.typ(tparam.Type().(*types2.TypeParam).Bound())
	}
}

func (w *writer) typeParamNames(tparams []*types2.TypeName) {
	w.sync(syncTypeParamNames)

	for _, tparam := range tparams {
		w.pos(tparam)
		w.localIdent(tparam)
	}
}

func (w *writer) method(meth *types2.Func) {
	decl, ok := w.p.funDecls[meth]
	assert(ok)
	sig := meth.Type().(*types2.Signature)

	assert(len(w.explicitIdx) == len(sig.RParams()))
	w.setTypeParams(sig.RParams())

	w.sync(syncMethod)
	w.pos(meth)
	w.selector(meth)
	w.typeParamNames(sig.RParams())
	w.param(sig.Recv())
	w.signature(sig)

	w.pos(decl) // XXX: Hack to workaround linker limitations.
	w.ext.funcExt(meth)
}

// qualifiedIdent writes out the name of an object declared at package
// scope. (For now, it's also used to refer to local defined types.)
func (w *writer) qualifiedIdent(obj types2.Object) {
	w.sync(syncSym)

	name := obj.Name()
	if isDefinedType(obj) && !isGlobal(obj) {
		// TODO(mdempsky): Find a better solution, this is terrible.
		decl, ok := w.p.typDecls[obj.(*types2.TypeName)]
		assert(ok)
		name = fmt.Sprintf("%s·%v", name, decl.gen)
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

	w.sync(syncFuncExt)
	w.pragmaFlag(pragma)
	w.linkname(obj)
	w.bool(false) // stub extension
	w.addBody(obj.Type().(*types2.Signature), decl.Body, make(map[types2.Object]int))
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

func (w *writer) addBody(sig *types2.Signature, block *syntax.BlockStmt, localsIdx map[types2.Object]int) {
	// TODO(mdempsky): Theoretically, I think at this point we want to
	// extend the implicit type parameters list with any new explicit
	// type parameters.
	//
	// However, I believe that's moot: declared functions and methods
	// have explicit type parameters, but are always declared at package
	// scope (which has no implicit type parameters); and function
	// literals can appear within a type-parameterized function (i.e.,
	// implicit type parameters), but cannot have explicit type
	// parameters of their own.
	//
	// So I think it's safe to just use whichever is non-empty.
	implicitIdx := w.implicitIdx
	if len(implicitIdx) == 0 {
		implicitIdx = w.explicitIdx
	} else {
		assert(len(w.explicitIdx) == 0)
	}

	w.sync(syncAddBody)
	w.reloc(relocBody, w.p.bodyIdx(w.p.curpkg, sig, block, implicitIdx, localsIdx))
}

func (pw *pkgWriter) bodyIdx(pkg *types2.Package, sig *types2.Signature, block *syntax.BlockStmt, implicitIdx map[*types2.TypeParam]int, localsIdx map[types2.Object]int) int {
	w := pw.newWriter(relocBody, syncFuncBody)
	w.implicitIdx = implicitIdx
	w.localsIdx = localsIdx

	w.funcargs(sig)
	if w.bool(block != nil) {
		w.stmts(block.List)
		w.pos(block.Rbrace)
	}

	return w.flush()
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

func (w *writer) addLocal(obj types2.Object) {
	w.sync(syncAddLocal)
	idx := len(w.localsIdx)
	if debug {
		w.int(idx)
	}
	w.localsIdx[obj] = idx
}

func (w *writer) useLocal(obj types2.Object) {
	w.sync(syncUseObjLocal)
	idx, ok := w.localsIdx[obj]
	assert(ok)
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
			w.assignList(stmt.Lhs)
			w.exprList(stmt.Rhs)
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
				w.assignList(name)
				w.exprList(values[i])
			}
			break
		}

		w.code(stmtAssign)
		w.pos(decl)
		w.assignList(namesAsExpr(decl.NameList))
		w.exprList(decl.Values)
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
		w.assignList(rang.Lhs)
		w.expr(rang.X)
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
	w.expr(stmt.Tag)

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

	obj, targs := lookupObj(w.p.info, expr)

	if tv, ok := w.p.info.Types[expr]; ok {
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
			w.value(tv.Type, tv.Value)

			// TODO(mdempsky): These details are only important for backend
			// diagnostics. Explore writing them out separately.
			w.op(constExprOp(expr))
			w.string(syntax.String(expr))
			return
		}
	}

	if obj != nil {
		if _, ok := w.localsIdx[obj]; ok {
			assert(len(targs) == 0)
			w.code(exprLocal)
			w.useLocal(obj)
			return
		}

		w.code(exprName)
		w.obj(obj, targs)
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
		w.code(exprCall)

		if inf, ok := w.p.info.Inferred[expr]; ok {
			obj, _ := lookupObj(w.p.info, expr.Fun)
			assert(obj != nil)

			// As if w.expr(expr.Fun), but using inf.TArgs instead.
			w.code(exprName)
			w.obj(obj, inf.TArgs)
		} else {
			w.expr(expr.Fun)
		}

		w.pos(expr)
		w.exprs(expr.ArgList)
		w.bool(expr.HasDots)

	case *syntax.TypeSwitchGuard:
		w.code(exprTypeSwitchGuard)
		w.pos(expr)
		if tag := expr.Lhs; w.bool(tag != nil) {
			w.pos(tag)
			w.string(tag.Value)
		}
		w.expr(expr.X)
	}
}

func (w *writer) compLit(lit *syntax.CompositeLit) {
	tv, ok := w.p.info.Types[lit]
	assert(ok)

	w.sync(syncCompLit)
	w.pos(lit)
	w.typ(tv.Type)

	typ := tv.Type
	if ptr, ok := typ.Underlying().(*types2.Pointer); ok {
		typ = ptr.Elem()
	}
	str, isStruct := typ.Underlying().(*types2.Struct)

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

	w.sync(syncFuncLit)
	w.pos(expr)
	w.pos(expr.Type) // for QuirksMode
	w.signature(sig)

	closureVars, localsIdx := w.captureVars(expr)
	w.len(len(closureVars))
	for _, closureVar := range closureVars {
		w.pos(closureVar.pos)
		w.useLocal(closureVar.obj)
	}

	w.addBody(sig, expr.Body, localsIdx)
}

type posObj struct {
	pos syntax.Pos
	obj types2.Object
}

// captureVars returns the free variables used by the given function
// literal.
func (w *writer) captureVars(expr *syntax.FuncLit) (closureVars []posObj, localsIdx map[types2.Object]int) {
	scope, ok := w.p.info.Scopes[expr.Type]
	assert(ok)

	localsIdx = make(map[types2.Object]int)

	// TODO(mdempsky): This code needs to be cleaned up (e.g., to avoid
	// traversing nested function literals multiple times). This will be
	// easier after we drop quirks mode.

	var rbracePos syntax.Pos

	var visitor func(n syntax.Node) bool
	visitor = func(n syntax.Node) bool {

		// Constant expressions don't count towards capturing.
		if n, ok := n.(syntax.Expr); ok {
			if tv, ok := w.p.info.Types[n]; ok && tv.Value != nil {
				return true
			}
		}

		switch n := n.(type) {
		case *syntax.Name:
			if obj, ok := w.p.info.Uses[n].(*types2.Var); ok && !obj.IsField() && obj.Pkg() == w.p.curpkg && obj.Parent() != obj.Pkg().Scope() {
				// Found a local variable. See if it chains up to scope.
				parent := obj.Parent()
				for {
					if parent == scope {
						break
					}
					if parent == obj.Pkg().Scope() {
						if _, present := localsIdx[obj]; !present {
							pos := rbracePos
							if pos == (syntax.Pos{}) {
								pos = n.Pos()
							}

							idx := len(closureVars)
							closureVars = append(closureVars, posObj{pos, obj})
							localsIdx[obj] = idx
						}
						break
					}
					parent = parent.Parent()
				}
			}

		case *syntax.FuncLit:
			// Quirk: typecheck uses the rbrace position position of the
			// function literal as the position of the intermediary capture.
			if quirksMode() && rbracePos == (syntax.Pos{}) {
				rbracePos = n.Body.Rbrace
				syntax.Walk(n.Body, visitor)
				rbracePos = syntax.Pos{}
				return true
			}

		case *syntax.AssignStmt:
			// Quirk: typecheck visits (and thus captures) the RHS of
			// assignment statements before the LHS.
			if quirksMode() && (n.Op == 0 || n.Op == syntax.Def) {
				syntax.Walk(n.Rhs, visitor)
				syntax.Walk(n.Lhs, visitor)
				return true
			}
		case *syntax.RangeClause:
			// Quirk: Similarly, it visits the expression to be iterated
			// over before the iteration variables.
			if quirksMode() {
				syntax.Walk(n.X, visitor)
				if n.Lhs != nil {
					syntax.Walk(n.Lhs, visitor)
				}
				return true
			}
		}

		return false
	}
	syntax.Walk(expr.Body, visitor)

	return
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

// @@@ Package initialization

// Caution: This code is still clumsy, because toolstash -cmp is
// particularly sensitive to it.

type typeDeclGen struct {
	*syntax.TypeDecl
	gen int
}

func (pw *pkgWriter) collectDecls(noders []*noder) {
	var typegen int

	for _, p := range noders {
		var importedEmbed, importedUnsafe bool

		syntax.Walk(p.file, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.File:
				pw.checkPragmas(n.Pragma, ir.GoBuildPragma, false)

			case *syntax.ImportDecl:
				pw.checkPragmas(n.Pragma, 0, false)

				switch pkgNameOf(pw.info, n).Imported().Path() {
				case "embed":
					importedEmbed = true
				case "unsafe":
					importedUnsafe = true
				}

			case *syntax.ConstDecl:
				pw.checkPragmas(n.Pragma, 0, false)

			case *syntax.FuncDecl:
				pw.checkPragmas(n.Pragma, funcPragmas, false)

				obj := pw.info.Defs[n.Name].(*types2.Func)
				pw.funDecls[obj] = n

			case *syntax.TypeDecl:
				obj := pw.info.Defs[n.Name].(*types2.TypeName)
				d := typeDeclGen{TypeDecl: n}

				if n.Alias {
					pw.checkPragmas(n.Pragma, 0, false)
				} else {
					pw.checkPragmas(n.Pragma, typePragmas, false)

					// Assign a unique ID to function-scoped defined types.
					if !isGlobal(obj) {
						typegen++
						d.gen = typegen
					}
				}

				pw.typDecls[obj] = d

			case *syntax.VarDecl:
				pw.checkPragmas(n.Pragma, 0, true)

				if p, ok := n.Pragma.(*pragmas); ok && len(p.Embeds) > 0 {
					obj := pw.info.Defs[n.NameList[0]].(*types2.Var)
					// TODO(mdempsky): isGlobal(obj) gives false positive errors
					// for //go:embed directives on package-scope blank
					// variables.
					if err := checkEmbed(n, importedEmbed, !isGlobal(obj)); err != nil {
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
			}
			return false
		})

		pw.cgoPragmas = append(pw.cgoPragmas, p.pragcgobuf...)

		for _, l := range p.linknames {
			if !importedUnsafe {
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
		obj := w.p.info.Defs[decl.Name].(*types2.Func)
		sig := obj.Type().(*types2.Signature)

		if sig.RParams() != nil || sig.TParams() != nil {
			break // skip generic functions
		}

		if recv := sig.Recv(); recv != nil && obj.Name() != "_" {
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

		name := w.p.info.Defs[decl.Name].(*types2.TypeName)

		// Skip type declarations for interfaces that are only usable as
		// type parameter bounds.
		if iface, ok := name.Type().Underlying().(*types2.Interface); ok && iface.IsConstraint() {
			break
		}

		// Skip aliases to uninstantiated generic types.
		// TODO(mdempsky): Revisit after #46477 is resolved.
		if name.IsAlias() {
			named, ok := name.Type().(*types2.Named)
			if ok && len(named.TParams()) != 0 && len(named.TArgs()) == 0 {
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
// is an explicit instantiation of a generic object, then the type
// arguments are returned as well.
func lookupObj(info *types2.Info, expr syntax.Expr) (obj types2.Object, targs []types2.Type) {
	if index, ok := expr.(*syntax.IndexExpr); ok {
		if inf, ok := info.Inferred[index]; ok {
			targs = inf.TArgs
		} else {
			args := unpackListExpr(index.Index)

			if len(args) == 1 {
				tv, ok := info.Types[args[0]]
				assert(ok)
				if tv.IsValue() {
					return // normal index expression
				}
			}

			targs = make([]types2.Type, len(args))
			for i, arg := range args {
				tv, ok := info.Types[arg]
				assert(ok)
				assert(tv.IsType())
				targs[i] = tv.Type
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
		obj, _ = info.Uses[name]
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
func objTypeParams(obj types2.Object) []*types2.TypeName {
	switch obj := obj.(type) {
	case *types2.Func:
		return obj.Type().(*types2.Signature).TParams()
	case *types2.TypeName:
		if !obj.IsAlias() {
			return obj.Type().(*types2.Named).TParams()
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
