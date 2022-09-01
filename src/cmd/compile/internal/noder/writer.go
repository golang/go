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

// This file implements the Unified IR package writer and defines the
// Unified IR export data format.
//
// Low-level coding details (e.g., byte-encoding of individual
// primitive values, or handling element bitstreams and
// cross-references) are handled by internal/pkgbits, so here we only
// concern ourselves with higher-level worries like mapping Go
// language constructs into elements.

// There are two central types in the writing process: the "writer"
// type handles writing out individual elements, while the "pkgWriter"
// type keeps track of which elements have already been created.
//
// For each sort of "thing" (e.g., position, package, object, type)
// that can be written into the export data, there are generally
// several methods that work together:
//
// - writer.thing handles writing out a *use* of a thing, which often
//   means writing a relocation to that thing's encoded index.
//
// - pkgWriter.thingIdx handles reserving an index for a thing, and
//   writing out any elements needed for the thing.
//
// - writer.doThing handles writing out the *definition* of a thing,
//   which in general is a mix of low-level coding primitives (e.g.,
//   ints and strings) or uses of other things.
//
// A design goal of Unified IR is to have a single, canonical writer
// implementation, but multiple reader implementations each tailored
// to their respective needs. For example, within cmd/compile's own
// backend, inlining is implemented largely by just re-running the
// function body reading code.

// TODO(mdempsky): Add an importer for Unified IR to the x/tools repo,
// and better document the file format boundary between public and
// private data.

// A pkgWriter constructs Unified IR export data from the results of
// running the types2 type checker on a Go compilation unit.
type pkgWriter struct {
	pkgbits.PkgEncoder

	m      posMap
	curpkg *types2.Package
	info   *types2.Info

	// Indices for previously written syntax and types2 things.

	posBasesIdx map[*syntax.PosBase]pkgbits.Index
	pkgsIdx     map[*types2.Package]pkgbits.Index
	typsIdx     map[types2.Type]pkgbits.Index
	objsIdx     map[types2.Object]pkgbits.Index

	// Maps from types2.Objects back to their syntax.Decl.

	funDecls map[*types2.Func]*syntax.FuncDecl
	typDecls map[*types2.TypeName]typeDeclGen

	// linknames maps package-scope objects to their linker symbol name,
	// if specified by a //go:linkname directive.
	linknames map[types2.Object]string

	// cgoPragmas accumulates any //go:cgo_* pragmas that need to be
	// passed through to cmd/link.
	cgoPragmas [][]string
}

// newPkgWriter returns an initialized pkgWriter for the specified
// package.
func newPkgWriter(m posMap, pkg *types2.Package, info *types2.Info) *pkgWriter {
	return &pkgWriter{
		PkgEncoder: pkgbits.NewPkgEncoder(base.Debug.SyncFrames),

		m:      m,
		curpkg: pkg,
		info:   info,

		pkgsIdx: make(map[*types2.Package]pkgbits.Index),
		objsIdx: make(map[types2.Object]pkgbits.Index),
		typsIdx: make(map[types2.Type]pkgbits.Index),

		posBasesIdx: make(map[*syntax.PosBase]pkgbits.Index),

		funDecls: make(map[*types2.Func]*syntax.FuncDecl),
		typDecls: make(map[*types2.TypeName]typeDeclGen),

		linknames: make(map[types2.Object]string),
	}
}

// errorf reports a user error about thing p.
func (pw *pkgWriter) errorf(p poser, msg string, args ...interface{}) {
	base.ErrorfAt(pw.m.pos(p), msg, args...)
}

// fatalf reports an internal compiler error about thing p.
func (pw *pkgWriter) fatalf(p poser, msg string, args ...interface{}) {
	base.FatalfAt(pw.m.pos(p), msg, args...)
}

// unexpected reports a fatal error about a thing of unexpected
// dynamic type.
func (pw *pkgWriter) unexpected(what string, p poser) {
	pw.fatalf(p, "unexpected %s: %v (%T)", what, p, p)
}

// typeOf returns the Type of the given value expression.
func (pw *pkgWriter) typeOf(expr syntax.Expr) types2.Type {
	tv, ok := pw.info.Types[expr]
	if !ok {
		pw.fatalf(expr, "missing Types entry: %v", syntax.String(expr))
	}
	if !tv.IsValue() {
		pw.fatalf(expr, "expected value: %v", syntax.String(expr))
	}
	return tv.Type
}

// A writer provides APIs for writing out an individual element.
type writer struct {
	p *pkgWriter

	pkgbits.Encoder

	// sig holds the signature for the current function body, if any.
	sig *types2.Signature

	// TODO(mdempsky): We should be able to prune localsIdx whenever a
	// scope closes, and then maybe we can just use the same map for
	// storing the TypeParams too (as their TypeName instead).

	// localsIdx tracks any local variables declared within this
	// function body. It's unused for writing out non-body things.
	localsIdx map[*types2.Var]int

	// closureVars tracks any free variables that are referenced by this
	// function body. It's unused for writing out non-body things.
	closureVars    []posVar
	closureVarsIdx map[*types2.Var]int // index of previously seen free variables

	dict *writerDict

	// derived tracks whether the type being written out references any
	// type parameters. It's unused for writing non-type things.
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
	derivedIdx map[types2.Type]pkgbits.Index

	// These slices correspond to entries in the runtime dictionary.
	typeParamMethodExprs []writerMethodExprInfo
	subdicts             []objInfo
	rtypes               []typeInfo
	itabs                []itabInfo
}

type itabInfo struct {
	typ   typeInfo
	iface typeInfo
}

// typeParamIndex returns the index of the given type parameter within
// the dictionary. This may differ from typ.Index() when there are
// implicit type parameters due to defined types declared within a
// generic function or method.
func (dict *writerDict) typeParamIndex(typ *types2.TypeParam) int {
	for idx, implicit := range dict.implicits {
		if implicit.Type().(*types2.TypeParam) == typ {
			return idx
		}
	}

	return len(dict.implicits) + typ.Index()
}

// A derivedInfo represents a reference to an encoded generic Go type.
type derivedInfo struct {
	idx    pkgbits.Index
	needed bool // TODO(mdempsky): Remove.
}

// A typeInfo represents a reference to an encoded Go type.
//
// If derived is true, then the typeInfo represents a generic Go type
// that contains type parameters. In this case, idx is an index into
// the readerDict.derived{,Types} arrays.
//
// Otherwise, the typeInfo represents a non-generic Go type, and idx
// is an index into the reader.typs array instead.
type typeInfo struct {
	idx     pkgbits.Index
	derived bool
}

// An objInfo represents a reference to an encoded, instantiated (if
// applicable) Go object.
type objInfo struct {
	idx       pkgbits.Index // index for the generic function declaration
	explicits []typeInfo    // info for the type arguments
}

// A selectorInfo represents a reference to an encoded field or method
// name (i.e., objects that can only be accessed using selector
// expressions).
type selectorInfo struct {
	pkgIdx  pkgbits.Index
	nameIdx pkgbits.Index
}

// anyDerived reports whether any of info's explicit type arguments
// are derived types.
func (info objInfo) anyDerived() bool {
	for _, explicit := range info.explicits {
		if explicit.derived {
			return true
		}
	}
	return false
}

// equals reports whether info and other represent the same Go object
// (i.e., same base object and identical type arguments, if any).
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

type writerMethodExprInfo struct {
	typeParamIdx int
	methodInfo   selectorInfo
}

// typeParamMethodExprIdx returns the index where the given encoded
// method expression function pointer appears within this dictionary's
// type parameters method expressions section, adding it if necessary.
func (dict *writerDict) typeParamMethodExprIdx(typeParamIdx int, methodInfo selectorInfo) int {
	newInfo := writerMethodExprInfo{typeParamIdx, methodInfo}

	for idx, oldInfo := range dict.typeParamMethodExprs {
		if oldInfo == newInfo {
			return idx
		}
	}

	idx := len(dict.typeParamMethodExprs)
	dict.typeParamMethodExprs = append(dict.typeParamMethodExprs, newInfo)
	return idx
}

// subdictIdx returns the index where the given encoded object's
// runtime dictionary appears within this dictionary's subdictionary
// section, adding it if necessary.
func (dict *writerDict) subdictIdx(newInfo objInfo) int {
	for idx, oldInfo := range dict.subdicts {
		if oldInfo.equals(newInfo) {
			return idx
		}
	}

	idx := len(dict.subdicts)
	dict.subdicts = append(dict.subdicts, newInfo)
	return idx
}

// rtypeIdx returns the index where the given encoded type's
// *runtime._type value appears within this dictionary's rtypes
// section, adding it if necessary.
func (dict *writerDict) rtypeIdx(newInfo typeInfo) int {
	for idx, oldInfo := range dict.rtypes {
		if oldInfo == newInfo {
			return idx
		}
	}

	idx := len(dict.rtypes)
	dict.rtypes = append(dict.rtypes, newInfo)
	return idx
}

// itabIdx returns the index where the given encoded type pair's
// *runtime.itab value appears within this dictionary's itabs section,
// adding it if necessary.
func (dict *writerDict) itabIdx(typInfo, ifaceInfo typeInfo) int {
	newInfo := itabInfo{typInfo, ifaceInfo}

	for idx, oldInfo := range dict.itabs {
		if oldInfo == newInfo {
			return idx
		}
	}

	idx := len(dict.itabs)
	dict.itabs = append(dict.itabs, newInfo)
	return idx
}

func (pw *pkgWriter) newWriter(k pkgbits.RelocKind, marker pkgbits.SyncMarker) *writer {
	return &writer{
		Encoder: pw.NewEncoder(k, marker),
		p:       pw,
	}
}

// @@@ Positions

// pos writes the position of p into the element bitstream.
func (w *writer) pos(p poser) {
	w.Sync(pkgbits.SyncPos)
	pos := p.Pos()

	// TODO(mdempsky): Track down the remaining cases here and fix them.
	if !w.Bool(pos.IsKnown()) {
		return
	}

	// TODO(mdempsky): Delta encoding.
	w.posBase(pos.Base())
	w.Uint(pos.Line())
	w.Uint(pos.Col())
}

// posBase writes a reference to the given PosBase into the element
// bitstream.
func (w *writer) posBase(b *syntax.PosBase) {
	w.Reloc(pkgbits.RelocPosBase, w.p.posBaseIdx(b))
}

// posBaseIdx returns the index for the given PosBase.
func (pw *pkgWriter) posBaseIdx(b *syntax.PosBase) pkgbits.Index {
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

// pkg writes a use of the given Package into the element bitstream.
func (w *writer) pkg(pkg *types2.Package) {
	w.pkgRef(w.p.pkgIdx(pkg))
}

func (w *writer) pkgRef(idx pkgbits.Index) {
	w.Sync(pkgbits.SyncPkg)
	w.Reloc(pkgbits.RelocPkg, idx)
}

// pkgIdx returns the index for the given package, adding it to the
// package export data if needed.
func (pw *pkgWriter) pkgIdx(pkg *types2.Package) pkgbits.Index {
	if idx, ok := pw.pkgsIdx[pkg]; ok {
		return idx
	}

	w := pw.newWriter(pkgbits.RelocPkg, pkgbits.SyncPkgDef)
	pw.pkgsIdx[pkg] = w.Idx

	// The universe and package unsafe need to be handled specially by
	// importers anyway, so we serialize them using just their package
	// path. This ensures that readers don't confuse them for
	// user-defined packages.
	switch pkg {
	case nil: // universe
		w.String("builtin") // same package path used by godoc
	case types2.Unsafe:
		w.String("unsafe")
	default:
		// TODO(mdempsky): Write out pkg.Path() for curpkg too.
		var path string
		if pkg != w.p.curpkg {
			path = pkg.Path()
		}
		base.Assertf(path != "builtin" && path != "unsafe", "unexpected path for user-defined package: %q", path)
		w.String(path)
		w.String(pkg.Name())

		w.Len(len(pkg.Imports()))
		for _, imp := range pkg.Imports() {
			w.pkg(imp)
		}
	}

	return w.Flush()
}

// @@@ Types

var (
	anyTypeName  = types2.Universe.Lookup("any").(*types2.TypeName)
	runeTypeName = types2.Universe.Lookup("rune").(*types2.TypeName)
)

// typ writes a use of the given type into the bitstream.
func (w *writer) typ(typ types2.Type) {
	w.typInfo(w.p.typIdx(typ, w.dict))
}

// typInfo writes a use of the given type (specified as a typeInfo
// instead) into the bitstream.
func (w *writer) typInfo(info typeInfo) {
	w.Sync(pkgbits.SyncType)
	if w.Bool(info.derived) {
		w.Len(int(info.idx))
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
		obj, targs := splitNamed(typ)

		// Defined types that are declared within a generic function (and
		// thus have implicit type parameters) are always derived types.
		if w.p.hasImplicitTypeParams(obj) {
			w.derived = true
		}

		w.Code(pkgbits.TypeNamed)
		w.obj(obj, targs)

	case *types2.TypeParam:
		w.derived = true
		w.Code(pkgbits.TypeTypeParam)
		w.Len(w.dict.typeParamIndex(typ))

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
		idx := pkgbits.Index(len(dict.derived))
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

// obj writes a use of the given object into the bitstream.
//
// If obj is a generic object, then explicits are the explicit type
// arguments used to instantiate it (i.e., used to substitute the
// object's own declared type parameters).
func (w *writer) obj(obj types2.Object, explicits *types2.TypeList) {
	w.objInfo(w.p.objInstIdx(obj, explicits, w.dict))
}

// objInfo writes a use of the given encoded object into the
// bitstream.
func (w *writer) objInfo(info objInfo) {
	w.Sync(pkgbits.SyncObject)
	w.Bool(false) // TODO(mdempsky): Remove; was derived func inst.
	w.Reloc(pkgbits.RelocObj, info.idx)

	w.Len(len(info.explicits))
	for _, info := range info.explicits {
		w.typInfo(info)
	}
}

// objInstIdx returns the indices for an object and a corresponding
// list of type arguments used to instantiate it, adding them to the
// export data as needed.
func (pw *pkgWriter) objInstIdx(obj types2.Object, explicits *types2.TypeList, dict *writerDict) objInfo {
	explicitInfos := make([]typeInfo, explicits.Len())
	for i := range explicitInfos {
		explicitInfos[i] = pw.typIdx(explicits.At(i), dict)
	}
	return objInfo{idx: pw.objIdx(obj), explicits: explicitInfos}
}

// objIdx returns the index for the given Object, adding it to the
// export data as needed.
func (pw *pkgWriter) objIdx(obj types2.Object) pkgbits.Index {
	// TODO(mdempsky): Validate that obj is a global object (or a local
	// defined type, which we hoist to global scope anyway).

	if idx, ok := pw.objsIdx[obj]; ok {
		return idx
	}

	dict := &writerDict{
		derivedIdx: make(map[types2.Type]pkgbits.Index),
	}

	if isDefinedType(obj) && obj.Pkg() == pw.curpkg {
		decl, ok := pw.typDecls[obj.(*types2.TypeName)]
		assert(ok)
		dict.implicits = decl.implicits
	}

	// We encode objects into 4 elements across different sections, all
	// sharing the same index:
	//
	// - RelocName has just the object's qualified name (i.e.,
	//   Object.Pkg and Object.Name) and the CodeObj indicating what
	//   specific type of Object it is (Var, Func, etc).
	//
	// - RelocObj has the remaining public details about the object,
	//   relevant to go/types importers.
	//
	// - RelocObjExt has additional private details about the object,
	//   which are only relevant to cmd/compile itself. This is
	//   separated from RelocObj so that go/types importers are
	//   unaffected by internal compiler changes.
	//
	// - RelocObjDict has public details about the object's type
	//   parameters and derived type's used by the object. This is
	//   separated to facilitate the eventual introduction of
	//   shape-based stenciling.
	//
	// TODO(mdempsky): Re-evaluate whether RelocName still makes sense
	// to keep separate from RelocObj.

	w := pw.newWriter(pkgbits.RelocObj, pkgbits.SyncObject1)
	wext := pw.newWriter(pkgbits.RelocObjExt, pkgbits.SyncObject1)
	wname := pw.newWriter(pkgbits.RelocName, pkgbits.SyncObject1)
	wdict := pw.newWriter(pkgbits.RelocObjDict, pkgbits.SyncObject1)

	pw.objsIdx[obj] = w.Idx // break cycles
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

// doObj writes the RelocObj definition for obj to w, and the
// RelocObjExt definition to wext.
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
//
// TODO(mdempsky): Document how this differs from exprType.
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

	// Write runtime dictionary information.
	//
	// N.B., the go/types importer reads up to the section, but doesn't
	// read any further, so it's safe to change. (See TODO above.)

	// For each type parameter, write out whether the constraint is a
	// basic interface. This is used to determine how aggressively we
	// can shape corresponding type arguments.
	//
	// This is somewhat redundant with writing out the full type
	// parameter constraints above, but the compiler currently skips
	// over those. Also, we don't care about the *declared* constraints,
	// but how the type parameters are actually *used*. E.g., if a type
	// parameter is constrained to `int | uint` but then never used in
	// arithmetic/conversions/etc, we could shape those together.
	for _, implicit := range dict.implicits {
		tparam := implicit.Type().(*types2.TypeParam)
		w.Bool(tparam.Underlying().(*types2.Interface).IsMethodSet())
	}
	for i := 0; i < ntparams; i++ {
		tparam := tparams.At(i)
		w.Bool(tparam.Underlying().(*types2.Interface).IsMethodSet())
	}

	w.Len(len(dict.typeParamMethodExprs))
	for _, info := range dict.typeParamMethodExprs {
		w.Len(info.typeParamIdx)
		w.selectorInfo(info.methodInfo)
	}

	w.Len(len(dict.subdicts))
	for _, info := range dict.subdicts {
		w.objInfo(info)
	}

	w.Len(len(dict.rtypes))
	for _, info := range dict.rtypes {
		w.typInfo(info)
	}

	w.Len(len(dict.itabs))
	for _, info := range dict.itabs {
		w.typInfo(info.typ)
		w.typInfo(info.iface)
	}

	assert(len(dict.derived) == nderived)
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
			// For local defined types, we embed a scope-disambiguation
			// number directly into their name. types.SplitVargenSuffix then
			// knows to look for this.
			//
			// TODO(mdempsky): Find a better solution; this is terrible.
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
// objects that can only be accessed by non-qualified name, within the
// context of a particular function).
func (w *writer) localIdent(obj types2.Object) {
	assert(!isGlobal(obj))
	w.Sync(pkgbits.SyncLocalIdent)
	w.pkg(obj.Pkg())
	w.String(obj.Name())
}

// selector writes the name of a field or method (i.e., objects that
// can only be accessed using selector expressions).
func (w *writer) selector(obj types2.Object) {
	w.selectorInfo(w.p.selectorIdx(obj))
}

func (w *writer) selectorInfo(info selectorInfo) {
	w.Sync(pkgbits.SyncSelector)
	w.pkgRef(info.pkgIdx)
	w.StringRef(info.nameIdx)
}

func (pw *pkgWriter) selectorIdx(obj types2.Object) selectorInfo {
	pkgIdx := pw.pkgIdx(obj.Pkg())
	nameIdx := pw.StringIdx(obj.Name())
	return selectorInfo{pkgIdx: pkgIdx, nameIdx: nameIdx}
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
		if (pragma&ir.UintptrKeepAlive != 0 && pragma&ir.UintptrEscapes == 0) && pragma&ir.Nosplit == 0 {
			// Stack growth can't handle uintptr arguments that may
			// be pointers (as we don't know which are pointers
			// when creating the stack map). Thus uintptrkeepalive
			// functions (and all transitive callees) must be
			// nosplit.
			//
			// N.B. uintptrescapes implies uintptrkeepalive but it
			// is OK since the arguments must escape to the heap.
			//
			// TODO(prattmic): Add recursive nosplit check of callees.
			// TODO(prattmic): Functions with no body (i.e.,
			// assembly) must also be nosplit, but we can't check
			// that here.
			w.p.errorf(decl, "go:uintptrkeepalive requires go:nosplit")
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
	body, closureVars := w.p.bodyIdx(sig, block, w.dict)
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

// bodyIdx returns the index for the given function body (specified by
// block), adding it to the export data
func (pw *pkgWriter) bodyIdx(sig *types2.Signature, block *syntax.BlockStmt, dict *writerDict) (idx pkgbits.Index, closureVars []posVar) {
	w := pw.newWriter(pkgbits.RelocBody, pkgbits.SyncFuncBody)
	w.sig = sig
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

// addLocal records the declaration of a new local variable.
func (w *writer) addLocal(obj *types2.Var) {
	idx := len(w.localsIdx)

	w.Sync(pkgbits.SyncAddLocal)
	if w.p.SyncMarkers() {
		w.Int(idx)
	}
	w.varDictIndex(obj)

	if w.localsIdx == nil {
		w.localsIdx = make(map[*types2.Var]int)
	}
	w.localsIdx[obj] = idx
}

// useLocal writes a reference to the given local or free variable
// into the bitstream.
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
		w.closureVars = append(w.closureVars, posVar{pos, obj})
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

// stmt writes the given statement into the function body bitstream.
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

			var typ types2.Type
			if stmt.Op != syntax.Shl && stmt.Op != syntax.Shr {
				typ = w.p.typeOf(stmt.Lhs)
			}
			w.implicitConvExpr(typ, stmt.Rhs)

		default:
			w.assignStmt(stmt, stmt.Lhs, stmt.Rhs)
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

		resultTypes := w.sig.Results()
		dstType := func(i int) types2.Type {
			return resultTypes.At(i).Type()
		}
		w.multiExpr(stmt, dstType, unpackListExpr(stmt.Results))

	case *syntax.SelectStmt:
		w.Code(stmtSelect)
		w.selectStmt(stmt)

	case *syntax.SendStmt:
		chanType := types2.CoreType(w.p.typeOf(stmt.Chan)).(*types2.Chan)

		w.Code(stmtSend)
		w.pos(stmt)
		w.expr(stmt.Chan)
		w.implicitConvExpr(chanType.Elem(), stmt.Value)

	case *syntax.SwitchStmt:
		w.Code(stmtSwitch)
		w.switchStmt(stmt)
	}
}

func (w *writer) assignList(expr syntax.Expr) {
	exprs := unpackListExpr(expr)
	w.Len(len(exprs))

	for _, expr := range exprs {
		w.assign(expr)
	}
}

func (w *writer) assign(expr syntax.Expr) {
	expr = unparen(expr)

	if name, ok := expr.(*syntax.Name); ok {
		if name.Value == "_" {
			w.Code(assignBlank)
			return
		}

		if obj, ok := w.p.info.Defs[name]; ok {
			obj := obj.(*types2.Var)

			w.Code(assignDef)
			w.pos(obj)
			w.localIdent(obj)
			w.typ(obj.Type())

			// TODO(mdempsky): Minimize locals index size by deferring
			// this until the variables actually come into scope.
			w.addLocal(obj)
			return
		}
	}

	w.Code(assignExpr)
	w.expr(expr)
}

func (w *writer) declStmt(decl syntax.Decl) {
	switch decl := decl.(type) {
	default:
		w.p.unexpected("declaration", decl)

	case *syntax.ConstDecl, *syntax.TypeDecl:

	case *syntax.VarDecl:
		w.assignStmt(decl, namesAsExpr(decl.NameList), decl.Values)
	}
}

// assignStmt writes out an assignment for "lhs = rhs".
func (w *writer) assignStmt(pos poser, lhs0, rhs0 syntax.Expr) {
	lhs := unpackListExpr(lhs0)
	rhs := unpackListExpr(rhs0)

	w.Code(stmtAssign)
	w.pos(pos)

	// As if w.assignList(lhs0).
	w.Len(len(lhs))
	for _, expr := range lhs {
		w.assign(expr)
	}

	dstType := func(i int) types2.Type {
		dst := lhs[i]

		// Finding dstType is somewhat involved, because for VarDecl
		// statements, the Names are only added to the info.{Defs,Uses}
		// maps, not to info.Types.
		if name, ok := unparen(dst).(*syntax.Name); ok {
			if name.Value == "_" {
				return nil // ok: no implicit conversion
			} else if def, ok := w.p.info.Defs[name].(*types2.Var); ok {
				return def.Type()
			} else if use, ok := w.p.info.Uses[name].(*types2.Var); ok {
				return use.Type()
			} else {
				w.p.fatalf(dst, "cannot find type of destination object: %v", dst)
			}
		}

		return w.p.typeOf(dst)
	}

	w.multiExpr(pos, dstType, rhs)
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
		w.assignList(rang.Lhs)
		w.expr(rang.X)

		xtyp := w.p.typeOf(rang.X)
		if _, isMap := types2.CoreType(xtyp).(*types2.Map); isMap {
			w.rtype(xtyp)
		}
		{
			lhs := unpackListExpr(rang.Lhs)
			assign := func(i int, src types2.Type) {
				if i >= len(lhs) {
					return
				}
				dst := unparen(lhs[i])
				if name, ok := dst.(*syntax.Name); ok && name.Value == "_" {
					return
				}

				var dstType types2.Type
				if rang.Def {
					// For `:=` assignments, the LHS names only appear in Defs,
					// not Types (as used by typeOf).
					dstType = w.p.info.Defs[dst.(*syntax.Name)].(*types2.Var).Type()
				} else {
					dstType = w.p.typeOf(dst)
				}

				w.convRTTI(src, dstType)
			}

			keyType, valueType := w.p.rangeTypes(rang.X)
			assign(0, keyType)
			assign(1, valueType)
		}

	} else {
		w.pos(stmt)
		w.stmt(stmt.Init)
		w.optExpr(stmt.Cond)
		w.stmt(stmt.Post)
	}

	w.blockStmt(stmt.Body)
	w.closeAnotherScope()
}

// rangeTypes returns the types of values produced by ranging over
// expr.
func (pw *pkgWriter) rangeTypes(expr syntax.Expr) (key, value types2.Type) {
	typ := pw.typeOf(expr)
	switch typ := types2.CoreType(typ).(type) {
	case *types2.Pointer: // must be pointer to array
		return types2.Typ[types2.Int], types2.CoreType(typ.Elem()).(*types2.Array).Elem()
	case *types2.Array:
		return types2.Typ[types2.Int], typ.Elem()
	case *types2.Slice:
		return types2.Typ[types2.Int], typ.Elem()
	case *types2.Basic:
		if typ.Info()&types2.IsString != 0 {
			return types2.Typ[types2.Int], runeTypeName.Type()
		}
	case *types2.Map:
		return typ.Key(), typ.Elem()
	case *types2.Chan:
		return typ.Elem(), nil
	}
	pw.fatalf(expr, "unexpected range type: %v", typ)
	panic("unreachable")
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

	var iface, tagType types2.Type
	if guard, ok := stmt.Tag.(*syntax.TypeSwitchGuard); w.Bool(ok) {
		iface = w.p.typeOf(guard.X)

		w.pos(guard)
		if tag := guard.Lhs; w.Bool(tag != nil) {
			w.pos(tag)
			w.String(tag.Value)
		}
		w.expr(guard.X)
	} else {
		tag := stmt.Tag

		if tag != nil {
			tagType = w.p.typeOf(tag)
		} else {
			tagType = types2.Typ[types2.Bool]
		}

		// Walk is going to emit comparisons between the tag value and
		// each case expression, and we want these comparisons to always
		// have the same type. If there are any case values that can't be
		// converted to the tag value's type, then convert everything to
		// `any` instead.
	Outer:
		for _, clause := range stmt.Body {
			for _, cas := range unpackListExpr(clause.Cases) {
				if casType := w.p.typeOf(cas); !types2.AssignableTo(casType, tagType) {
					tagType = types2.NewInterfaceType(nil, nil)
					break Outer
				}
			}
		}

		if w.Bool(tag != nil) {
			w.implicitConvExpr(tagType, tag)
		}
	}

	w.Len(len(stmt.Body))
	for i, clause := range stmt.Body {
		if i > 0 {
			w.closeScope(clause.Pos())
		}
		w.openScope(clause.Pos())

		w.pos(clause)

		cases := unpackListExpr(clause.Cases)
		if iface != nil {
			w.Len(len(cases))
			for _, cas := range cases {
				if w.Bool(isNil(w.p.info, cas)) {
					continue
				}
				w.exprType(iface, cas)
			}
		} else {
			// As if w.exprList(clause.Cases),
			// but with implicit conversions to tagType.

			w.Sync(pkgbits.SyncExprList)
			w.Sync(pkgbits.SyncExprs)
			w.Len(len(cases))
			for _, cas := range cases {
				w.implicitConvExpr(tagType, cas)
			}
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

// expr writes the given expression into the function body bitstream.
func (w *writer) expr(expr syntax.Expr) {
	base.Assertf(expr != nil, "missing expression")

	expr = unparen(expr) // skip parens; unneeded after typecheck

	obj, inst := lookupObj(w.p.info, expr)
	targs := inst.TypeArgs

	if tv, ok := w.p.info.Types[expr]; ok {
		if tv.IsType() {
			w.p.fatalf(expr, "unexpected type expression %v", syntax.String(expr))
		}

		if tv.Value != nil {
			w.Code(exprConst)
			w.pos(expr)
			typ := idealType(tv)
			assert(typ != nil)
			w.typ(typ)
			w.Value(tv.Value)

			// TODO(mdempsky): These details are only important for backend
			// diagnostics. Explore writing them out separately.
			w.op(constExprOp(expr))
			w.String(syntax.String(expr))
			return
		}

		if _, isNil := obj.(*types2.Nil); isNil {
			w.Code(exprNil)
			w.pos(expr)
			w.typ(tv.Type)
			return
		}

		// With shape types (and particular pointer shaping), we may have
		// an expression of type "go.shape.*uint8", but need to reshape it
		// to another shape-identical type to allow use in field
		// selection, indexing, etc.
		if typ := tv.Type; !tv.IsBuiltin() && !isTuple(typ) && !isUntyped(typ) {
			w.Code(exprReshape)
			w.typ(typ)
			// fallthrough
		}
	}

	if obj != nil {
		if targs.Len() != 0 {
			obj := obj.(*types2.Func)

			w.Code(exprFuncInst)
			w.pos(expr)
			w.funcInst(obj, targs)
			return
		}

		if isGlobal(obj) {
			w.Code(exprGlobal)
			w.obj(obj, nil)
			return
		}

		obj := obj.(*types2.Var)
		assert(!obj.IsField())

		w.Code(exprLocal)
		w.useLocal(expr.Pos(), obj)
		return
	}

	switch expr := expr.(type) {
	default:
		w.p.unexpected("expression", expr)

	case *syntax.CompositeLit:
		w.Code(exprCompLit)
		w.compLit(expr)

	case *syntax.FuncLit:
		w.Code(exprFuncLit)
		w.funcLit(expr)

	case *syntax.SelectorExpr:
		sel, ok := w.p.info.Selections[expr]
		assert(ok)

		switch sel.Kind() {
		default:
			w.p.fatalf(expr, "unexpected selection kind: %v", sel.Kind())

		case types2.FieldVal:
			w.Code(exprFieldVal)
			w.expr(expr.X)
			w.pos(expr)
			w.selector(sel.Obj())

		case types2.MethodVal:
			w.Code(exprMethodVal)
			typ := w.recvExpr(expr, sel)
			w.pos(expr)
			w.methodExpr(expr, typ, sel)

		case types2.MethodExpr:
			w.Code(exprMethodExpr)

			tv, ok := w.p.info.Types[expr.X]
			assert(ok)
			assert(tv.IsType())

			index := sel.Index()
			implicits := index[:len(index)-1]

			typ := tv.Type
			w.typ(typ)

			w.Len(len(implicits))
			for _, ix := range implicits {
				w.Len(ix)
				typ = deref2(typ).Underlying().(*types2.Struct).Field(ix).Type()
			}

			recv := sel.Obj().(*types2.Func).Type().(*types2.Signature).Recv().Type()
			if w.Bool(isPtrTo(typ, recv)) { // need deref
				typ = recv
			} else if w.Bool(isPtrTo(recv, typ)) { // need addr
				typ = recv
			}

			w.pos(expr)
			w.methodExpr(expr, typ, sel)
		}

	case *syntax.IndexExpr:
		_ = w.p.typeOf(expr.Index) // ensure this is an index expression, not an instantiation

		xtyp := w.p.typeOf(expr.X)

		var keyType types2.Type
		if mapType, ok := types2.CoreType(xtyp).(*types2.Map); ok {
			keyType = mapType.Key()
		}

		w.Code(exprIndex)
		w.expr(expr.X)
		w.pos(expr)
		w.implicitConvExpr(keyType, expr.Index)
		if keyType != nil {
			w.rtype(xtyp)
		}

	case *syntax.SliceExpr:
		w.Code(exprSlice)
		w.expr(expr.X)
		w.pos(expr)
		for _, n := range &expr.Index {
			w.optExpr(n)
		}

	case *syntax.AssertExpr:
		iface := w.p.typeOf(expr.X)

		w.Code(exprAssert)
		w.expr(expr.X)
		w.pos(expr)
		w.exprType(iface, expr.Type)
		w.rtype(iface)

	case *syntax.Operation:
		if expr.Y == nil {
			w.Code(exprUnaryOp)
			w.op(unOps[expr.Op])
			w.pos(expr)
			w.expr(expr.X)
			break
		}

		var commonType types2.Type
		switch expr.Op {
		case syntax.Shl, syntax.Shr:
			// ok: operands are allowed to have different types
		default:
			xtyp := w.p.typeOf(expr.X)
			ytyp := w.p.typeOf(expr.Y)
			switch {
			case types2.AssignableTo(xtyp, ytyp):
				commonType = ytyp
			case types2.AssignableTo(ytyp, xtyp):
				commonType = xtyp
			default:
				w.p.fatalf(expr, "failed to find common type between %v and %v", xtyp, ytyp)
			}
		}

		w.Code(exprBinaryOp)
		w.op(binOps[expr.Op])
		w.implicitConvExpr(commonType, expr.X)
		w.pos(expr)
		w.implicitConvExpr(commonType, expr.Y)

	case *syntax.CallExpr:
		tv, ok := w.p.info.Types[expr.Fun]
		assert(ok)
		if tv.IsType() {
			assert(len(expr.ArgList) == 1)
			assert(!expr.HasDots)
			w.convertExpr(tv.Type, expr.ArgList[0], false)
			break
		}

		var rtype types2.Type
		if tv.IsBuiltin() {
			switch obj, _ := lookupObj(w.p.info, expr.Fun); obj.Name() {
			case "make":
				assert(len(expr.ArgList) >= 1)
				assert(!expr.HasDots)

				w.Code(exprMake)
				w.pos(expr)
				w.exprType(nil, expr.ArgList[0])
				w.exprs(expr.ArgList[1:])

				typ := w.p.typeOf(expr)
				switch coreType := types2.CoreType(typ).(type) {
				default:
					w.p.fatalf(expr, "unexpected core type: %v", coreType)
				case *types2.Chan:
					w.rtype(typ)
				case *types2.Map:
					w.rtype(typ)
				case *types2.Slice:
					w.rtype(sliceElem(typ))
				}

				return

			case "new":
				assert(len(expr.ArgList) == 1)
				assert(!expr.HasDots)

				w.Code(exprNew)
				w.pos(expr)
				w.exprType(nil, expr.ArgList[0])
				return

			case "append":
				rtype = sliceElem(w.p.typeOf(expr))
			case "copy":
				typ := w.p.typeOf(expr.ArgList[0])
				if tuple, ok := typ.(*types2.Tuple); ok { // "copy(g())"
					typ = tuple.At(0).Type()
				}
				rtype = sliceElem(typ)
			case "delete":
				typ := w.p.typeOf(expr.ArgList[0])
				if tuple, ok := typ.(*types2.Tuple); ok { // "delete(g())"
					typ = tuple.At(0).Type()
				}
				rtype = typ
			case "Slice":
				rtype = sliceElem(w.p.typeOf(expr))
			}
		}

		writeFunExpr := func() {
			fun := unparen(expr.Fun)

			if selector, ok := fun.(*syntax.SelectorExpr); ok {
				if sel, ok := w.p.info.Selections[selector]; ok && sel.Kind() == types2.MethodVal {
					w.Bool(true) // method call
					typ := w.recvExpr(selector, sel)
					w.methodExpr(selector, typ, sel)
					return
				}
			}

			w.Bool(false) // not a method call (i.e., normal function call)

			if obj, inst := lookupObj(w.p.info, fun); w.Bool(obj != nil && inst.TypeArgs.Len() != 0) {
				obj := obj.(*types2.Func)

				w.pos(fun)
				w.funcInst(obj, inst.TypeArgs)
				return
			}

			w.expr(fun)
		}

		sigType := types2.CoreType(tv.Type).(*types2.Signature)
		paramTypes := sigType.Params()

		w.Code(exprCall)
		writeFunExpr()
		w.pos(expr)

		paramType := func(i int) types2.Type {
			if sigType.Variadic() && !expr.HasDots && i >= paramTypes.Len()-1 {
				return paramTypes.At(paramTypes.Len() - 1).Type().(*types2.Slice).Elem()
			}
			return paramTypes.At(i).Type()
		}

		w.multiExpr(expr, paramType, expr.ArgList)
		w.Bool(expr.HasDots)
		if rtype != nil {
			w.rtype(rtype)
		}
	}
}

func sliceElem(typ types2.Type) types2.Type {
	return types2.CoreType(typ).(*types2.Slice).Elem()
}

func (w *writer) optExpr(expr syntax.Expr) {
	if w.Bool(expr != nil) {
		w.expr(expr)
	}
}

// recvExpr writes out expr.X, but handles any implicit addressing,
// dereferencing, and field selections appropriate for the method
// selection.
func (w *writer) recvExpr(expr *syntax.SelectorExpr, sel *types2.Selection) types2.Type {
	index := sel.Index()
	implicits := index[:len(index)-1]

	w.Code(exprRecv)
	w.expr(expr.X)
	w.pos(expr)
	w.Len(len(implicits))

	typ := w.p.typeOf(expr.X)
	for _, ix := range implicits {
		typ = deref2(typ).Underlying().(*types2.Struct).Field(ix).Type()
		w.Len(ix)
	}

	recv := sel.Obj().(*types2.Func).Type().(*types2.Signature).Recv().Type()
	if w.Bool(isPtrTo(typ, recv)) { // needs deref
		typ = recv
	} else if w.Bool(isPtrTo(recv, typ)) { // needs addr
		typ = recv
	}

	return typ
}

// funcInst writes a reference to an instantiated function.
func (w *writer) funcInst(obj *types2.Func, targs *types2.TypeList) {
	info := w.p.objInstIdx(obj, targs, w.dict)

	// Type arguments list contains derived types; we can emit a static
	// call to the shaped function, but need to dynamically compute the
	// runtime dictionary pointer.
	if w.Bool(info.anyDerived()) {
		w.Len(w.dict.subdictIdx(info))
		return
	}

	// Type arguments list is statically known; we can emit a static
	// call with a statically reference to the respective runtime
	// dictionary.
	w.objInfo(info)
}

// methodExpr writes out a reference to the method selected by
// expr. sel should be the corresponding types2.Selection, and recv
// the type produced after any implicit addressing, dereferencing, and
// field selection. (Note: recv might differ from sel.Obj()'s receiver
// parameter in the case of interface types, and is needed for
// handling type parameter methods.)
func (w *writer) methodExpr(expr *syntax.SelectorExpr, recv types2.Type, sel *types2.Selection) {
	fun := sel.Obj().(*types2.Func)
	sig := fun.Type().(*types2.Signature)

	w.typ(recv)
	w.typ(sig)
	w.pos(expr)
	w.selector(fun)

	// Method on a type parameter. These require an indirect call
	// through the current function's runtime dictionary.
	if typeParam, ok := recv.(*types2.TypeParam); w.Bool(ok) {
		typeParamIdx := w.dict.typeParamIndex(typeParam)
		methodInfo := w.p.selectorIdx(fun)

		w.Len(w.dict.typeParamMethodExprIdx(typeParamIdx, methodInfo))
		return
	}

	if isInterface(recv) != isInterface(sig.Recv().Type()) {
		w.p.fatalf(expr, "isInterface inconsistency: %v and %v", recv, sig.Recv().Type())
	}

	if !isInterface(recv) {
		if named, ok := deref2(recv).(*types2.Named); ok {
			obj, targs := splitNamed(named)
			info := w.p.objInstIdx(obj, targs, w.dict)

			// Method on a derived receiver type. These can be handled by a
			// static call to the shaped method, but require dynamically
			// looking up the appropriate dictionary argument in the current
			// function's runtime dictionary.
			if w.p.hasImplicitTypeParams(obj) || info.anyDerived() {
				w.Bool(true) // dynamic subdictionary
				w.Len(w.dict.subdictIdx(info))
				return
			}

			// Method on a fully known receiver type. These can be handled
			// by a static call to the shaped method, and with a static
			// reference to the receiver type's dictionary.
			if targs.Len() != 0 {
				w.Bool(false) // no dynamic subdictionary
				w.Bool(true)  // static dictionary
				w.objInfo(info)
				return
			}
		}
	}

	w.Bool(false) // no dynamic subdictionary
	w.Bool(false) // no static dictionary
}

// multiExpr writes a sequence of expressions, where the i'th value is
// implicitly converted to dstType(i). It also handles when exprs is a
// single, multi-valued expression (e.g., the multi-valued argument in
// an f(g()) call, or the RHS operand in a comma-ok assignment).
func (w *writer) multiExpr(pos poser, dstType func(int) types2.Type, exprs []syntax.Expr) {
	w.Sync(pkgbits.SyncMultiExpr)

	if len(exprs) == 1 {
		expr := exprs[0]
		if tuple, ok := w.p.typeOf(expr).(*types2.Tuple); ok {
			assert(tuple.Len() > 1)
			w.Bool(true) // N:1 assignment
			w.pos(pos)
			w.expr(expr)

			w.Len(tuple.Len())
			for i := 0; i < tuple.Len(); i++ {
				src := tuple.At(i).Type()
				// TODO(mdempsky): Investigate not writing src here. I think
				// the reader should be able to infer it from expr anyway.
				w.typ(src)
				if dst := dstType(i); w.Bool(dst != nil && !types2.Identical(src, dst)) {
					if src == nil || dst == nil {
						w.p.fatalf(pos, "src is %v, dst is %v", src, dst)
					}
					if !types2.AssignableTo(src, dst) {
						w.p.fatalf(pos, "%v is not assignable to %v", src, dst)
					}
					w.typ(dst)
					w.convRTTI(src, dst)
				}
			}
			return
		}
	}

	w.Bool(false) // N:N assignment
	w.Len(len(exprs))
	for i, expr := range exprs {
		w.implicitConvExpr(dstType(i), expr)
	}
}

// implicitConvExpr is like expr, but if dst is non-nil and different
// from expr's type, then an implicit conversion operation is inserted
// at expr's position.
func (w *writer) implicitConvExpr(dst types2.Type, expr syntax.Expr) {
	w.convertExpr(dst, expr, true)
}

func (w *writer) convertExpr(dst types2.Type, expr syntax.Expr, implicit bool) {
	src := w.p.typeOf(expr)

	// Omit implicit no-op conversions.
	identical := dst == nil || types2.Identical(src, dst)
	if implicit && identical {
		w.expr(expr)
		return
	}

	if implicit && !types2.AssignableTo(src, dst) {
		w.p.fatalf(expr, "%v is not assignable to %v", src, dst)
	}

	w.Code(exprConvert)
	w.Bool(implicit)
	w.typ(dst)
	w.pos(expr)
	w.convRTTI(src, dst)
	w.Bool(isTypeParam(dst))
	w.expr(expr)
}

func (w *writer) compLit(lit *syntax.CompositeLit) {
	typ := w.p.typeOf(lit)

	w.Sync(pkgbits.SyncCompLit)
	w.pos(lit)
	w.typ(typ)

	if ptr, ok := types2.CoreType(typ).(*types2.Pointer); ok {
		typ = ptr.Elem()
	}
	var keyType, elemType types2.Type
	var structType *types2.Struct
	switch typ0 := typ; typ := types2.CoreType(typ).(type) {
	default:
		w.p.fatalf(lit, "unexpected composite literal type: %v", typ)
	case *types2.Array:
		elemType = typ.Elem()
	case *types2.Map:
		w.rtype(typ0)
		keyType, elemType = typ.Key(), typ.Elem()
	case *types2.Slice:
		elemType = typ.Elem()
	case *types2.Struct:
		structType = typ
	}

	w.Len(len(lit.ElemList))
	for i, elem := range lit.ElemList {
		elemType := elemType
		if structType != nil {
			if kv, ok := elem.(*syntax.KeyValueExpr); ok {
				// use position of expr.Key rather than of elem (which has position of ':')
				w.pos(kv.Key)
				i = fieldIndex(w.p.info, structType, kv.Key.(*syntax.Name))
				elem = kv.Value
			} else {
				w.pos(elem)
			}
			elemType = structType.Field(i).Type()
			w.Len(i)
		} else {
			if kv, ok := elem.(*syntax.KeyValueExpr); w.Bool(ok) {
				// use position of expr.Key rather than of elem (which has position of ':')
				w.pos(kv.Key)
				w.implicitConvExpr(keyType, kv.Key)
				elem = kv.Value
			}
		}
		w.pos(elem)
		w.implicitConvExpr(elemType, elem)
	}
}

func (w *writer) funcLit(expr *syntax.FuncLit) {
	sig := w.p.typeOf(expr).(*types2.Signature)

	body, closureVars := w.p.bodyIdx(sig, expr.Body, w.dict)

	w.Sync(pkgbits.SyncFuncLit)
	w.pos(expr)
	w.signature(sig)

	w.Len(len(closureVars))
	for _, cv := range closureVars {
		w.pos(cv.pos)
		w.useLocal(cv.pos, cv.var_)
	}

	w.Reloc(pkgbits.RelocBody, body)
}

type posVar struct {
	pos  syntax.Pos
	var_ *types2.Var
}

func (w *writer) exprList(expr syntax.Expr) {
	w.Sync(pkgbits.SyncExprList)
	w.exprs(unpackListExpr(expr))
}

func (w *writer) exprs(exprs []syntax.Expr) {
	w.Sync(pkgbits.SyncExprs)
	w.Len(len(exprs))
	for _, expr := range exprs {
		w.expr(expr)
	}
}

// rtype writes information so that the reader can construct an
// expression of type *runtime._type representing typ.
func (w *writer) rtype(typ types2.Type) {
	typ = types2.Default(typ)

	w.Sync(pkgbits.SyncRType)

	info := w.p.typIdx(typ, w.dict)
	if w.Bool(info.derived) {
		w.Len(w.dict.rtypeIdx(info))
	} else {
		w.typInfo(info)
	}
}

// varDictIndex writes out information for populating DictIndex for
// the ir.Name that will represent obj.
func (w *writer) varDictIndex(obj *types2.Var) {
	info := w.p.typIdx(obj.Type(), w.dict)
	if w.Bool(info.derived) {
		w.Len(w.dict.rtypeIdx(info))
	}
}

func isUntyped(typ types2.Type) bool {
	basic, ok := typ.(*types2.Basic)
	return ok && basic.Info()&types2.IsUntyped != 0
}

func isTuple(typ types2.Type) bool {
	_, ok := typ.(*types2.Tuple)
	return ok
}

func (w *writer) itab(typ, iface types2.Type) {
	typ = types2.Default(typ)
	iface = types2.Default(iface)

	typInfo := w.p.typIdx(typ, w.dict)
	ifaceInfo := w.p.typIdx(iface, w.dict)
	if w.Bool(typInfo.derived || ifaceInfo.derived) {
		w.Len(w.dict.itabIdx(typInfo, ifaceInfo))
	} else {
		w.typInfo(typInfo)
		w.typInfo(ifaceInfo)
	}
}

// convRTTI writes information so that the reader can construct
// expressions for converting from src to dst.
func (w *writer) convRTTI(src, dst types2.Type) {
	w.Sync(pkgbits.SyncConvRTTI)
	w.itab(src, dst)
}

func (w *writer) exprType(iface types2.Type, typ syntax.Expr) {
	base.Assertf(iface == nil || isInterface(iface), "%v must be nil or an interface type", iface)

	tv, ok := w.p.info.Types[typ]
	assert(ok)
	assert(tv.IsType())

	w.Sync(pkgbits.SyncExprType)
	w.pos(typ)

	if w.Bool(iface != nil && !iface.Underlying().(*types2.Interface).Empty()) {
		w.itab(tv.Type, iface)
	} else {
		w.rtype(tv.Type)

		info := w.p.typIdx(tv.Type, w.dict)
		w.Bool(info.derived)
	}
}

// isInterface reports whether typ is known to be an interface type.
// If typ is a type parameter, then isInterface reports an internal
// compiler error instead.
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

// op writes an Op into the bitstream.
func (w *writer) op(op ir.Op) {
	// TODO(mdempsky): Remove in favor of explicit codes? Would make
	// export data more stable against internal refactorings, but low
	// priority at the moment.
	assert(op != 0)
	w.Sync(pkgbits.SyncOp)
	w.Len(int(op))
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

// declCollector is a visitor type that collects compiler-needed
// information about declarations that types2 doesn't track.
//
// Notably, it maps declared types and functions back to their
// declaration statement, keeps track of implicit type parameters, and
// assigns unique type "generation" numbers to local defined types.
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
			pw.checkPragmas(n.Pragma, 0, false)

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
				pw.errorf(l.pos, "//go:linkname must refer to declared function or variable")
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

		// TODO(mdempsky): It would make sense to use multiExpr here, but
		// that results in IR that confuses pkginit/initorder.go. So we
		// continue using exprList, and let typecheck handle inserting any
		// implicit conversions. That's okay though, because package-scope
		// assignments never require dictionaries.
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

// hasImplicitTypeParams reports whether obj is a defined type with
// implicit type parameters (e.g., declared within a generic function
// or method).
func (p *pkgWriter) hasImplicitTypeParams(obj *types2.TypeName) bool {
	if obj.Pkg() == p.curpkg {
		decl, ok := p.typDecls[obj]
		assert(ok)
		if len(decl.implicits) != 0 {
			return true
		}
	}
	return false
}

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

// isMultiValueExpr reports whether expr is a function call expression
// that yields multiple values.
func isMultiValueExpr(info *types2.Info, expr syntax.Expr) bool {
	tv, ok := info.Types[expr]
	assert(ok)
	assert(tv.IsValue())
	if tuple, ok := tv.Type.(*types2.Tuple); ok {
		assert(tuple.Len() > 1)
		return true
	}
	return false
}

// isNil reports whether expr is a (possibly parenthesized) reference
// to the predeclared nil value.
func isNil(info *types2.Info, expr syntax.Expr) bool {
	tv, ok := info.Types[expr]
	assert(ok)
	return tv.IsNil()
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

// splitNamed decomposes a use of a defined type into its original
// type definition and the type arguments used to instantiate it.
func splitNamed(typ *types2.Named) (*types2.TypeName, *types2.TypeList) {
	base.Assertf(typ.TypeParams().Len() == typ.TypeArgs().Len(), "use of uninstantiated type: %v", typ)

	orig := typ.Origin()
	base.Assertf(orig.TypeArgs() == nil, "origin %v of %v has type arguments", orig, typ)
	base.Assertf(typ.Obj() == orig.Obj(), "%v has object %v, but %v has object %v", typ, typ.Obj(), orig, orig.Obj())

	return typ.Obj(), typ.TypeArgs()
}

func asPragmaFlag(p syntax.Pragma) ir.PragmaFlag {
	if p == nil {
		return 0
	}
	return p.(*pragmas).Flag
}

// isPtrTo reports whether from is the type *to.
func isPtrTo(from, to types2.Type) bool {
	ptr, ok := from.(*types2.Pointer)
	return ok && types2.Identical(ptr.Elem(), to)
}
