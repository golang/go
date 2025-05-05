// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter

import (
	"go/token"
	"go/types"
	"internal/godebug"
	"internal/pkgbits"
	"slices"
	"strings"
)

// A pkgReader holds the shared state for reading a unified IR package
// description.
type pkgReader struct {
	pkgbits.PkgDecoder

	fake fakeFileSet

	ctxt    *types.Context
	imports map[string]*types.Package // previously imported packages, indexed by path

	// lazily initialized arrays corresponding to the unified IR
	// PosBase, Pkg, and Type sections, respectively.
	posBases []string // position bases (i.e., file names)
	pkgs     []*types.Package
	typs     []types.Type

	// laterFns holds functions that need to be invoked at the end of
	// import reading.
	laterFns []func()

	// ifaces holds a list of constructed Interfaces, which need to have
	// Complete called after importing is done.
	ifaces []*types.Interface
}

// later adds a function to be invoked at the end of import reading.
func (pr *pkgReader) later(fn func()) {
	pr.laterFns = append(pr.laterFns, fn)
}

// readUnifiedPackage reads a package description from the given
// unified IR export data decoder.
func readUnifiedPackage(fset *token.FileSet, ctxt *types.Context, imports map[string]*types.Package, input pkgbits.PkgDecoder) *types.Package {
	pr := pkgReader{
		PkgDecoder: input,

		fake: fakeFileSet{
			fset:  fset,
			files: make(map[string]*fileInfo),
		},

		ctxt:    ctxt,
		imports: imports,

		posBases: make([]string, input.NumElems(pkgbits.RelocPosBase)),
		pkgs:     make([]*types.Package, input.NumElems(pkgbits.RelocPkg)),
		typs:     make([]types.Type, input.NumElems(pkgbits.RelocType)),
	}
	defer pr.fake.setLines()

	r := pr.newReader(pkgbits.RelocMeta, pkgbits.PublicRootIdx, pkgbits.SyncPublic)
	pkg := r.pkg()
	if r.Version().Has(pkgbits.HasInit) {
		r.Bool()
	}

	for i, n := 0, r.Len(); i < n; i++ {
		// As if r.obj(), but avoiding the Scope.Lookup call,
		// to avoid eager loading of imports.
		r.Sync(pkgbits.SyncObject)
		if r.Version().Has(pkgbits.DerivedFuncInstance) {
			assert(!r.Bool())
		}
		r.p.objIdx(r.Reloc(pkgbits.RelocObj))
		assert(r.Len() == 0)
	}

	r.Sync(pkgbits.SyncEOF)

	for _, fn := range pr.laterFns {
		fn()
	}

	for _, iface := range pr.ifaces {
		iface.Complete()
	}

	// Imports() of pkg are all of the transitive packages that were loaded.
	var imps []*types.Package
	for _, imp := range pr.pkgs {
		if imp != nil && imp != pkg {
			imps = append(imps, imp)
		}
	}
	slices.SortFunc(imps, func(a, b *types.Package) int {
		return strings.Compare(a.Path(), b.Path())
	})
	pkg.SetImports(imps)

	pkg.MarkComplete()
	return pkg
}

// A reader holds the state for reading a single unified IR element
// within a package.
type reader struct {
	pkgbits.Decoder

	p *pkgReader

	dict *readerDict
}

// A readerDict holds the state for type parameters that parameterize
// the current unified IR element.
type readerDict struct {
	// bounds is a slice of typeInfos corresponding to the underlying
	// bounds of the element's type parameters.
	bounds []typeInfo

	// tparams is a slice of the constructed TypeParams for the element.
	tparams []*types.TypeParam

	// derived is a slice of types derived from tparams, which may be
	// instantiated while reading the current element.
	derived      []derivedInfo
	derivedTypes []types.Type // lazily instantiated from derived
}

func (pr *pkgReader) newReader(k pkgbits.SectionKind, idx pkgbits.Index, marker pkgbits.SyncMarker) *reader {
	return &reader{
		Decoder: pr.NewDecoder(k, idx, marker),
		p:       pr,
	}
}

func (pr *pkgReader) tempReader(k pkgbits.SectionKind, idx pkgbits.Index, marker pkgbits.SyncMarker) *reader {
	return &reader{
		Decoder: pr.TempDecoder(k, idx, marker),
		p:       pr,
	}
}

func (pr *pkgReader) retireReader(r *reader) {
	pr.RetireDecoder(&r.Decoder)
}

// @@@ Positions

func (r *reader) pos() token.Pos {
	r.Sync(pkgbits.SyncPos)
	if !r.Bool() {
		return token.NoPos
	}

	// TODO(mdempsky): Delta encoding.
	posBase := r.posBase()
	line := r.Uint()
	col := r.Uint()
	return r.p.fake.pos(posBase, int(line), int(col))
}

func (r *reader) posBase() string {
	return r.p.posBaseIdx(r.Reloc(pkgbits.RelocPosBase))
}

func (pr *pkgReader) posBaseIdx(idx pkgbits.Index) string {
	if b := pr.posBases[idx]; b != "" {
		return b
	}

	var filename string
	{
		r := pr.tempReader(pkgbits.RelocPosBase, idx, pkgbits.SyncPosBase)

		// Within types2, position bases have a lot more details (e.g.,
		// keeping track of where //line directives appeared exactly).
		//
		// For go/types, we just track the file name.

		filename = r.String()

		if r.Bool() { // file base
			// Was: "b = token.NewTrimmedFileBase(filename, true)"
		} else { // line base
			pos := r.pos()
			line := r.Uint()
			col := r.Uint()

			// Was: "b = token.NewLineBase(pos, filename, true, line, col)"
			_, _, _ = pos, line, col
		}
		pr.retireReader(r)
	}
	b := filename
	pr.posBases[idx] = b
	return b
}

// @@@ Packages

func (r *reader) pkg() *types.Package {
	r.Sync(pkgbits.SyncPkg)
	return r.p.pkgIdx(r.Reloc(pkgbits.RelocPkg))
}

func (pr *pkgReader) pkgIdx(idx pkgbits.Index) *types.Package {
	// TODO(mdempsky): Consider using some non-nil pointer to indicate
	// the universe scope, so we don't need to keep re-reading it.
	if pkg := pr.pkgs[idx]; pkg != nil {
		return pkg
	}

	pkg := pr.newReader(pkgbits.RelocPkg, idx, pkgbits.SyncPkgDef).doPkg()
	pr.pkgs[idx] = pkg
	return pkg
}

func (r *reader) doPkg() *types.Package {
	path := r.String()
	switch path {
	case "":
		path = r.p.PkgPath()
	case "builtin":
		return nil // universe
	case "unsafe":
		return types.Unsafe
	}

	if pkg := r.p.imports[path]; pkg != nil {
		return pkg
	}

	name := r.String()

	pkg := types.NewPackage(path, name)
	r.p.imports[path] = pkg

	return pkg
}

// @@@ Types

func (r *reader) typ() types.Type {
	return r.p.typIdx(r.typInfo(), r.dict)
}

func (r *reader) typInfo() typeInfo {
	r.Sync(pkgbits.SyncType)
	if r.Bool() {
		return typeInfo{idx: pkgbits.Index(r.Len()), derived: true}
	}
	return typeInfo{idx: r.Reloc(pkgbits.RelocType), derived: false}
}

func (pr *pkgReader) typIdx(info typeInfo, dict *readerDict) types.Type {
	idx := info.idx
	var where *types.Type
	if info.derived {
		where = &dict.derivedTypes[idx]
		idx = dict.derived[idx].idx
	} else {
		where = &pr.typs[idx]
	}

	if typ := *where; typ != nil {
		return typ
	}

	var typ types.Type
	{
		r := pr.tempReader(pkgbits.RelocType, idx, pkgbits.SyncTypeIdx)
		r.dict = dict

		typ = r.doTyp()
		assert(typ != nil)
		pr.retireReader(r)
	}
	// See comment in pkgReader.typIdx explaining how this happens.
	if prev := *where; prev != nil {
		return prev
	}

	*where = typ
	return typ
}

func (r *reader) doTyp() (res types.Type) {
	switch tag := pkgbits.CodeType(r.Code(pkgbits.SyncType)); tag {
	default:
		errorf("unhandled type tag: %v", tag)
		panic("unreachable")

	case pkgbits.TypeBasic:
		return types.Typ[r.Len()]

	case pkgbits.TypeNamed:
		obj, targs := r.obj()
		name := obj.(*types.TypeName)
		if len(targs) != 0 {
			t, _ := types.Instantiate(r.p.ctxt, name.Type(), targs, false)
			return t
		}
		return name.Type()

	case pkgbits.TypeTypeParam:
		return r.dict.tparams[r.Len()]

	case pkgbits.TypeArray:
		len := int64(r.Uint64())
		return types.NewArray(r.typ(), len)
	case pkgbits.TypeChan:
		dir := types.ChanDir(r.Len())
		return types.NewChan(dir, r.typ())
	case pkgbits.TypeMap:
		return types.NewMap(r.typ(), r.typ())
	case pkgbits.TypePointer:
		return types.NewPointer(r.typ())
	case pkgbits.TypeSignature:
		return r.signature(nil, nil, nil)
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

func (r *reader) structType() *types.Struct {
	fields := make([]*types.Var, r.Len())
	var tags []string
	for i := range fields {
		pos := r.pos()
		pkg, name := r.selector()
		ftyp := r.typ()
		tag := r.String()
		embedded := r.Bool()

		fields[i] = types.NewField(pos, pkg, name, ftyp, embedded)
		if tag != "" {
			for len(tags) < i {
				tags = append(tags, "")
			}
			tags = append(tags, tag)
		}
	}
	return types.NewStruct(fields, tags)
}

func (r *reader) unionType() *types.Union {
	terms := make([]*types.Term, r.Len())
	for i := range terms {
		terms[i] = types.NewTerm(r.Bool(), r.typ())
	}
	return types.NewUnion(terms)
}

func (r *reader) interfaceType() *types.Interface {
	methods := make([]*types.Func, r.Len())
	embeddeds := make([]types.Type, r.Len())
	implicit := len(methods) == 0 && len(embeddeds) == 1 && r.Bool()

	for i := range methods {
		pos := r.pos()
		pkg, name := r.selector()
		mtyp := r.signature(nil, nil, nil)
		methods[i] = types.NewFunc(pos, pkg, name, mtyp)
	}

	for i := range embeddeds {
		embeddeds[i] = r.typ()
	}

	iface := types.NewInterfaceType(methods, embeddeds)
	if implicit {
		iface.MarkImplicit()
	}

	// We need to call iface.Complete(), but if there are any embedded
	// defined types, then we may not have set their underlying
	// interface type yet. So we need to defer calling Complete until
	// after we've called SetUnderlying everywhere.
	//
	// TODO(mdempsky): After CL 424876 lands, it should be safe to call
	// iface.Complete() immediately.
	r.p.ifaces = append(r.p.ifaces, iface)

	return iface
}

func (r *reader) signature(recv *types.Var, rtparams, tparams []*types.TypeParam) *types.Signature {
	r.Sync(pkgbits.SyncSignature)

	params := r.params(types.ParamVar)
	results := r.params(types.ResultVar)
	variadic := r.Bool()

	return types.NewSignatureType(recv, rtparams, tparams, params, results, variadic)
}

func (r *reader) params(kind types.VarKind) *types.Tuple {
	r.Sync(pkgbits.SyncParams)

	params := make([]*types.Var, r.Len())
	for i := range params {
		params[i] = r.param(kind)
	}

	return types.NewTuple(params...)
}

func (r *reader) param(kind types.VarKind) *types.Var {
	r.Sync(pkgbits.SyncParam)

	pos := r.pos()
	pkg, name := r.localIdent()
	typ := r.typ()

	param := types.NewParam(pos, pkg, name, typ)
	param.SetKind(kind) // ∈ {Recv,Param,Result}Var
	return param
}

// @@@ Objects

func (r *reader) obj() (types.Object, []types.Type) {
	r.Sync(pkgbits.SyncObject)

	if r.Version().Has(pkgbits.DerivedFuncInstance) {
		assert(!r.Bool())
	}

	pkg, name := r.p.objIdx(r.Reloc(pkgbits.RelocObj))
	obj := pkgScope(pkg).Lookup(name)

	targs := make([]types.Type, r.Len())
	for i := range targs {
		targs[i] = r.typ()
	}

	return obj, targs
}

func (pr *pkgReader) objIdx(idx pkgbits.Index) (*types.Package, string) {

	var objPkg *types.Package
	var objName string
	var tag pkgbits.CodeObj
	{
		rname := pr.tempReader(pkgbits.RelocName, idx, pkgbits.SyncObject1)

		objPkg, objName = rname.qualifiedIdent()
		assert(objName != "")

		tag = pkgbits.CodeObj(rname.Code(pkgbits.SyncCodeObj))
		pr.retireReader(rname)
	}

	if tag == pkgbits.ObjStub {
		assert(objPkg == nil || objPkg == types.Unsafe)
		return objPkg, objName
	}

	// Ignore local types promoted to global scope (#55110).
	if _, suffix := splitVargenSuffix(objName); suffix != "" {
		return objPkg, objName
	}

	if objPkg.Scope().Lookup(objName) == nil {
		dict := pr.objDictIdx(idx)

		r := pr.newReader(pkgbits.RelocObj, idx, pkgbits.SyncObject1)
		r.dict = dict

		declare := func(obj types.Object) {
			objPkg.Scope().Insert(obj)
		}

		switch tag {
		default:
			panic("weird")

		case pkgbits.ObjAlias:
			pos := r.pos()
			var tparams []*types.TypeParam
			if r.Version().Has(pkgbits.AliasTypeParamNames) {
				tparams = r.typeParamNames()
			}
			typ := r.typ()
			declare(newAliasTypeName(pos, objPkg, objName, typ, tparams))

		case pkgbits.ObjConst:
			pos := r.pos()
			typ := r.typ()
			val := r.Value()
			declare(types.NewConst(pos, objPkg, objName, typ, val))

		case pkgbits.ObjFunc:
			pos := r.pos()
			tparams := r.typeParamNames()
			sig := r.signature(nil, nil, tparams)
			declare(types.NewFunc(pos, objPkg, objName, sig))

		case pkgbits.ObjType:
			pos := r.pos()

			obj := types.NewTypeName(pos, objPkg, objName, nil)
			named := types.NewNamed(obj, nil, nil)
			declare(obj)

			named.SetTypeParams(r.typeParamNames())

			underlying := r.typ().Underlying()

			// If the underlying type is an interface, we need to
			// duplicate its methods so we can replace the receiver
			// parameter's type (#49906).
			if iface, ok := underlying.(*types.Interface); ok && iface.NumExplicitMethods() != 0 {
				methods := make([]*types.Func, iface.NumExplicitMethods())
				for i := range methods {
					fn := iface.ExplicitMethod(i)
					sig := fn.Signature()

					recv := types.NewVar(fn.Pos(), fn.Pkg(), "", named)
					recv.SetKind(types.RecvVar)
					methods[i] = types.NewFunc(fn.Pos(), fn.Pkg(), fn.Name(), types.NewSignature(recv, sig.Params(), sig.Results(), sig.Variadic()))
				}

				embeds := make([]types.Type, iface.NumEmbeddeds())
				for i := range embeds {
					embeds[i] = iface.EmbeddedType(i)
				}

				newIface := types.NewInterfaceType(methods, embeds)
				r.p.ifaces = append(r.p.ifaces, newIface)
				underlying = newIface
			}

			named.SetUnderlying(underlying)

			for i, n := 0, r.Len(); i < n; i++ {
				named.AddMethod(r.method())
			}

		case pkgbits.ObjVar:
			pos := r.pos()
			typ := r.typ()
			declare(types.NewVar(pos, objPkg, objName, typ))
		}
	}

	return objPkg, objName
}

func (pr *pkgReader) objDictIdx(idx pkgbits.Index) *readerDict {

	var dict readerDict

	{
		r := pr.tempReader(pkgbits.RelocObjDict, idx, pkgbits.SyncObject1)
		if implicits := r.Len(); implicits != 0 {
			errorf("unexpected object with %v implicit type parameter(s)", implicits)
		}

		dict.bounds = make([]typeInfo, r.Len())
		for i := range dict.bounds {
			dict.bounds[i] = r.typInfo()
		}

		dict.derived = make([]derivedInfo, r.Len())
		dict.derivedTypes = make([]types.Type, len(dict.derived))
		for i := range dict.derived {
			dict.derived[i] = derivedInfo{idx: r.Reloc(pkgbits.RelocType)}
			if r.Version().Has(pkgbits.DerivedInfoNeeded) {
				assert(!r.Bool())
			}
		}

		pr.retireReader(r)
	}
	// function references follow, but reader doesn't need those

	return &dict
}

func (r *reader) typeParamNames() []*types.TypeParam {
	r.Sync(pkgbits.SyncTypeParamNames)

	// Note: This code assumes it only processes objects without
	// implement type parameters. This is currently fine, because
	// reader is only used to read in exported declarations, which are
	// always package scoped.

	if len(r.dict.bounds) == 0 {
		return nil
	}

	// Careful: Type parameter lists may have cycles. To allow for this,
	// we construct the type parameter list in two passes: first we
	// create all the TypeNames and TypeParams, then we construct and
	// set the bound type.

	r.dict.tparams = make([]*types.TypeParam, len(r.dict.bounds))
	for i := range r.dict.bounds {
		pos := r.pos()
		pkg, name := r.localIdent()

		tname := types.NewTypeName(pos, pkg, name, nil)
		r.dict.tparams[i] = types.NewTypeParam(tname, nil)
	}

	typs := make([]types.Type, len(r.dict.bounds))
	for i, bound := range r.dict.bounds {
		typs[i] = r.p.typIdx(bound, r.dict)
	}

	// TODO(mdempsky): This is subtle, elaborate further.
	//
	// We have to save tparams outside of the closure, because
	// typeParamNames() can be called multiple times with the same
	// dictionary instance.
	//
	// Also, this needs to happen later to make sure SetUnderlying has
	// been called.
	//
	// TODO(mdempsky): Is it safe to have a single "later" slice or do
	// we need to have multiple passes? See comments on CL 386002 and
	// go.dev/issue/52104.
	tparams := r.dict.tparams
	r.p.later(func() {
		for i, typ := range typs {
			tparams[i].SetConstraint(typ)
		}
	})

	return r.dict.tparams
}

func (r *reader) method() *types.Func {
	r.Sync(pkgbits.SyncMethod)
	pos := r.pos()
	pkg, name := r.selector()

	rparams := r.typeParamNames()
	sig := r.signature(r.param(types.RecvVar), rparams, nil)

	_ = r.pos() // TODO(mdempsky): Remove; this is a hacker for linker.go.
	return types.NewFunc(pos, pkg, name, sig)
}

func (r *reader) qualifiedIdent() (*types.Package, string) { return r.ident(pkgbits.SyncSym) }
func (r *reader) localIdent() (*types.Package, string)     { return r.ident(pkgbits.SyncLocalIdent) }
func (r *reader) selector() (*types.Package, string)       { return r.ident(pkgbits.SyncSelector) }

func (r *reader) ident(marker pkgbits.SyncMarker) (*types.Package, string) {
	r.Sync(marker)
	return r.pkg(), r.String()
}

// pkgScope returns pkg.Scope().
// If pkg is nil, it returns types.Universe instead.
//
// TODO(mdempsky): Remove after x/tools can depend on Go 1.19.
func pkgScope(pkg *types.Package) *types.Scope {
	if pkg != nil {
		return pkg.Scope()
	}
	return types.Universe
}

// newAliasTypeName returns a new TypeName, with a materialized *types.Alias if supported.
func newAliasTypeName(pos token.Pos, pkg *types.Package, name string, rhs types.Type, tparams []*types.TypeParam) *types.TypeName {
	// When GODEBUG=gotypesalias=1 or unset, the Type() of the return value is a
	// *types.Alias. Copied from x/tools/internal/aliases.NewAlias.
	switch godebug.New("gotypesalias").Value() {
	case "", "1":
		tname := types.NewTypeName(pos, pkg, name, nil)
		a := types.NewAlias(tname, rhs) // form TypeName -> Alias cycle
		a.SetTypeParams(tparams)
		return tname
	}
	assert(len(tparams) == 0)
	return types.NewTypeName(pos, pkg, name, rhs)
}
