// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
	"internal/pkgbits"
)

type pkgReader struct {
	pkgbits.PkgDecoder

	ctxt        *types2.Context
	imports     map[string]*types2.Package
	enableAlias bool // whether to use aliases

	posBases []*syntax.PosBase
	pkgs     []*types2.Package
	typs     []types2.Type
}

func ReadPackage(ctxt *types2.Context, imports map[string]*types2.Package, input pkgbits.PkgDecoder) *types2.Package {
	pr := pkgReader{
		PkgDecoder: input,

		ctxt:        ctxt,
		imports:     imports,
		enableAlias: true,

		posBases: make([]*syntax.PosBase, input.NumElems(pkgbits.RelocPosBase)),
		pkgs:     make([]*types2.Package, input.NumElems(pkgbits.RelocPkg)),
		typs:     make([]types2.Type, input.NumElems(pkgbits.RelocType)),
	}

	r := pr.newReader(pkgbits.RelocMeta, pkgbits.PublicRootIdx, pkgbits.SyncPublic)
	pkg := r.pkg()
	r.Bool() // TODO(mdempsky): Remove; was "has init"

	for i, n := 0, r.Len(); i < n; i++ {
		// As if r.obj(), but avoiding the Scope.Lookup call,
		// to avoid eager loading of imports.
		r.Sync(pkgbits.SyncObject)
		assert(!r.Bool())
		r.p.objIdx(r.Reloc(pkgbits.RelocObj))
		assert(r.Len() == 0)
	}

	r.Sync(pkgbits.SyncEOF)

	pkg.MarkComplete()
	return pkg
}

type reader struct {
	pkgbits.Decoder

	p *pkgReader

	dict *readerDict
}

type readerDict struct {
	bounds []typeInfo

	tparams []*types2.TypeParam

	derived      []derivedInfo
	derivedTypes []types2.Type
}

type readerTypeBound struct {
	derived  bool
	boundIdx int
}

func (pr *pkgReader) newReader(k pkgbits.RelocKind, idx pkgbits.Index, marker pkgbits.SyncMarker) *reader {
	return &reader{
		Decoder: pr.NewDecoder(k, idx, marker),
		p:       pr,
	}
}

func (pr *pkgReader) tempReader(k pkgbits.RelocKind, idx pkgbits.Index, marker pkgbits.SyncMarker) *reader {
	return &reader{
		Decoder: pr.TempDecoder(k, idx, marker),
		p:       pr,
	}
}

func (pr *pkgReader) retireReader(r *reader) {
	pr.RetireDecoder(&r.Decoder)
}

// @@@ Positions

func (r *reader) pos() syntax.Pos {
	r.Sync(pkgbits.SyncPos)
	if !r.Bool() {
		return syntax.Pos{}
	}

	// TODO(mdempsky): Delta encoding.
	posBase := r.posBase()
	line := r.Uint()
	col := r.Uint()
	return syntax.MakePos(posBase, line, col)
}

func (r *reader) posBase() *syntax.PosBase {
	return r.p.posBaseIdx(r.Reloc(pkgbits.RelocPosBase))
}

func (pr *pkgReader) posBaseIdx(idx pkgbits.Index) *syntax.PosBase {
	if b := pr.posBases[idx]; b != nil {
		return b
	}
	var b *syntax.PosBase
	{
		r := pr.tempReader(pkgbits.RelocPosBase, idx, pkgbits.SyncPosBase)

		filename := r.String()

		if r.Bool() {
			b = syntax.NewTrimmedFileBase(filename, true)
		} else {
			pos := r.pos()
			line := r.Uint()
			col := r.Uint()
			b = syntax.NewLineBase(pos, filename, true, line, col)
		}
		pr.retireReader(r)
	}

	pr.posBases[idx] = b
	return b
}

// @@@ Packages

func (r *reader) pkg() *types2.Package {
	r.Sync(pkgbits.SyncPkg)
	return r.p.pkgIdx(r.Reloc(pkgbits.RelocPkg))
}

func (pr *pkgReader) pkgIdx(idx pkgbits.Index) *types2.Package {
	// TODO(mdempsky): Consider using some non-nil pointer to indicate
	// the universe scope, so we don't need to keep re-reading it.
	if pkg := pr.pkgs[idx]; pkg != nil {
		return pkg
	}

	pkg := pr.newReader(pkgbits.RelocPkg, idx, pkgbits.SyncPkgDef).doPkg()
	pr.pkgs[idx] = pkg
	return pkg
}

func (r *reader) doPkg() *types2.Package {
	path := r.String()
	switch path {
	case "":
		path = r.p.PkgPath()
	case "builtin":
		return nil // universe
	case "unsafe":
		return types2.Unsafe
	}

	if pkg := r.p.imports[path]; pkg != nil {
		return pkg
	}

	name := r.String()
	pkg := types2.NewPackage(path, name)
	r.p.imports[path] = pkg

	// TODO(mdempsky): The list of imported packages is important for
	// go/types, but we could probably skip populating it for types2.
	imports := make([]*types2.Package, r.Len())
	for i := range imports {
		imports[i] = r.pkg()
	}
	pkg.SetImports(imports)

	return pkg
}

// @@@ Types

func (r *reader) typ() types2.Type {
	return r.p.typIdx(r.typInfo(), r.dict)
}

func (r *reader) typInfo() typeInfo {
	r.Sync(pkgbits.SyncType)
	if r.Bool() {
		return typeInfo{idx: pkgbits.Index(r.Len()), derived: true}
	}
	return typeInfo{idx: r.Reloc(pkgbits.RelocType), derived: false}
}

func (pr *pkgReader) typIdx(info typeInfo, dict *readerDict) types2.Type {
	idx := info.idx
	var where *types2.Type
	if info.derived {
		where = &dict.derivedTypes[idx]
		idx = dict.derived[idx].idx
	} else {
		where = &pr.typs[idx]
	}

	if typ := *where; typ != nil {
		return typ
	}

	var typ types2.Type
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

func (r *reader) doTyp() (res types2.Type) {
	switch tag := pkgbits.CodeType(r.Code(pkgbits.SyncType)); tag {
	default:
		base.FatalfAt(src.NoXPos, "unhandled type tag: %v", tag)
		panic("unreachable")

	case pkgbits.TypeBasic:
		return types2.Typ[r.Len()]

	case pkgbits.TypeNamed:
		obj, targs := r.obj()
		name := obj.(*types2.TypeName)
		if len(targs) != 0 {
			t, _ := types2.Instantiate(r.p.ctxt, name.Type(), targs, false)
			return t
		}
		return name.Type()

	case pkgbits.TypeTypeParam:
		return r.dict.tparams[r.Len()]

	case pkgbits.TypeArray:
		len := int64(r.Uint64())
		return types2.NewArray(r.typ(), len)
	case pkgbits.TypeChan:
		dir := types2.ChanDir(r.Len())
		return types2.NewChan(dir, r.typ())
	case pkgbits.TypeMap:
		return types2.NewMap(r.typ(), r.typ())
	case pkgbits.TypePointer:
		return types2.NewPointer(r.typ())
	case pkgbits.TypeSignature:
		return r.signature(nil, nil, nil)
	case pkgbits.TypeSlice:
		return types2.NewSlice(r.typ())
	case pkgbits.TypeStruct:
		return r.structType()
	case pkgbits.TypeInterface:
		return r.interfaceType()
	case pkgbits.TypeUnion:
		return r.unionType()
	}
}

func (r *reader) structType() *types2.Struct {
	fields := make([]*types2.Var, r.Len())
	var tags []string
	for i := range fields {
		pos := r.pos()
		pkg, name := r.selector()
		ftyp := r.typ()
		tag := r.String()
		embedded := r.Bool()

		fields[i] = types2.NewField(pos, pkg, name, ftyp, embedded)
		if tag != "" {
			for len(tags) < i {
				tags = append(tags, "")
			}
			tags = append(tags, tag)
		}
	}
	return types2.NewStruct(fields, tags)
}

func (r *reader) unionType() *types2.Union {
	terms := make([]*types2.Term, r.Len())
	for i := range terms {
		terms[i] = types2.NewTerm(r.Bool(), r.typ())
	}
	return types2.NewUnion(terms)
}

func (r *reader) interfaceType() *types2.Interface {
	methods := make([]*types2.Func, r.Len())
	embeddeds := make([]types2.Type, r.Len())
	implicit := len(methods) == 0 && len(embeddeds) == 1 && r.Bool()

	for i := range methods {
		pos := r.pos()
		pkg, name := r.selector()
		mtyp := r.signature(nil, nil, nil)
		methods[i] = types2.NewFunc(pos, pkg, name, mtyp)
	}

	for i := range embeddeds {
		embeddeds[i] = r.typ()
	}

	iface := types2.NewInterfaceType(methods, embeddeds)
	if implicit {
		iface.MarkImplicit()
	}
	return iface
}

func (r *reader) signature(recv *types2.Var, rtparams, tparams []*types2.TypeParam) *types2.Signature {
	r.Sync(pkgbits.SyncSignature)

	params := r.params()
	results := r.params()
	variadic := r.Bool()

	return types2.NewSignatureType(recv, rtparams, tparams, params, results, variadic)
}

func (r *reader) params() *types2.Tuple {
	r.Sync(pkgbits.SyncParams)
	params := make([]*types2.Var, r.Len())
	for i := range params {
		params[i] = r.param()
	}
	return types2.NewTuple(params...)
}

func (r *reader) param() *types2.Var {
	r.Sync(pkgbits.SyncParam)

	pos := r.pos()
	pkg, name := r.localIdent()
	typ := r.typ()

	return types2.NewParam(pos, pkg, name, typ)
}

// @@@ Objects

func (r *reader) obj() (types2.Object, []types2.Type) {
	r.Sync(pkgbits.SyncObject)

	assert(!r.Bool())

	pkg, name := r.p.objIdx(r.Reloc(pkgbits.RelocObj))
	obj := pkg.Scope().Lookup(name)

	targs := make([]types2.Type, r.Len())
	for i := range targs {
		targs[i] = r.typ()
	}

	return obj, targs
}

func (pr *pkgReader) objIdx(idx pkgbits.Index) (*types2.Package, string) {
	var objPkg *types2.Package
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
		base.Assertf(objPkg == nil || objPkg == types2.Unsafe, "unexpected stub package: %v", objPkg)
		return objPkg, objName
	}

	objPkg.Scope().InsertLazy(objName, func() types2.Object {
		dict := pr.objDictIdx(idx)

		r := pr.newReader(pkgbits.RelocObj, idx, pkgbits.SyncObject1)
		r.dict = dict

		switch tag {
		default:
			panic("weird")

		case pkgbits.ObjAlias:
			pos := r.pos()
			var tparams []*types2.TypeParam // TODO(#68778): Read tparams for unified IR.
			typ := r.typ()
			return newAliasTypeName(pr.enableAlias, pos, objPkg, objName, typ, tparams)

		case pkgbits.ObjConst:
			pos := r.pos()
			typ := r.typ()
			val := r.Value()
			return types2.NewConst(pos, objPkg, objName, typ, val)

		case pkgbits.ObjFunc:
			pos := r.pos()
			tparams := r.typeParamNames()
			sig := r.signature(nil, nil, tparams)
			return types2.NewFunc(pos, objPkg, objName, sig)

		case pkgbits.ObjType:
			pos := r.pos()

			return types2.NewTypeNameLazy(pos, objPkg, objName, func(named *types2.Named) (tparams []*types2.TypeParam, underlying types2.Type, methods []*types2.Func) {
				tparams = r.typeParamNames()

				// TODO(mdempsky): Rewrite receiver types to underlying is an
				// Interface? The go/types importer does this (I think because
				// unit tests expected that), but cmd/compile doesn't care
				// about it, so maybe we can avoid worrying about that here.
				underlying = r.typ().Underlying()

				methods = make([]*types2.Func, r.Len())
				for i := range methods {
					methods[i] = r.method()
				}

				return
			})

		case pkgbits.ObjVar:
			pos := r.pos()
			typ := r.typ()
			return types2.NewVar(pos, objPkg, objName, typ)
		}
	})

	return objPkg, objName
}

func (pr *pkgReader) objDictIdx(idx pkgbits.Index) *readerDict {
	var dict readerDict
	{
		r := pr.tempReader(pkgbits.RelocObjDict, idx, pkgbits.SyncObject1)

		if implicits := r.Len(); implicits != 0 {
			base.Fatalf("unexpected object with %v implicit type parameter(s)", implicits)
		}

		dict.bounds = make([]typeInfo, r.Len())
		for i := range dict.bounds {
			dict.bounds[i] = r.typInfo()
		}

		dict.derived = make([]derivedInfo, r.Len())
		dict.derivedTypes = make([]types2.Type, len(dict.derived))
		for i := range dict.derived {
			dict.derived[i] = derivedInfo{r.Reloc(pkgbits.RelocType), r.Bool()}
		}

		pr.retireReader(r)
	}
	// function references follow, but reader doesn't need those

	return &dict
}

func (r *reader) typeParamNames() []*types2.TypeParam {
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

	r.dict.tparams = make([]*types2.TypeParam, len(r.dict.bounds))
	for i := range r.dict.bounds {
		pos := r.pos()
		pkg, name := r.localIdent()

		tname := types2.NewTypeName(pos, pkg, name, nil)
		r.dict.tparams[i] = types2.NewTypeParam(tname, nil)
	}

	for i, bound := range r.dict.bounds {
		r.dict.tparams[i].SetConstraint(r.p.typIdx(bound, r.dict))
	}

	return r.dict.tparams
}

func (r *reader) method() *types2.Func {
	r.Sync(pkgbits.SyncMethod)
	pos := r.pos()
	pkg, name := r.selector()

	rtparams := r.typeParamNames()
	sig := r.signature(r.param(), rtparams, nil)

	_ = r.pos() // TODO(mdempsky): Remove; this is a hacker for linker.go.
	return types2.NewFunc(pos, pkg, name, sig)
}

func (r *reader) qualifiedIdent() (*types2.Package, string) { return r.ident(pkgbits.SyncSym) }
func (r *reader) localIdent() (*types2.Package, string)     { return r.ident(pkgbits.SyncLocalIdent) }
func (r *reader) selector() (*types2.Package, string)       { return r.ident(pkgbits.SyncSelector) }

func (r *reader) ident(marker pkgbits.SyncMarker) (*types2.Package, string) {
	r.Sync(marker)
	return r.pkg(), r.String()
}

// newAliasTypeName returns a new TypeName, with a materialized *types2.Alias if supported.
func newAliasTypeName(aliases bool, pos syntax.Pos, pkg *types2.Package, name string, rhs types2.Type, tparams []*types2.TypeParam) *types2.TypeName {
	// Copied from x/tools/internal/aliases.NewAlias via
	// GOROOT/src/go/internal/gcimporter/ureader.go.
	if aliases {
		tname := types2.NewTypeName(pos, pkg, name, nil)
		a := types2.NewAlias(tname, rhs) // form TypeName -> Alias cycle
		a.SetTypeParams(tparams)
		return tname
	}
	assert(len(tparams) == 0)
	return types2.NewTypeName(pos, pkg, name, rhs)
}
