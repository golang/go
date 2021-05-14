// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

type pkgReader2 struct {
	pkgDecoder

	check   *types2.Checker
	imports map[string]*types2.Package

	posBases []*syntax.PosBase
	pkgs     []*types2.Package
	typs     []types2.Type
}

func readPackage2(check *types2.Checker, imports map[string]*types2.Package, input pkgDecoder) *types2.Package {
	pr := pkgReader2{
		pkgDecoder: input,

		check:   check,
		imports: imports,

		posBases: make([]*syntax.PosBase, input.numElems(relocPosBase)),
		pkgs:     make([]*types2.Package, input.numElems(relocPkg)),
		typs:     make([]types2.Type, input.numElems(relocType)),
	}

	r := pr.newReader(relocMeta, publicRootIdx, syncPublic)
	pkg := r.pkg()
	r.bool() // has init

	for i, n := 0, r.len(); i < n; i++ {
		r.obj()
	}

	r.sync(syncEOF)

	pkg.MarkComplete()
	return pkg
}

type reader2 struct {
	decoder

	p *pkgReader2

	tparams []*types2.TypeName
}

func (pr *pkgReader2) newReader(k reloc, idx int, marker syncMarker) *reader2 {
	return &reader2{
		decoder: pr.newDecoder(k, idx, marker),
		p:       pr,
	}
}

// @@@ Positions

func (r *reader2) pos() syntax.Pos {
	r.sync(syncPos)
	if !r.bool() {
		return syntax.Pos{}
	}

	// TODO(mdempsky): Delta encoding.
	posBase := r.posBase()
	line := r.uint()
	col := r.uint()
	return syntax.MakePos(posBase, line, col)
}

func (r *reader2) posBase() *syntax.PosBase {
	return r.p.posBaseIdx(r.reloc(relocPosBase))
}

func (pr *pkgReader2) posBaseIdx(idx int) *syntax.PosBase {
	if b := pr.posBases[idx]; b != nil {
		return b
	}

	r := pr.newReader(relocPosBase, idx, syncPosBase)
	var b *syntax.PosBase

	filename := r.string()
	_ = r.string() // absolute file name

	if r.bool() {
		b = syntax.NewFileBase(filename)
	} else {
		pos := r.pos()
		line := r.uint()
		col := r.uint()
		b = syntax.NewLineBase(pos, filename, line, col)
	}

	pr.posBases[idx] = b
	return b
}

// @@@ Packages

func (r *reader2) pkg() *types2.Package {
	r.sync(syncPkg)
	return r.p.pkgIdx(r.reloc(relocPkg))
}

func (pr *pkgReader2) pkgIdx(idx int) *types2.Package {
	// TODO(mdempsky): Consider using some non-nil pointer to indicate
	// the universe scope, so we don't need to keep re-reading it.
	if pkg := pr.pkgs[idx]; pkg != nil {
		return pkg
	}

	pkg := pr.newReader(relocPkg, idx, syncPkgDef).doPkg()
	pr.pkgs[idx] = pkg
	return pkg
}

func (r *reader2) doPkg() *types2.Package {
	path := r.string()
	if path == "builtin" {
		return nil // universe
	}
	if path == "" {
		path = r.p.pkgPath
	}

	if pkg := r.p.imports[path]; pkg != nil {
		return pkg
	}

	name := r.string()
	height := r.len()

	pkg := types2.NewPackageHeight(path, name, height)
	r.p.imports[path] = pkg

	// TODO(mdempsky): The list of imported packages is important for
	// go/types, but we could probably skip populating it for types2.
	imports := make([]*types2.Package, r.len())
	for i := range imports {
		imports[i] = r.pkg()
	}
	pkg.SetImports(imports)

	return pkg
}

// @@@ Types

func (r *reader2) typ() types2.Type {
	r.sync(syncType)
	return r.p.typIdx(r.reloc(relocType), r.tparams)
}

func (pr *pkgReader2) typIdx(idx int, tparams []*types2.TypeName) types2.Type {
	if typ := pr.typs[idx]; typ != nil {
		return typ
	}

	r := pr.newReader(relocType, idx, syncTypeIdx)
	r.tparams = tparams
	typ := r.doTyp()
	assert(typ != nil)

	if pr.typs[idx] != nil {
		// See comment in pkgReader.typIdx.
		return pr.typs[idx]
	}

	if len(tparams) == 0 {
		pr.typs[idx] = typ
	}

	return typ
}

func (r *reader2) doTyp() (res types2.Type) {
	switch tag := codeType(r.code(syncType)); tag {
	default:
		base.FatalfAt(src.NoXPos, "unhandled type tag: %v", tag)
		panic("unreachable")

	case typeBasic:
		return types2.Typ[r.len()]

	case typeNamed:
		obj, targs := r.obj()
		name := obj.(*types2.TypeName)
		if len(targs) != 0 {
			return r.p.check.InstantiateLazy(syntax.Pos{}, name.Type(), targs)
		}
		return name.Type()

	case typeTypeParam:
		idx := r.len()
		return r.tparams[idx].Type().(*types2.TypeParam)

	case typeArray:
		len := int64(r.uint64())
		return types2.NewArray(r.typ(), len)
	case typeChan:
		dir := types2.ChanDir(r.len())
		return types2.NewChan(dir, r.typ())
	case typeMap:
		return types2.NewMap(r.typ(), r.typ())
	case typePointer:
		return types2.NewPointer(r.typ())
	case typeSignature:
		return r.signature(nil)
	case typeSlice:
		return types2.NewSlice(r.typ())
	case typeStruct:
		return r.structType()
	case typeInterface:
		return r.interfaceType()
	case typeUnion:
		return r.unionType()
	}
}

func (r *reader2) structType() *types2.Struct {
	fields := make([]*types2.Var, r.len())
	var tags []string
	for i := range fields {
		pos := r.pos()
		pkg, name := r.selector()
		ftyp := r.typ()
		tag := r.string()
		embedded := r.bool()

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

func (r *reader2) unionType() *types2.Union {
	terms := make([]types2.Type, r.len())
	tildes := make([]bool, len(terms))
	for i := range terms {
		terms[i] = r.typ()
		tildes[i] = r.bool()
	}
	return types2.NewUnion(terms, tildes)
}

func (r *reader2) interfaceType() *types2.Interface {
	methods := make([]*types2.Func, r.len())
	embeddeds := make([]types2.Type, r.len())

	for i := range methods {
		pos := r.pos()
		pkg, name := r.selector()
		mtyp := r.signature(nil)
		methods[i] = types2.NewFunc(pos, pkg, name, mtyp)
	}

	for i := range embeddeds {
		embeddeds[i] = r.typ()
	}

	typ := types2.NewInterfaceType(methods, embeddeds)
	typ.Complete()
	return typ
}

func (r *reader2) signature(recv *types2.Var) *types2.Signature {
	r.sync(syncSignature)

	params := r.params()
	results := r.params()
	variadic := r.bool()

	return types2.NewSignature(recv, params, results, variadic)
}

func (r *reader2) params() *types2.Tuple {
	r.sync(syncParams)
	params := make([]*types2.Var, r.len())
	for i := range params {
		params[i] = r.param()
	}
	return types2.NewTuple(params...)
}

func (r *reader2) param() *types2.Var {
	r.sync(syncParam)

	pos := r.pos()
	pkg, name := r.localIdent()
	typ := r.typ()

	return types2.NewParam(pos, pkg, name, typ)
}

// @@@ Objects

func (r *reader2) obj() (types2.Object, []types2.Type) {
	r.sync(syncObject)

	pkg, name := r.p.objIdx(r.reloc(relocObj))
	obj := pkg.Scope().Lookup(name)

	targs := make([]types2.Type, r.len())
	for i := range targs {
		targs[i] = r.typ()
	}

	return obj, targs
}

func (pr *pkgReader2) objIdx(idx int) (*types2.Package, string) {
	r := pr.newReader(relocObj, idx, syncObject1)
	objPkg, objName := r.qualifiedIdent()
	assert(objName != "")

	bounds := r.typeParamBounds()
	tag := codeObj(r.code(syncCodeObj))

	if tag == objStub {
		assert(objPkg == nil)
		return objPkg, objName
	}

	objPkg.Scope().InsertLazy(objName, func() types2.Object {
		switch tag {
		default:
			panic("weird")

		case objAlias:
			pos := r.pos()
			typ := r.typ()
			return types2.NewTypeName(pos, objPkg, objName, typ)

		case objConst:
			pos := r.pos()
			typ, val := r.value()
			return types2.NewConst(pos, objPkg, objName, typ, val)

		case objFunc:
			pos := r.pos()
			r.typeParamNames(bounds)
			sig := r.signature(nil)
			if len(r.tparams) != 0 {
				sig.SetTParams(r.tparams)
			}
			return types2.NewFunc(pos, objPkg, objName, sig)

		case objType:
			pos := r.pos()

			return types2.NewTypeNameLazy(pos, objPkg, objName, func(named *types2.Named) (tparams []*types2.TypeName, underlying types2.Type, methods []*types2.Func) {
				r.typeParamNames(bounds)
				if len(r.tparams) != 0 {
					tparams = r.tparams
				}

				// TODO(mdempsky): Rewrite receiver types to underlying is an
				// Interface? The go/types importer does this (I think because
				// unit tests expected that), but cmd/compile doesn't care
				// about it, so maybe we can avoid worrying about that here.
				underlying = r.typ().Underlying()

				methods = make([]*types2.Func, r.len())
				for i := range methods {
					methods[i] = r.method(bounds)
				}

				return
			})

		case objVar:
			pos := r.pos()
			typ := r.typ()
			return types2.NewVar(pos, objPkg, objName, typ)
		}
	})

	return objPkg, objName
}

func (r *reader2) value() (types2.Type, constant.Value) {
	r.sync(syncValue)
	return r.typ(), r.rawValue()
}

func (r *reader2) typeParamBounds() []int {
	r.sync(syncTypeParamBounds)

	// exported types never have implicit type parameters
	// TODO(mdempsky): Hide this from public importer.
	assert(r.len() == 0)

	bounds := make([]int, r.len())
	for i := range bounds {
		r.sync(syncType)
		bounds[i] = r.reloc(relocType)
	}
	return bounds
}

func (r *reader2) typeParamNames(bounds []int) {
	r.sync(syncTypeParamNames)

	r.tparams = make([]*types2.TypeName, len(bounds))

	for i := range r.tparams {
		pos := r.pos()
		pkg, name := r.localIdent()

		obj := types2.NewTypeName(pos, pkg, name, nil)
		r.p.check.NewTypeParam(obj, i, nil)
		r.tparams[i] = obj
	}

	for i, tparam := range r.tparams {
		bound := r.p.typIdx(bounds[i], r.tparams)
		tparam.Type().(*types2.TypeParam).SetBound(bound)
	}
}

func (r *reader2) method(bounds []int) *types2.Func {
	r.sync(syncMethod)
	pos := r.pos()
	pkg, name := r.selector()

	r.typeParamNames(bounds)
	sig := r.signature(r.param())
	if len(r.tparams) != 0 {
		sig.SetRParams(r.tparams)
	}

	_ = r.pos() // TODO(mdempsky): Remove; this is a hacker for linker.go.
	return types2.NewFunc(pos, pkg, name, sig)
}

func (r *reader2) qualifiedIdent() (*types2.Package, string) { return r.ident(syncSym) }
func (r *reader2) localIdent() (*types2.Package, string)     { return r.ident(syncLocalIdent) }
func (r *reader2) selector() (*types2.Package, string)       { return r.ident(syncSelector) }

func (r *reader2) ident(marker syncMarker) (*types2.Package, string) {
	r.sync(marker)
	return r.pkg(), r.string()
}
