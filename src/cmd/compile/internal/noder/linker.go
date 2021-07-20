// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"io"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/types"
	"cmd/internal/goobj"
	"cmd/internal/obj"
)

// This file implements the unified IR linker, which combines the
// local package's stub data with imported package data to produce a
// complete export data file. It also rewrites the compiler's
// extension data sections based on the results of compilation (e.g.,
// the function inlining cost and linker symbol index assignments).
//
// TODO(mdempsky): Using the name "linker" here is confusing, because
// readers are likely to mistake references to it for cmd/link. But
// there's a shortage of good names for "something that combines
// multiple parts into a cohesive whole"... e.g., "assembler" and
// "compiler" are also already taken.

type linker struct {
	pw pkgEncoder

	pkgs  map[string]int
	decls map[*types.Sym]int
}

func (l *linker) relocAll(pr *pkgReader, relocs []relocEnt) []relocEnt {
	res := make([]relocEnt, len(relocs))
	for i, rent := range relocs {
		rent.idx = l.relocIdx(pr, rent.kind, rent.idx)
		res[i] = rent
	}
	return res
}

func (l *linker) relocIdx(pr *pkgReader, k reloc, idx int) int {
	assert(pr != nil)

	absIdx := pr.absIdx(k, idx)

	if newidx := pr.newindex[absIdx]; newidx != 0 {
		return ^newidx
	}

	var newidx int
	switch k {
	case relocString:
		newidx = l.relocString(pr, idx)
	case relocPkg:
		newidx = l.relocPkg(pr, idx)
	case relocObj:
		newidx = l.relocObj(pr, idx)

	default:
		// Generic relocations.
		//
		// TODO(mdempsky): Deduplicate more sections? In fact, I think
		// every section could be deduplicated. This would also be easier
		// if we do external relocations.

		w := l.pw.newEncoderRaw(k)
		l.relocCommon(pr, &w, k, idx)
		newidx = w.idx
	}

	pr.newindex[absIdx] = ^newidx

	return newidx
}

func (l *linker) relocString(pr *pkgReader, idx int) int {
	return l.pw.stringIdx(pr.stringIdx(idx))
}

func (l *linker) relocPkg(pr *pkgReader, idx int) int {
	path := pr.peekPkgPath(idx)

	if newidx, ok := l.pkgs[path]; ok {
		return newidx
	}

	r := pr.newDecoder(relocPkg, idx, syncPkgDef)
	w := l.pw.newEncoder(relocPkg, syncPkgDef)
	l.pkgs[path] = w.idx

	// TODO(mdempsky): We end up leaving an empty string reference here
	// from when the package was originally written as "". Probably not
	// a big deal, but a little annoying. Maybe relocating
	// cross-references in place is the way to go after all.
	w.relocs = l.relocAll(pr, r.relocs)

	_ = r.string() // original path
	w.string(path)

	io.Copy(&w.data, &r.data)

	return w.flush()
}

func (l *linker) relocObj(pr *pkgReader, idx int) int {
	path, name, tag := pr.peekObj(idx)
	sym := types.NewPkg(path, "").Lookup(name)

	if newidx, ok := l.decls[sym]; ok {
		return newidx
	}

	if tag == objStub && path != "builtin" && path != "unsafe" {
		pri, ok := objReader[sym]
		if !ok {
			base.Fatalf("missing reader for %q.%v", path, name)
		}
		assert(ok)

		pr = pri.pr
		idx = pri.idx

		path2, name2, tag2 := pr.peekObj(idx)
		sym2 := types.NewPkg(path2, "").Lookup(name2)
		assert(sym == sym2)
		assert(tag2 != objStub)
	}

	w := l.pw.newEncoderRaw(relocObj)
	wext := l.pw.newEncoderRaw(relocObjExt)
	wname := l.pw.newEncoderRaw(relocName)
	wdict := l.pw.newEncoderRaw(relocObjDict)

	l.decls[sym] = w.idx
	assert(wext.idx == w.idx)
	assert(wname.idx == w.idx)
	assert(wdict.idx == w.idx)

	l.relocCommon(pr, &w, relocObj, idx)
	l.relocCommon(pr, &wname, relocName, idx)
	l.relocCommon(pr, &wdict, relocObjDict, idx)

	var obj *ir.Name
	if path == "" {
		var ok bool
		obj, ok = sym.Def.(*ir.Name)

		// Generic types and functions and declared constraint types won't
		// have definitions.
		// For now, just generically copy their extension data.
		// TODO(mdempsky): Restore assertion.
		if !ok && false {
			base.Fatalf("missing definition for %v", sym)
		}
	}

	if obj != nil {
		wext.sync(syncObject1)
		switch tag {
		case objFunc:
			l.relocFuncExt(&wext, obj)
		case objType:
			l.relocTypeExt(&wext, obj)
		case objVar:
			l.relocVarExt(&wext, obj)
		}
		wext.flush()
	} else {
		l.relocCommon(pr, &wext, relocObjExt, idx)
	}

	return w.idx
}

func (l *linker) relocCommon(pr *pkgReader, w *encoder, k reloc, idx int) {
	r := pr.newDecoderRaw(k, idx)
	w.relocs = l.relocAll(pr, r.relocs)
	io.Copy(&w.data, &r.data)
	w.flush()
}

func (l *linker) pragmaFlag(w *encoder, pragma ir.PragmaFlag) {
	w.sync(syncPragma)
	w.int(int(pragma))
}

func (l *linker) relocFuncExt(w *encoder, name *ir.Name) {
	w.sync(syncFuncExt)

	l.pragmaFlag(w, name.Func.Pragma)
	l.linkname(w, name)

	// Relocated extension data.
	w.bool(true)

	// Record definition ABI so cross-ABI calls can be direct.
	// This is important for the performance of calling some
	// common functions implemented in assembly (e.g., bytealg).
	w.uint64(uint64(name.Func.ABI))

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(name.Type()).FieldSlice() {
			w.string(f.Note)
		}
	}

	if inl := name.Func.Inl; w.bool(inl != nil) {
		w.len(int(inl.Cost))
		w.bool(inl.CanDelayResults)

		pri, ok := bodyReader[name.Func]
		assert(ok)
		w.reloc(relocBody, l.relocIdx(pri.pr, relocBody, pri.idx))
	}

	w.sync(syncEOF)
}

func (l *linker) relocTypeExt(w *encoder, name *ir.Name) {
	w.sync(syncTypeExt)

	typ := name.Type()

	l.pragmaFlag(w, name.Pragma())

	// For type T, export the index of type descriptor symbols of T and *T.
	l.lsymIdx(w, "", reflectdata.TypeLinksym(typ))
	l.lsymIdx(w, "", reflectdata.TypeLinksym(typ.PtrTo()))

	if typ.Kind() != types.TINTER {
		for _, method := range typ.Methods().Slice() {
			l.relocFuncExt(w, method.Nname.(*ir.Name))
		}
	}
}

func (l *linker) relocVarExt(w *encoder, name *ir.Name) {
	w.sync(syncVarExt)
	l.linkname(w, name)
}

func (l *linker) linkname(w *encoder, name *ir.Name) {
	w.sync(syncLinkname)

	linkname := name.Sym().Linkname
	if !l.lsymIdx(w, linkname, name.Linksym()) {
		w.string(linkname)
	}
}

func (l *linker) lsymIdx(w *encoder, linkname string, lsym *obj.LSym) bool {
	if lsym.PkgIdx > goobj.PkgIdxSelf || (lsym.PkgIdx == goobj.PkgIdxInvalid && !lsym.Indexed()) || linkname != "" {
		w.int64(-1)
		return false
	}

	// For a defined symbol, export its index.
	// For re-exporting an imported symbol, pass its index through.
	w.int64(int64(lsym.SymIdx))
	return true
}

// @@@ Helpers

// TODO(mdempsky): These should probably be removed. I think they're a
// smell that the export data format is not yet quite right.

func (pr *pkgDecoder) peekPkgPath(idx int) string {
	r := pr.newDecoder(relocPkg, idx, syncPkgDef)
	path := r.string()
	if path == "" {
		path = pr.pkgPath
	}
	return path
}

func (pr *pkgDecoder) peekObj(idx int) (string, string, codeObj) {
	r := pr.newDecoder(relocName, idx, syncObject1)
	r.sync(syncSym)
	r.sync(syncPkg)
	path := pr.peekPkgPath(r.reloc(relocPkg))
	name := r.string()
	assert(name != "")

	tag := codeObj(r.code(syncCodeObj))

	return path, name, tag
}
