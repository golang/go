// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"internal/pkgbits"
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

// TODO(mdempsky): Should linker go into pkgbits? Probably the
// low-level linking details can be moved there, but the logic for
// handling extension data needs to stay in the compiler.

type linker struct {
	pw pkgbits.PkgEncoder

	pkgs  map[string]int
	decls map[*types.Sym]int
}

func (l *linker) relocAll(pr *pkgReader, relocs []pkgbits.RelocEnt) []pkgbits.RelocEnt {
	res := make([]pkgbits.RelocEnt, len(relocs))
	for i, rent := range relocs {
		rent.Idx = l.relocIdx(pr, rent.Kind, rent.Idx)
		res[i] = rent
	}
	return res
}

func (l *linker) relocIdx(pr *pkgReader, k pkgbits.RelocKind, idx int) int {
	assert(pr != nil)

	absIdx := pr.AbsIdx(k, idx)

	if newidx := pr.newindex[absIdx]; newidx != 0 {
		return ^newidx
	}

	var newidx int
	switch k {
	case pkgbits.RelocString:
		newidx = l.relocString(pr, idx)
	case pkgbits.RelocPkg:
		newidx = l.relocPkg(pr, idx)
	case pkgbits.RelocObj:
		newidx = l.relocObj(pr, idx)

	default:
		// Generic relocations.
		//
		// TODO(mdempsky): Deduplicate more sections? In fact, I think
		// every section could be deduplicated. This would also be easier
		// if we do external relocations.

		w := l.pw.NewEncoderRaw(k)
		l.relocCommon(pr, &w, k, idx)
		newidx = w.Idx
	}

	pr.newindex[absIdx] = ^newidx

	return newidx
}

func (l *linker) relocString(pr *pkgReader, idx int) int {
	return l.pw.StringIdx(pr.StringIdx(idx))
}

func (l *linker) relocPkg(pr *pkgReader, idx int) int {
	path := pr.PeekPkgPath(idx)

	if newidx, ok := l.pkgs[path]; ok {
		return newidx
	}

	r := pr.NewDecoder(pkgbits.RelocPkg, idx, pkgbits.SyncPkgDef)
	w := l.pw.NewEncoder(pkgbits.RelocPkg, pkgbits.SyncPkgDef)
	l.pkgs[path] = w.Idx

	// TODO(mdempsky): We end up leaving an empty string reference here
	// from when the package was originally written as "". Probably not
	// a big deal, but a little annoying. Maybe relocating
	// cross-references in place is the way to go after all.
	w.Relocs = l.relocAll(pr, r.Relocs)

	_ = r.String() // original path
	w.String(path)

	io.Copy(&w.Data, &r.Data)

	return w.Flush()
}

func (l *linker) relocObj(pr *pkgReader, idx int) int {
	path, name, tag := pr.PeekObj(idx)
	sym := types.NewPkg(path, "").Lookup(name)

	if newidx, ok := l.decls[sym]; ok {
		return newidx
	}

	if tag == pkgbits.ObjStub && path != "builtin" && path != "unsafe" {
		pri, ok := objReader[sym]
		if !ok {
			base.Fatalf("missing reader for %q.%v", path, name)
		}
		assert(ok)

		pr = pri.pr
		idx = pri.idx

		path2, name2, tag2 := pr.PeekObj(idx)
		sym2 := types.NewPkg(path2, "").Lookup(name2)
		assert(sym == sym2)
		assert(tag2 != pkgbits.ObjStub)
	}

	w := l.pw.NewEncoderRaw(pkgbits.RelocObj)
	wext := l.pw.NewEncoderRaw(pkgbits.RelocObjExt)
	wname := l.pw.NewEncoderRaw(pkgbits.RelocName)
	wdict := l.pw.NewEncoderRaw(pkgbits.RelocObjDict)

	l.decls[sym] = w.Idx
	assert(wext.Idx == w.Idx)
	assert(wname.Idx == w.Idx)
	assert(wdict.Idx == w.Idx)

	l.relocCommon(pr, &w, pkgbits.RelocObj, idx)
	l.relocCommon(pr, &wname, pkgbits.RelocName, idx)
	l.relocCommon(pr, &wdict, pkgbits.RelocObjDict, idx)

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
		wext.Sync(pkgbits.SyncObject1)
		switch tag {
		case pkgbits.ObjFunc:
			l.relocFuncExt(&wext, obj)
		case pkgbits.ObjType:
			l.relocTypeExt(&wext, obj)
		case pkgbits.ObjVar:
			l.relocVarExt(&wext, obj)
		}
		wext.Flush()
	} else {
		l.relocCommon(pr, &wext, pkgbits.RelocObjExt, idx)
	}

	return w.Idx
}

func (l *linker) relocCommon(pr *pkgReader, w *pkgbits.Encoder, k pkgbits.RelocKind, idx int) {
	r := pr.NewDecoderRaw(k, idx)
	w.Relocs = l.relocAll(pr, r.Relocs)
	io.Copy(&w.Data, &r.Data)
	w.Flush()
}

func (l *linker) pragmaFlag(w *pkgbits.Encoder, pragma ir.PragmaFlag) {
	w.Sync(pkgbits.SyncPragma)
	w.Int(int(pragma))
}

func (l *linker) relocFuncExt(w *pkgbits.Encoder, name *ir.Name) {
	w.Sync(pkgbits.SyncFuncExt)

	l.pragmaFlag(w, name.Func.Pragma)
	l.linkname(w, name)

	// Relocated extension data.
	w.Bool(true)

	// Record definition ABI so cross-ABI calls can be direct.
	// This is important for the performance of calling some
	// common functions implemented in assembly (e.g., bytealg).
	w.Uint64(uint64(name.Func.ABI))

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(name.Type()).FieldSlice() {
			w.String(f.Note)
		}
	}

	if inl := name.Func.Inl; w.Bool(inl != nil) {
		w.Len(int(inl.Cost))
		w.Bool(inl.CanDelayResults)

		pri, ok := bodyReader[name.Func]
		assert(ok)
		w.Reloc(pkgbits.RelocBody, l.relocIdx(pri.pr, pkgbits.RelocBody, pri.idx))
	}

	w.Sync(pkgbits.SyncEOF)
}

func (l *linker) relocTypeExt(w *pkgbits.Encoder, name *ir.Name) {
	w.Sync(pkgbits.SyncTypeExt)

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

func (l *linker) relocVarExt(w *pkgbits.Encoder, name *ir.Name) {
	w.Sync(pkgbits.SyncVarExt)
	l.linkname(w, name)
}

func (l *linker) linkname(w *pkgbits.Encoder, name *ir.Name) {
	w.Sync(pkgbits.SyncLinkname)

	linkname := name.Sym().Linkname
	if !l.lsymIdx(w, linkname, name.Linksym()) {
		w.String(linkname)
	}
}

func (l *linker) lsymIdx(w *pkgbits.Encoder, linkname string, lsym *obj.LSym) bool {
	if lsym.PkgIdx > goobj.PkgIdxSelf || (lsym.PkgIdx == goobj.PkgIdxInvalid && !lsym.Indexed()) || linkname != "" {
		w.Int64(-1)
		return false
	}

	// For a defined symbol, export its index.
	// For re-exporting an imported symbol, pass its index through.
	w.Int64(int64(lsym.SymIdx))
	return true
}
