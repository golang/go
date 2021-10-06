// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"fmt"
	"internal/goversion"
	"io"
	"runtime"
	"sort"

	"cmd/compile/internal/base"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// localPkgReader holds the package reader used for reading the local
// package. It exists so the unified IR linker can refer back to it
// later.
var localPkgReader *pkgReader

// unified construct the local package's IR from syntax's AST.
//
// The pipeline contains 2 steps:
//
// (1) Generate package export data "stub".
//
// (2) Generate package IR from package export data.
//
// The package data "stub" at step (1) contains everything from the local package,
// but nothing that have been imported. When we're actually writing out export data
// to the output files (see writeNewExport function), we run the "linker", which does
// a few things:
//
// + Updates compiler extensions data (e.g., inlining cost, escape analysis results).
//
// + Handles re-exporting any transitive dependencies.
//
// + Prunes out any unnecessary details (e.g., non-inlineable functions, because any
//   downstream importers only care about inlinable functions).
//
// The source files are typechecked twice, once before writing export data
// using types2 checker, once after read export data using gc/typecheck.
// This duplication of work will go away once we always use types2 checker,
// we can remove the gc/typecheck pass. The reason it is still here:
//
// + It reduces engineering costs in maintaining a fork of typecheck
//   (e.g., no need to backport fixes like CL 327651).
//
// + It makes it easier to pass toolstash -cmp.
//
// + Historically, we would always re-run the typechecker after import, even though
//   we know the imported data is valid. It's not ideal, but also not causing any
//   problem either.
//
// + There's still transformation that being done during gc/typecheck, like rewriting
//   multi-valued function call, or transform ir.OINDEX -> ir.OINDEXMAP.
//
// Using syntax+types2 tree, which already has a complete representation of generics,
// the unified IR has the full typed AST for doing introspection during step (1).
// In other words, we have all necessary information to build the generic IR form
// (see writer.captureVars for an example).
func unified(noders []*noder) {
	inline.NewInline = InlineCall

	if !quirksMode() {
		writeNewExportFunc = writeNewExport
	} else if base.Flag.G != 0 {
		base.Errorf("cannot use -G and -d=quirksmode together")
	}

	newReadImportFunc = func(data string, pkg1 *types.Pkg, ctxt *types2.Context, packages map[string]*types2.Package) (pkg2 *types2.Package, err error) {
		pr := newPkgDecoder(pkg1.Path, data)

		// Read package descriptors for both types2 and compiler backend.
		readPackage(newPkgReader(pr), pkg1)
		pkg2 = readPackage2(ctxt, packages, pr)
		return
	}

	data := writePkgStub(noders)

	// We already passed base.Flag.Lang to types2 to handle validating
	// the user's source code. Bump it up now to the current version and
	// re-parse, so typecheck doesn't complain if we construct IR that
	// utilizes newer Go features.
	base.Flag.Lang = fmt.Sprintf("go1.%d", goversion.Version)
	types.ParseLangFlag()

	assert(types.LocalPkg.Path == "")
	types.LocalPkg.Height = 0 // reset so pkgReader.pkgIdx doesn't complain
	target := typecheck.Target

	typecheck.TypecheckAllowed = true

	localPkgReader = newPkgReader(newPkgDecoder(types.LocalPkg.Path, data))
	readPackage(localPkgReader, types.LocalPkg)

	r := localPkgReader.newReader(relocMeta, privateRootIdx, syncPrivate)
	r.pkgInit(types.LocalPkg, target)

	// Type-check any top-level assignments. We ignore non-assignments
	// here because other declarations are typechecked as they're
	// constructed.
	for i, ndecls := 0, len(target.Decls); i < ndecls; i++ {
		switch n := target.Decls[i]; n.Op() {
		case ir.OAS, ir.OAS2:
			target.Decls[i] = typecheck.Stmt(n)
		}
	}

	// Don't use range--bodyIdx can add closures to todoBodies.
	for len(todoBodies) > 0 {
		// The order we expand bodies doesn't matter, so pop from the end
		// to reduce todoBodies reallocations if it grows further.
		fn := todoBodies[len(todoBodies)-1]
		todoBodies = todoBodies[:len(todoBodies)-1]

		pri, ok := bodyReader[fn]
		assert(ok)
		pri.funcBody(fn)

		// Instantiated generic function: add to Decls for typechecking
		// and compilation.
		if fn.OClosure == nil && len(pri.dict.targs) != 0 {
			target.Decls = append(target.Decls, fn)
		}
	}
	todoBodies = nil
	todoBodiesDone = true

	// Check that nothing snuck past typechecking.
	for _, n := range target.Decls {
		if n.Typecheck() == 0 {
			base.FatalfAt(n.Pos(), "missed typecheck: %v", n)
		}

		// For functions, check that at least their first statement (if
		// any) was typechecked too.
		if fn, ok := n.(*ir.Func); ok && len(fn.Body) != 0 {
			if stmt := fn.Body[0]; stmt.Typecheck() == 0 {
				base.FatalfAt(stmt.Pos(), "missed typecheck: %v", stmt)
			}
		}
	}

	base.ExitIfErrors() // just in case
}

// writePkgStub type checks the given parsed source files,
// writes an export data package stub representing them,
// and returns the result.
func writePkgStub(noders []*noder) string {
	m, pkg, info := checkFiles(noders)

	pw := newPkgWriter(m, pkg, info)

	pw.collectDecls(noders)

	publicRootWriter := pw.newWriter(relocMeta, syncPublic)
	privateRootWriter := pw.newWriter(relocMeta, syncPrivate)

	assert(publicRootWriter.idx == publicRootIdx)
	assert(privateRootWriter.idx == privateRootIdx)

	{
		w := publicRootWriter
		w.pkg(pkg)
		w.bool(false) // has init; XXX

		scope := pkg.Scope()
		names := scope.Names()
		w.len(len(names))
		for _, name := range scope.Names() {
			w.obj(scope.Lookup(name), nil)
		}

		w.sync(syncEOF)
		w.flush()
	}

	{
		w := privateRootWriter
		w.pkgInit(noders)
		w.flush()
	}

	var sb bytes.Buffer // TODO(mdempsky): strings.Builder after #44505 is resolved
	pw.dump(&sb)

	// At this point, we're done with types2. Make sure the package is
	// garbage collected.
	freePackage(pkg)

	return sb.String()
}

// freePackage ensures the given package is garbage collected.
func freePackage(pkg *types2.Package) {
	// The GC test below relies on a precise GC that runs finalizers as
	// soon as objects are unreachable. Our implementation provides
	// this, but other/older implementations may not (e.g., Go 1.4 does
	// not because of #22350). To avoid imposing unnecessary
	// restrictions on the GOROOT_BOOTSTRAP toolchain, we skip the test
	// during bootstrapping.
	if base.CompilerBootstrap {
		return
	}

	// Set a finalizer on pkg so we can detect if/when it's collected.
	done := make(chan struct{})
	runtime.SetFinalizer(pkg, func(*types2.Package) { close(done) })

	// Important: objects involved in cycles are not finalized, so zero
	// out pkg to break its cycles and allow the finalizer to run.
	*pkg = types2.Package{}

	// It typically takes just 1 or 2 cycles to release pkg, but it
	// doesn't hurt to try a few more times.
	for i := 0; i < 10; i++ {
		select {
		case <-done:
			return
		default:
			runtime.GC()
		}
	}

	base.Fatalf("package never finalized")
}

func readPackage(pr *pkgReader, importpkg *types.Pkg) {
	r := pr.newReader(relocMeta, publicRootIdx, syncPublic)

	pkg := r.pkg()
	assert(pkg == importpkg)

	if r.bool() {
		sym := pkg.Lookup(".inittask")
		task := ir.NewNameAt(src.NoXPos, sym)
		task.Class = ir.PEXTERN
		sym.Def = task
	}

	for i, n := 0, r.len(); i < n; i++ {
		r.sync(syncObject)
		assert(!r.bool())
		idx := r.reloc(relocObj)
		assert(r.len() == 0)

		path, name, code := r.p.peekObj(idx)
		if code != objStub {
			objReader[types.NewPkg(path, "").Lookup(name)] = pkgReaderIndex{pr, idx, nil}
		}
	}
}

func writeNewExport(out io.Writer) {
	l := linker{
		pw: newPkgEncoder(),

		pkgs:  make(map[string]int),
		decls: make(map[*types.Sym]int),
	}

	publicRootWriter := l.pw.newEncoder(relocMeta, syncPublic)
	assert(publicRootWriter.idx == publicRootIdx)

	var selfPkgIdx int

	{
		pr := localPkgReader
		r := pr.newDecoder(relocMeta, publicRootIdx, syncPublic)

		r.sync(syncPkg)
		selfPkgIdx = l.relocIdx(pr, relocPkg, r.reloc(relocPkg))

		r.bool() // has init

		for i, n := 0, r.len(); i < n; i++ {
			r.sync(syncObject)
			assert(!r.bool())
			idx := r.reloc(relocObj)
			assert(r.len() == 0)

			xpath, xname, xtag := pr.peekObj(idx)
			assert(xpath == pr.pkgPath)
			assert(xtag != objStub)

			if types.IsExported(xname) {
				l.relocIdx(pr, relocObj, idx)
			}
		}

		r.sync(syncEOF)
	}

	{
		var idxs []int
		for _, idx := range l.decls {
			idxs = append(idxs, idx)
		}
		sort.Ints(idxs)

		w := publicRootWriter

		w.sync(syncPkg)
		w.reloc(relocPkg, selfPkgIdx)

		w.bool(typecheck.Lookup(".inittask").Def != nil)

		w.len(len(idxs))
		for _, idx := range idxs {
			w.sync(syncObject)
			w.bool(false)
			w.reloc(relocObj, idx)
			w.len(0)
		}

		w.sync(syncEOF)
		w.flush()
	}

	l.pw.dump(out)
}
