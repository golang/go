// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/goversion"
	"internal/pkgbits"
	"io"
	"runtime"
	"sort"
	"strings"

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

// unified constructs the local package's Internal Representation (IR)
// from its syntax tree (AST).
//
// The pipeline contains 2 steps:
//
//  1. Generate the export data "stub".
//
//  2. Generate the IR from the export data above.
//
// The package data "stub" at step (1) contains everything from the local package,
// but nothing that has been imported. When we're actually writing out export data
// to the output files (see writeNewExport), we run the "linker", which:
//
//   - Updates compiler extensions data (e.g. inlining cost, escape analysis results).
//
//   - Handles re-exporting any transitive dependencies.
//
//   - Prunes out any unnecessary details (e.g. non-inlineable functions, because any
//     downstream importers only care about inlinable functions).
//
// The source files are typechecked twice: once before writing the export data
// using types2, and again after reading the export data using gc/typecheck.
// The duplication of work will go away once we only use the types2 type checker,
// removing the gc/typecheck step. For now, it is kept because:
//
//   - It reduces the engineering costs in maintaining a fork of typecheck
//     (e.g. no need to backport fixes like CL 327651).
//
//   - It makes it easier to pass toolstash -cmp.
//
//   - Historically, we would always re-run the typechecker after importing a package,
//     even though we know the imported data is valid. It's not ideal, but it's
//     not causing any problems either.
//
//   - gc/typecheck is still in charge of some transformations, such as rewriting
//     multi-valued function calls or transforming ir.OINDEX to ir.OINDEXMAP.
//
// Using the syntax tree with types2, which has a complete representation of generics,
// the unified IR has the full typed AST needed for introspection during step (1).
// In other words, we have all the necessary information to build the generic IR form
// (see writer.captureVars for an example).
func unified(noders []*noder) {
	inline.NewInline = InlineCall

	data := writePkgStub(noders)

	// We already passed base.Flag.Lang to types2 to handle validating
	// the user's source code. Bump it up now to the current version and
	// re-parse, so typecheck doesn't complain if we construct IR that
	// utilizes newer Go features.
	base.Flag.Lang = fmt.Sprintf("go1.%d", goversion.Version)
	types.ParseLangFlag()

	target := typecheck.Target

	typecheck.TypecheckAllowed = true

	localPkgReader = newPkgReader(pkgbits.NewPkgDecoder(types.LocalPkg.Path, data))
	readPackage(localPkgReader, types.LocalPkg, true)

	r := localPkgReader.newReader(pkgbits.RelocMeta, pkgbits.PrivateRootIdx, pkgbits.SyncPrivate)
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

	readBodies(target)

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

// readBodies iteratively expands all pending dictionaries and
// function bodies.
func readBodies(target *ir.Package) {
	// Don't use range--bodyIdx can add closures to todoBodies.
	for {
		// The order we expand dictionaries and bodies doesn't matter, so
		// pop from the end to reduce todoBodies reallocations if it grows
		// further.
		//
		// However, we do at least need to flush any pending dictionaries
		// before reading bodies, because bodies might reference the
		// dictionaries.

		if len(todoDicts) > 0 {
			fn := todoDicts[len(todoDicts)-1]
			todoDicts = todoDicts[:len(todoDicts)-1]
			fn()
			continue
		}

		if len(todoBodies) > 0 {
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

			continue
		}

		break
	}

	todoDicts = nil
	todoBodies = nil
}

// writePkgStub type checks the given parsed source files,
// writes an export data package stub representing them,
// and returns the result.
func writePkgStub(noders []*noder) string {
	m, pkg, info := checkFiles(noders)

	pw := newPkgWriter(m, pkg, info)

	pw.collectDecls(noders)

	publicRootWriter := pw.newWriter(pkgbits.RelocMeta, pkgbits.SyncPublic)
	privateRootWriter := pw.newWriter(pkgbits.RelocMeta, pkgbits.SyncPrivate)

	assert(publicRootWriter.Idx == pkgbits.PublicRootIdx)
	assert(privateRootWriter.Idx == pkgbits.PrivateRootIdx)

	{
		w := publicRootWriter
		w.pkg(pkg)
		w.Bool(false) // TODO(mdempsky): Remove; was "has init"

		scope := pkg.Scope()
		names := scope.Names()
		w.Len(len(names))
		for _, name := range names {
			w.obj(scope.Lookup(name), nil)
		}

		w.Sync(pkgbits.SyncEOF)
		w.Flush()
	}

	{
		w := privateRootWriter
		w.pkgInit(noders)
		w.Flush()
	}

	var sb strings.Builder
	pw.DumpTo(&sb)

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

// readPackage reads package export data from pr to populate
// importpkg.
//
// localStub indicates whether pr is reading the stub export data for
// the local package, as opposed to relocated export data for an
// import.
func readPackage(pr *pkgReader, importpkg *types.Pkg, localStub bool) {
	{
		r := pr.newReader(pkgbits.RelocMeta, pkgbits.PublicRootIdx, pkgbits.SyncPublic)

		pkg := r.pkg()
		base.Assertf(pkg == importpkg, "have package %q (%p), want package %q (%p)", pkg.Path, pkg, importpkg.Path, importpkg)

		r.Bool() // TODO(mdempsky): Remove; was "has init"

		for i, n := 0, r.Len(); i < n; i++ {
			r.Sync(pkgbits.SyncObject)
			assert(!r.Bool())
			idx := r.Reloc(pkgbits.RelocObj)
			assert(r.Len() == 0)

			path, name, code := r.p.PeekObj(idx)
			if code != pkgbits.ObjStub {
				objReader[types.NewPkg(path, "").Lookup(name)] = pkgReaderIndex{pr, idx, nil, nil, nil}
			}
		}

		r.Sync(pkgbits.SyncEOF)
	}

	if !localStub {
		r := pr.newReader(pkgbits.RelocMeta, pkgbits.PrivateRootIdx, pkgbits.SyncPrivate)

		if r.Bool() {
			sym := importpkg.Lookup(".inittask")
			task := ir.NewNameAt(src.NoXPos, sym)
			task.Class = ir.PEXTERN
			sym.Def = task
		}

		for i, n := 0, r.Len(); i < n; i++ {
			path := r.String()
			name := r.String()
			idx := r.Reloc(pkgbits.RelocBody)

			sym := types.NewPkg(path, "").Lookup(name)
			if _, ok := importBodyReader[sym]; !ok {
				importBodyReader[sym] = pkgReaderIndex{pr, idx, nil, nil, nil}
			}
		}

		r.Sync(pkgbits.SyncEOF)
	}
}

// writeUnifiedExport writes to `out` the finalized, self-contained
// Unified IR export data file for the current compilation unit.
func writeUnifiedExport(out io.Writer) {
	l := linker{
		pw: pkgbits.NewPkgEncoder(base.Debug.SyncFrames),

		pkgs:   make(map[string]pkgbits.Index),
		decls:  make(map[*types.Sym]pkgbits.Index),
		bodies: make(map[*types.Sym]pkgbits.Index),
	}

	publicRootWriter := l.pw.NewEncoder(pkgbits.RelocMeta, pkgbits.SyncPublic)
	privateRootWriter := l.pw.NewEncoder(pkgbits.RelocMeta, pkgbits.SyncPrivate)
	assert(publicRootWriter.Idx == pkgbits.PublicRootIdx)
	assert(privateRootWriter.Idx == pkgbits.PrivateRootIdx)

	var selfPkgIdx pkgbits.Index

	{
		pr := localPkgReader
		r := pr.NewDecoder(pkgbits.RelocMeta, pkgbits.PublicRootIdx, pkgbits.SyncPublic)

		r.Sync(pkgbits.SyncPkg)
		selfPkgIdx = l.relocIdx(pr, pkgbits.RelocPkg, r.Reloc(pkgbits.RelocPkg))

		r.Bool() // TODO(mdempsky): Remove; was "has init"

		for i, n := 0, r.Len(); i < n; i++ {
			r.Sync(pkgbits.SyncObject)
			assert(!r.Bool())
			idx := r.Reloc(pkgbits.RelocObj)
			assert(r.Len() == 0)

			xpath, xname, xtag := pr.PeekObj(idx)
			assert(xpath == pr.PkgPath())
			assert(xtag != pkgbits.ObjStub)

			if types.IsExported(xname) {
				l.relocIdx(pr, pkgbits.RelocObj, idx)
			}
		}

		r.Sync(pkgbits.SyncEOF)
	}

	{
		var idxs []pkgbits.Index
		for _, idx := range l.decls {
			idxs = append(idxs, idx)
		}
		sort.Slice(idxs, func(i, j int) bool { return idxs[i] < idxs[j] })

		w := publicRootWriter

		w.Sync(pkgbits.SyncPkg)
		w.Reloc(pkgbits.RelocPkg, selfPkgIdx)
		w.Bool(false) // TODO(mdempsky): Remove; was "has init"

		w.Len(len(idxs))
		for _, idx := range idxs {
			w.Sync(pkgbits.SyncObject)
			w.Bool(false)
			w.Reloc(pkgbits.RelocObj, idx)
			w.Len(0)
		}

		w.Sync(pkgbits.SyncEOF)
		w.Flush()
	}

	{
		type symIdx struct {
			sym *types.Sym
			idx pkgbits.Index
		}
		var bodies []symIdx
		for sym, idx := range l.bodies {
			bodies = append(bodies, symIdx{sym, idx})
		}
		sort.Slice(bodies, func(i, j int) bool { return bodies[i].idx < bodies[j].idx })

		w := privateRootWriter

		w.Bool(typecheck.Lookup(".inittask").Def != nil)

		w.Len(len(bodies))
		for _, body := range bodies {
			w.String(body.sym.Pkg.Path)
			w.String(body.sym.Name)
			w.Reloc(pkgbits.RelocBody, body.idx)
		}

		w.Sync(pkgbits.SyncEOF)
		w.Flush()
	}

	base.Ctxt.Fingerprint = l.pw.DumpTo(out)
}
