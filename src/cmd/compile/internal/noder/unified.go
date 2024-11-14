// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/pkgbits"
	"internal/types/errors"
	"io"
	"runtime"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// localPkgReader holds the package reader used for reading the local
// package. It exists so the unified IR linker can refer back to it
// later.
var localPkgReader *pkgReader

// LookupFunc returns the ir.Func for an arbitrary full symbol name if
// that function exists in the set of available export data.
//
// This allows lookup of arbitrary functions and methods that aren't otherwise
// referenced by the local package and thus haven't been read yet.
//
// TODO(prattmic): Does not handle instantiation of generic types. Currently
// profiles don't contain the original type arguments, so we won't be able to
// create the runtime dictionaries.
//
// TODO(prattmic): Hit rate of this function is usually fairly low, and errors
// are only used when debug logging is enabled. Consider constructing cheaper
// errors by default.
func LookupFunc(fullName string) (*ir.Func, error) {
	pkgPath, symName, err := ir.ParseLinkFuncName(fullName)
	if err != nil {
		return nil, fmt.Errorf("error parsing symbol name %q: %v", fullName, err)
	}

	pkg, ok := types.PkgMap()[pkgPath]
	if !ok {
		return nil, fmt.Errorf("pkg %s doesn't exist in %v", pkgPath, types.PkgMap())
	}

	// Symbol naming is ambiguous. We can't necessarily distinguish between
	// a method and a closure. e.g., is foo.Bar.func1 a closure defined in
	// function Bar, or a method on type Bar? Thus we must simply attempt
	// to lookup both.

	fn, err := lookupFunction(pkg, symName)
	if err == nil {
		return fn, nil
	}

	fn, mErr := lookupMethod(pkg, symName)
	if mErr == nil {
		return fn, nil
	}

	return nil, fmt.Errorf("%s is not a function (%v) or method (%v)", fullName, err, mErr)
}

// PostLookupCleanup performs cleanup operations needed
// after a series of calls to LookupFunc, specifically invoking
// readBodies to post-process any funcs on the "todoBodies" list
// that were added as a result of the lookup operations.
func PostLookupCleanup() {
	readBodies(typecheck.Target, false)
}

func lookupFunction(pkg *types.Pkg, symName string) (*ir.Func, error) {
	sym := pkg.Lookup(symName)

	// TODO(prattmic): Enclosed functions (e.g., foo.Bar.func1) are not
	// present in objReader, only as OCLOSURE nodes in the enclosing
	// function.
	pri, ok := objReader[sym]
	if !ok {
		return nil, fmt.Errorf("func sym %v missing objReader", sym)
	}

	node, err := pri.pr.objIdxMayFail(pri.idx, nil, nil, false)
	if err != nil {
		return nil, fmt.Errorf("func sym %v lookup error: %w", sym, err)
	}
	name := node.(*ir.Name)
	if name.Op() != ir.ONAME || name.Class != ir.PFUNC {
		return nil, fmt.Errorf("func sym %v refers to non-function name: %v", sym, name)
	}
	return name.Func, nil
}

func lookupMethod(pkg *types.Pkg, symName string) (*ir.Func, error) {
	// N.B. readPackage creates a Sym for every object in the package to
	// initialize objReader and importBodyReader, even if the object isn't
	// read.
	//
	// However, objReader is only initialized for top-level objects, so we
	// must first lookup the type and use that to find the method rather
	// than looking for the method directly.
	typ, meth, err := ir.LookupMethodSelector(pkg, symName)
	if err != nil {
		return nil, fmt.Errorf("error looking up method symbol %q: %v", symName, err)
	}

	pri, ok := objReader[typ]
	if !ok {
		return nil, fmt.Errorf("type sym %v missing objReader", typ)
	}

	node, err := pri.pr.objIdxMayFail(pri.idx, nil, nil, false)
	if err != nil {
		return nil, fmt.Errorf("func sym %v lookup error: %w", typ, err)
	}
	name := node.(*ir.Name)
	if name.Op() != ir.OTYPE {
		return nil, fmt.Errorf("type sym %v refers to non-type name: %v", typ, name)
	}
	if name.Alias() {
		return nil, fmt.Errorf("type sym %v refers to alias", typ)
	}
	if name.Type().IsInterface() {
		return nil, fmt.Errorf("type sym %v refers to interface type", typ)
	}

	for _, m := range name.Type().Methods() {
		if m.Sym == meth {
			fn := m.Nname.(*ir.Name).Func
			return fn, nil
		}
	}

	return nil, fmt.Errorf("method %s missing from method set of %v", symName, typ)
}

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
func unified(m posMap, noders []*noder) {
	inline.InlineCall = unifiedInlineCall
	typecheck.HaveInlineBody = unifiedHaveInlineBody
	pgoir.LookupFunc = LookupFunc
	pgoir.PostLookupCleanup = PostLookupCleanup

	data := writePkgStub(m, noders)

	target := typecheck.Target

	localPkgReader = newPkgReader(pkgbits.NewPkgDecoder(types.LocalPkg.Path, data))
	readPackage(localPkgReader, types.LocalPkg, true)

	r := localPkgReader.newReader(pkgbits.RelocMeta, pkgbits.PrivateRootIdx, pkgbits.SyncPrivate)
	r.pkgInit(types.LocalPkg, target)

	readBodies(target, false)

	// Check that nothing snuck past typechecking.
	for _, fn := range target.Funcs {
		if fn.Typecheck() == 0 {
			base.FatalfAt(fn.Pos(), "missed typecheck: %v", fn)
		}

		// For functions, check that at least their first statement (if
		// any) was typechecked too.
		if len(fn.Body) != 0 {
			if stmt := fn.Body[0]; stmt.Typecheck() == 0 {
				base.FatalfAt(stmt.Pos(), "missed typecheck: %v", stmt)
			}
		}
	}

	// For functions originally came from package runtime,
	// mark as norace to prevent instrumenting, see issue #60439.
	for _, fn := range target.Funcs {
		if !base.Flag.CompilingRuntime && types.RuntimeSymName(fn.Sym()) != "" {
			fn.Pragma |= ir.Norace
		}
	}

	base.ExitIfErrors() // just in case
}

// readBodies iteratively expands all pending dictionaries and
// function bodies.
//
// If duringInlining is true, then the inline.InlineDecls is called as
// necessary on instantiations of imported generic functions, so their
// inlining costs can be computed.
func readBodies(target *ir.Package, duringInlining bool) {
	var inlDecls []*ir.Func

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
				// cmd/link does not support a type symbol referencing a method symbol
				// across DSO boundary, so force re-compiling methods on a generic type
				// even it was seen from imported package in linkshared mode, see #58966.
				canSkipNonGenericMethod := !(base.Ctxt.Flag_linkshared && ir.IsMethod(fn))
				if duringInlining && canSkipNonGenericMethod {
					inlDecls = append(inlDecls, fn)
				} else {
					target.Funcs = append(target.Funcs, fn)
				}
			}

			continue
		}

		break
	}

	todoDicts = nil
	todoBodies = nil

	if len(inlDecls) != 0 {
		// If we instantiated any generic functions during inlining, we need
		// to call CanInline on them so they'll be transitively inlined
		// correctly (#56280).
		//
		// We know these functions were already compiled in an imported
		// package though, so we don't need to actually apply InlineCalls or
		// save the function bodies any further than this.
		//
		// We can also lower the -m flag to 0, to suppress duplicate "can
		// inline" diagnostics reported against the imported package. Again,
		// we already reported those diagnostics in the original package, so
		// it's pointless repeating them here.

		oldLowerM := base.Flag.LowerM
		base.Flag.LowerM = 0
		inline.CanInlineFuncs(inlDecls, nil)
		base.Flag.LowerM = oldLowerM

		for _, fn := range inlDecls {
			fn.Body = nil // free memory
		}
	}
}

// writePkgStub type checks the given parsed source files,
// writes an export data package stub representing them,
// and returns the result.
func writePkgStub(m posMap, noders []*noder) string {
	pkg, info, otherInfo := checkFiles(m, noders)

	pw := newPkgWriter(m, pkg, info, otherInfo)

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
	if base.CompilerBootstrap || base.Debug.GCCheck == 0 {
		*pkg = types2.Package{}
		return
	}

	// Set a finalizer on pkg so we can detect if/when it's collected.
	done := make(chan struct{})
	runtime.SetFinalizer(pkg, func { close(done) })

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
		if pkg != importpkg {
			base.ErrorfAt(base.AutogeneratedPos, errors.BadImportPath, "mismatched import path, have %q (%p), want %q (%p)", pkg.Path, pkg, importpkg.Path, importpkg)
			base.ErrorExit()
		}

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
			task := ir.NewNameAt(src.NoXPos, sym, nil)
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
		sort.Slice(idxs, func { i, j -> idxs[i] < idxs[j] })

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
		sort.Slice(bodies, func { i, j -> bodies[i].idx < bodies[j].idx })

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
