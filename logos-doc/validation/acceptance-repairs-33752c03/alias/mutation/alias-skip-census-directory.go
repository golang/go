// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"context"
	"errors"
	"fmt"
	"go/build"
	"internal/logosfile"
	"maps"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
)

// IsLogos reports whether p is a loGo/s package: its first known source
// file has a loGo/s extension.
//
// When GoFiles is empty the kind falls back across CgoFiles, TestGoFiles,
// and XTestGoFiles in that order, so test-only and cgo-only packages keep
// their kind instead of silently reading as Go. In plain go mode .lgo files
// never reach these lists, so ordinary Go packages are unaffected.
func (p *Package) IsLogos() bool {
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		if len(files) > 0 {
			return logosfile.Is(files[0])
		}
	}
	return false
}

func logosScrubGoFactsFlags(flags []string) []string {
	kept := make([]string, 0, len(flags))
	for _, f := range flags {
		if f == "-logos.gofacts" || strings.HasPrefix(f, "-logos.gofacts=") {
			continue
		}
		kept = append(kept, f)
	}
	return kept
}

// moduleKinds separates a module's source census from its loaded packages.
// Only a loaded, error-free package can carry a diagnostic to the caller.
type moduleKinds struct {
	goPkgs    map[string]bool
	logosPkgs map[string]bool
	module    module.Version
	carrier   *Package
}

// logosModuleError identifies a failed module census while retaining its
// diagnostic and cause. Unlike ordinary dependency errors, a failed census
// must also make default logos list reject the selected package graph.
type logosModuleError struct{ err error }

func (e *logosModuleError) Error() string { return e.err.Error() }

func (e *logosModuleError) Unwrap() error { return e.err }

// ReportLogosModuleDependencyErrors reports census failures below roots.
// The caller reports errors on roots and owns the exit status.
func ReportLogosModuleDependencyErrors(roots []*Package) {
	if !cfg.Logos {
		return
	}
	selected := make(map[*Package]bool, len(roots))
	for _, p := range roots {
		selected[p] = true
	}
	for _, p := range PackageList(roots) {
		if selected[p] || p.Error == nil {
			continue
		}
		if _, ok := errors.AsType[*logosModuleError](p.Error.Err); ok {
			DefaultPrinter().Errorf(p, "%v", p.Error)
		}
	}
}

func (g *moduleKinds) add(p *Package) {
	if p.IsLogos() {
		g.logosPkgs[p.ImportPath] = true
	} else {
		g.goPkgs[p.ImportPath] = true
	}
	if p.Error == nil && (g.carrier == nil || p.ImportPath < g.carrier.ImportPath) {
		g.carrier = p
	}
}

// checkLogosModules reports [correctness::mixed_module] once per module
// whose active packages span both kinds, including unselected siblings.
// The error rides the existing
// per-package error mechanism on the lexicographically first error-free
// package of the mixed module. The generated test main is exempt from the
// census. Packages outside any module (standard library,
// command-line-arguments without a resolved module) and packages with no
// source files are skipped.
func checkLogosModules(ctx context.Context, ld *modload.Loader, all []*Package) {
	if !cfg.Logos {
		return
	}
	byModule := censusModuleKinds(all)
	for _, modPath := range slices.Sorted(maps.Keys(byModule)) {
		g := byModule[modPath]
		if g.carrier == nil {
			continue
		}
		if err := scanModuleKinds(ctx, ld, g); err != nil {
			g.carrier.Error = &PackageError{Err: &logosModuleError{err: err}}
			g.carrier.Incomplete = true
			continue
		}
		reportMixedModule(modPath, g)
	}
}

func scanModuleKinds(ctx context.Context, ld *modload.Loader, g *moduleKinds) error {
	m := g.module
	dirs, err := modload.LogosModulePackageDirs(ctx, ld, m)
	if err != nil {
		return fmt.Errorf("classifying packages in module %s: %w", m.Path, err)
	}
	for _, name := range slices.Sorted(maps.Keys(dirs)) {
		if g.goPkgs[name] || g.logosPkgs[name] {
			continue
		}
		p, err := cfg.BuildContext.ImportDir(dirs[name], 0)
		if err != nil {
			if _, noGo := errors.AsType[*build.NoGoError](err); noGo {
				continue
			}
			if _, duplicate := errors.AsType[logosfile.DuplicateStemError](err); !duplicate {
				return fmt.Errorf("classifying package %s in module %s: %w", name, m.Path, err)
			}
			continue
			// Duplicate stems invalidate this package, but its complete file
			// lists must still contribute to the module's source kinds.
		}
		for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
			for _, file := range files {
				if logosfile.Is(file) {
					g.logosPkgs[name] = true
				} else {
					g.goPkgs[name] = true
				}
			}
		}
	}
	return nil
}

// addLogosRTForGoImporter loads the implicit logos/rt import for a Go
// package directly importing a loGo/s package (M3-S7, beside the M1-S4
// needsLogosRT hook in pkg.go). It runs during load after the package's
// own imports resolve, so the rt archive joins the importcfg; plain Go
// packages without loGo/s imports keep no new dependency.
func addLogosRTForGoImporter(ld *modload.Loader, ctx context.Context, opts PackageOpts, p *Package) {
	if p == nil || p.IsLogos() || !cfg.Logos {
		return
	}
	hasLogosDep := false
	for _, q := range p.Internal.Imports {
		if q != nil && q.IsLogos() {
			hasLogosDep = true
			break
		}
	}
	if !hasLogosDep {
		return
	}
	for _, dep := range p.Internal.CompiledImports {
		if dep == LogosRTPkg {
			return
		}
	}
	for _, q := range p.Internal.Imports {
		if q != nil && q.ImportPath == LogosRTPkg {
			return
		}
	}
	nstk := ImportStack{{Pkg: p.ImportPath}}
	p1, err := loadImport(ld, ctx, opts, nil, LogosRTPkg, p.Dir, p, &nstk, nil, ResolveImport)
	if err != nil && p.Error == nil {
		p.Error = err
	}
	if err != nil || p1.Incomplete {
		p.Incomplete = true
	}
	p.Internal.Imports = append(p.Internal.Imports, p1)
	p.Internal.CompiledImports = append(p.Internal.CompiledImports, p1.ImportPath)
}

// hasSourceFiles reports whether any source list of p holds a file.
// Packages with no files carry no kind and stay out of the census, so a
// stray empty package cannot report a false mixed module.
func hasSourceFiles(p *Package) bool {
	return len(p.GoFiles)+len(p.CgoFiles)+len(p.TestGoFiles)+len(p.XTestGoFiles) > 0
}

// censusModuleKinds groups loaded packages by module path.
func censusModuleKinds(all []*Package) map[string]*moduleKinds {
	byModule := make(map[string]*moduleKinds)
	for _, p := range all {
		if p.Internal.TestmainGo != nil || p.Module == nil || !hasSourceFiles(p) {
			continue
		}
		g := byModule[p.Module.Path]
		if g == nil {
			g = &moduleKinds{
				goPkgs:    make(map[string]bool),
				logosPkgs: make(map[string]bool),
				module:    module.Version{Path: p.Module.Path, Version: p.Module.Version},
			}
			byModule[p.Module.Path] = g
		}
		g.add(p)
	}
	return byModule
}

// reportMixedModule records the mixed-module error when a module holds
// packages of both kinds. Package names are sorted for determinism.
func reportMixedModule(modPath string, g *moduleKinds) {
	if len(g.goPkgs) == 0 || len(g.logosPkgs) == 0 {
		return
	}
	goNames := slices.Sorted(maps.Keys(g.goPkgs))
	logosNames := slices.Sorted(maps.Keys(g.logosPkgs))
	carrier := g.carrier
	if carrier == nil {
		return
	}
	carrier.Error = &PackageError{
		Err: &logosModuleError{err: fmt.Errorf("[correctness::mixed_module] module %s has Go packages (%s) and loGo/s packages (%s)", modPath, strings.Join(goNames, ", "), strings.Join(logosNames, ", "))},
	}
	carrier.Incomplete = true
}

// needsLogosRT decides the implicit logos/rt import: every logos package
// gets it, so lowered code can call TraceErr, Join, and the helpers later
// milestones add, even when no source file names the import. A Go main
// package gets it only when its build emits the wasm host loader (SPEC Sec
// 14.2), so its //go:wasmexport functions share the loader helpers; other
// plain Go packages never get it. M11 appends logos/rt/json and M14 appends
// simd/archsimd after this import, in that fixed order.
func needsLogosRT(p *Package) bool {
	return p.IsLogos() || EmitsWasmLoader(p)
}

// LogosRTPkg is the import path of the loGo/s support package implicitly
// imported by every logos package.
const LogosRTPkg = "logos/rt"

// LogosSimdPkg is the import path implicitly imported by every logos
// package when GOEXPERIMENT=simd, so lowered fixed-array SIMD code can
// call into it without naming it (M14-S1).
const LogosSimdPkg = "simd/archsimd"

// LogosJSONPkg is the support package of generated static-JSON methods
// (M11). The compiler imports it only in files whose lowering generated
// MarshalJSON or UnmarshalJSON, but the import is synthesized after cmd/go
// built the import list, so every logos package carries it and unused
// entries are harmless.
const LogosJSONPkg = "logos/rt/json"

// needsLogosJSON reports whether p needs the implicit logos/rt/json
// import: every logos package, mirroring needsLogosRT.
func needsLogosJSON(p *Package) bool {
	return p.IsLogos()
}

// needsLogosSimd reports whether p needs the implicit simd/archsimd
// import: every logos package gets it when the simd experiment is on
// for amd64 or arm64, the only targets whose lowering calls archsimd
// (SPEC T-204.2). Other targets keep scalar code and no archsimd import.
func needsLogosSimd(p *Package) bool {
	if !p.IsLogos() || !cfg.Experiment.SIMD {
		return false
	}
	return cfg.BuildContext.GOARCH == "amd64" || cfg.BuildContext.GOARCH == "arm64"
}

// needsLogosInterp reports whether p needs the implicit strconv and
// unsafe imports: every logos package gets them, so lowered
// interpolations can append without naming them (M6-S1). Plain Go
// packages never get them.
func needsLogosInterp(p *Package) bool {
	return p.IsLogos()
}

// addLogosRTXTestImport loads the implicit LogosRTPkg import for the
// external test package of a logos package. The base package gains the
// import in load and the internal test package inherits it, but the
// external test package is built by hand in TestPackagesAndErrors, so it
// records the load on the test package itself: the error, the incomplete
// flag, and the import lists. Plain Go packages never take this path.
func addLogosRTXTestImport(ld *modload.Loader, ctx context.Context, opts PackageOpts, pre *preload, pxtest *Package) {
	if pxtest == nil || !needsLogosRT(pxtest) {
		return
	}
	for _, q := range pxtest.Internal.Imports {
		if q.ImportPath == LogosRTPkg {
			addLogosInterpXTestImport(ld, ctx, opts, pre, pxtest)
			addLogosSimdXTestImport(ld, ctx, opts, pre, pxtest)
			addLogosJSONXTestImport(ld, ctx, opts, pre, pxtest)
			addLogosBinaryXTestImport(ld, ctx, opts, pre, pxtest)
			return
		}
	}
	stk := ImportStack{{Pkg: pxtest.ImportPath}}
	p1, err := loadImport(ld, ctx, opts, pre, LogosRTPkg, pxtest.Dir, pxtest, &stk, nil, ResolveImport)
	if err != nil && pxtest.Error == nil {
		pxtest.Error = err
	}
	if err != nil || p1.Incomplete {
		pxtest.Incomplete = true
	}
	pxtest.Internal.Imports = append(pxtest.Internal.Imports, p1)
	pxtest.Internal.CompiledImports = append(pxtest.Internal.CompiledImports, p1.ImportPath)
	addLogosInterpXTestImport(ld, ctx, opts, pre, pxtest)
	addLogosSimdXTestImport(ld, ctx, opts, pre, pxtest)
	addLogosJSONXTestImport(ld, ctx, opts, pre, pxtest)
	addLogosBinaryXTestImport(ld, ctx, opts, pre, pxtest)
}

// addLogosSimdXTestImport loads the implicit simd/archsimd import for
// the external test package of a logos package (M14-S1), mirroring
// addLogosRTXTestImport: lowered array operators in external test
// files need the same archive under GOEXPERIMENT=simd.
func addLogosSimdXTestImport(ld *modload.Loader, ctx context.Context, opts PackageOpts, pre *preload, pxtest *Package) {
	if pxtest == nil || !needsLogosSimd(pxtest) {
		return
	}
	for _, q := range pxtest.Internal.Imports {
		if q.ImportPath == LogosSimdPkg {
			return
		}
	}
	stk := ImportStack{{Pkg: pxtest.ImportPath}}
	p1, err := loadImport(ld, ctx, opts, pre, LogosSimdPkg, pxtest.Dir, pxtest, &stk, nil, ResolveImport)
	if err != nil && pxtest.Error == nil {
		pxtest.Error = err
	}
	if err != nil || p1.Incomplete {
		pxtest.Incomplete = true
	}
	pxtest.Internal.Imports = append(pxtest.Internal.Imports, p1)
	pxtest.Internal.CompiledImports = append(pxtest.Internal.CompiledImports, p1.ImportPath)
}

// addLogosInterpXTestImport loads the implicit strconv and unsafe
// imports for the external test package of a logos package (M6-S1),
// mirroring addLogosRTXTestImport: lowered interpolations in external
// test files need the same archives.
func addLogosInterpXTestImport(ld *modload.Loader, ctx context.Context, opts PackageOpts, pre *preload, pxtest *Package) {
	if pxtest == nil || !needsLogosInterp(pxtest) {
		return
	}
	for _, path := range []string{"strconv", "unsafe"} {
		present := false
		for _, q := range pxtest.Internal.Imports {
			if q.ImportPath == path {
				present = true
				break
			}
		}
		if present {
			continue
		}
		stk := ImportStack{{Pkg: pxtest.ImportPath}}
		p1, err := loadImport(ld, ctx, opts, pre, path, pxtest.Dir, pxtest, &stk, nil, ResolveImport)
		if err != nil && pxtest.Error == nil {
			pxtest.Error = err
		}
		if err != nil || p1.Incomplete {
			pxtest.Incomplete = true
		}
		pxtest.Internal.Imports = append(pxtest.Internal.Imports, p1)
		pxtest.Internal.CompiledImports = append(pxtest.Internal.CompiledImports, p1.ImportPath)
	}
}

// needsLogosBinary reports whether p needs the implicit binary-helper
// imports: every logos package, mirroring needsLogosJSON. The nojsonv2
// fallback synthesizes MarshalBinary helpers that directly import fmt,
// encoding/binary, math and slices (json_binary.go binaryImportTable);
// unsafe already rides the interp edge. jsonv2 mode leaves them unused
// and harmless. Plain Go packages never take this path.
func needsLogosBinary(p *Package) bool {
	return p.IsLogos()
}

// addLogosJSONXTestImport loads the implicit logos/rt/json import for
// the external test package of a logos package (M11), mirroring
// addLogosRTXTestImport: generated static-JSON methods in external
// test files need the same archive. Plain Go packages never take it.
func addLogosJSONXTestImport(ld *modload.Loader, ctx context.Context, opts PackageOpts, pre *preload, pxtest *Package) {
	if pxtest == nil || !needsLogosJSON(pxtest) {
		return
	}
	for _, q := range pxtest.Internal.Imports {
		if q.ImportPath == LogosJSONPkg {
			return
		}
	}
	stk := ImportStack{{Pkg: pxtest.ImportPath}}
	p1, err := loadImport(ld, ctx, opts, pre, LogosJSONPkg, pxtest.Dir, pxtest, &stk, nil, ResolveImport)
	if err != nil && pxtest.Error == nil {
		pxtest.Error = err
	}
	if err != nil || p1.Incomplete {
		pxtest.Incomplete = true
	}
	pxtest.Internal.Imports = append(pxtest.Internal.Imports, p1)
	pxtest.Internal.CompiledImports = append(pxtest.Internal.CompiledImports, p1.ImportPath)
}

// addLogosBinaryXTestImport loads the implicit binary-helper imports
// for the external test package of a logos package, mirroring
// addLogosRTXTestImport: synthesized nojsonv2 MarshalBinary helpers in
// external test files need the same archives. Plain Go packages never
// take it.
func addLogosBinaryXTestImport(ld *modload.Loader, ctx context.Context, opts PackageOpts, pre *preload, pxtest *Package) {
	if pxtest == nil || !needsLogosBinary(pxtest) {
		return
	}
	for _, path := range []string{"fmt", "encoding/binary", "math", "slices"} {
		present := false
		for _, q := range pxtest.Internal.Imports {
			if q.ImportPath == path {
				present = true
				break
			}
		}
		if present {
			continue
		}
		stk := ImportStack{{Pkg: pxtest.ImportPath}}
		p1, err := loadImport(ld, ctx, opts, pre, path, pxtest.Dir, pxtest, &stk, nil, ResolveImport)
		if err != nil && pxtest.Error == nil {
			pxtest.Error = err
		}
		if err != nil || p1.Incomplete {
			pxtest.Incomplete = true
		}
		pxtest.Internal.Imports = append(pxtest.Internal.Imports, p1)
		pxtest.Internal.CompiledImports = append(pxtest.Internal.CompiledImports, p1.ImportPath)
	}
}
