// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
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
			return cfg.IsLogosFile(files[0])
		}
	}
	return false
}

// moduleKinds groups one module's loaded packages by kind, keyed by import
// path so repeated observations of one package stay a single entry.
type moduleKinds struct {
	goPkgs    map[string]*Package
	logosPkgs map[string]*Package
}

func (g *moduleKinds) add(p *Package) {
	if p.IsLogos() {
		g.logosPkgs[p.ImportPath] = p
	} else {
		g.goPkgs[p.ImportPath] = p
	}
}

// checkLogosModules reports [correctness::mixed_module] once per module
// whose loaded packages span both kinds. The error rides the existing
// per-package error mechanism on the lexicographically first error-free
// package of the mixed module. The generated test main is exempt from the
// census. Packages outside any module (standard library,
// command-line-arguments without a resolved module) and packages with no
// source files are skipped.
func checkLogosModules(all []*Package) {
	if !cfg.Logos {
		return
	}
	byModule := censusModuleKinds(all)
	for _, modPath := range slices.Sorted(maps.Keys(byModule)) {
		reportMixedModule(modPath, byModule[modPath])
	}
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
				goPkgs:    make(map[string]*Package),
				logosPkgs: make(map[string]*Package),
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
	carrier := firstCleanPackage(g, goNames, logosNames)
	if carrier == nil {
		return
	}
	carrier.Error = &PackageError{
		Err: fmt.Errorf("[correctness::mixed_module] module %s has Go packages (%s) and loGo/s packages (%s)", modPath, strings.Join(goNames, ", "), strings.Join(logosNames, ", ")),
	}
	carrier.Incomplete = true
}

// firstCleanPackage returns the lexicographically first error-free package
// of a mixed module, or nil when every package already has an error. The
// census diagnostic lands there so it never overwrites a more specific
// error, and it is suppressed only when the module already fails everywhere.
func firstCleanPackage(g *moduleKinds, goNames, logosNames []string) *Package {
	for len(goNames) > 0 && len(logosNames) > 0 {
		if goNames[0] < logosNames[0] {
			if p := g.goPkgs[goNames[0]]; p.Error == nil {
				return p
			}
			goNames = goNames[1:]
			continue
		}
		if p := g.logosPkgs[logosNames[0]]; p.Error == nil {
			return p
		}
		logosNames = logosNames[1:]
	}
	for _, name := range goNames {
		if p := g.goPkgs[name]; p.Error == nil {
			return p
		}
	}
	for _, name := range logosNames {
		if p := g.logosPkgs[name]; p.Error == nil {
			return p
		}
	}
	return nil
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
