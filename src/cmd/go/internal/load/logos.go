// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
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

// logosSupportImport reports whether path is one of the designated loGo/s
// support namespaces. A loGo/s package may import them without tripping the
// kind boundary, and the allowance is path-based so it holds before the
// support packages exist in the tree.
func logosSupportImport(path string) bool {
	return path == "logos/rt" || path == "logos/sync" ||
		strings.HasPrefix(path, "logos/rt/") || strings.HasPrefix(path, "logos/sync/")
}

// logosIsCanonicalLgo reports whether name is a canonical .lgo file, after
// stripping trailing variation selectors the same way the shared filename
// predicate does.
func logosIsCanonicalLgo(name string) bool {
	return strings.HasSuffix(strings.TrimRight(name, "\uFE0E\uFE0F"), ".lgo")
}

// checkLogosImports enforces the loGo/s and Go import boundary for one
// package after its imports have resolved. A loGo/s package must not import
// a Go package and a Go package must not import a loGo/s package; the
// standard library and the designated support namespaces are allowed in
// either direction. The generated test main is exempt: it is a Go-kind
// harness that necessarily imports the package under test. The loadImport
// edge hook reports each edge first, so this per-package pass converges on
// the same verdict and then emits the alias warnings.
func checkLogosImports(p *Package) {
	if !cfg.Logos || p.Internal.TestmainGo != nil {
		return
	}
	for _, imp := range p.Internal.Imports {
		if logosImportViolation(p, imp) {
			break
		}
	}
	emitLogosAliasWarnings(p)
}

// logosImportsGoError reports a loGo/s package importing a Go package.
func logosImportsGoError(importerPath, impPath string) *PackageError {
	return &PackageError{
		Err: fmt.Errorf("[correctness::logos_imports_go] loGo/s package %s imports Go package %s; loGo/s and Go code do not import each other", importerPath, impPath),
	}
}

// goImportsLogosError reports a Go package importing a loGo/s package.
func goImportsLogosError(importerPath, impPath string) *PackageError {
	return &PackageError{
		Err: fmt.Errorf("[correctness::go_imports_logos] Go package %s imports loGo/s package %s", importerPath, impPath),
	}
}

// logosImportEdgeError returns the cross-kind violation for an import edge
// from importer to imp, or nil when the edge is allowed. It is pure:
// callers decide where the error lands. Imports that failed to load
// are skipped so a real load error keeps its own diagnostic, and the
// standard library plus the designated support namespaces are allowed in
// either direction.
func logosImportEdgeError(importer, imp *Package) *PackageError {
	if imp.Error != nil || imp.Standard || logosSupportImport(imp.ImportPath) {
		return nil
	}
	if importer.IsLogos() == imp.IsLogos() {
		return nil
	}
	if importer.IsLogos() {
		return logosImportsGoError(importer.ImportPath, imp.ImportPath)
	}
	return goImportsLogosError(importer.ImportPath, imp.ImportPath)
}

// checkLogosImportEdge enforces the kind boundary for one resolved import
// edge. loadImport is the single funnel for every edge, including test and
// external test imports resolved for test variants, so this covers edges
// that never appear in an importer's own file lists. It emits no warnings.
func checkLogosImportEdge(parent, imp *Package) *PackageError {
	if !cfg.Logos || parent == nil || parent.Internal.TestmainGo != nil {
		return nil
	}
	return logosImportEdgeError(parent, imp)
}

// logosImportViolation records a cross-kind import from p to imp and reports
// whether the scan should stop. Only the first violation is recorded, and
// only when the package has no error yet, so an upstream load failure keeps
// its own diagnostic.
func logosImportViolation(p, imp *Package) bool {
	err := logosImportEdgeError(p, imp)
	if err == nil {
		return false
	}
	if p.Error == nil {
		p.Error = err
		p.Incomplete = true
	}
	return true
}

// emitLogosAliasWarnings prints one warning per accepted alias-extension
// source file, including the test lists. Canonical .lgo files never warn.
// Only packages in a main module whose path starts with a dotted element
// warn: those are the modules that could be published, where the module
// zip rules reject alias extensions. Each package is checked once when it
// finishes loading, so no warning state is kept beyond the load.
func emitLogosAliasWarnings(p *Package) {
	mod := p.Module
	if mod == nil || !mod.Main {
		return
	}
	elem, _, _ := strings.Cut(mod.Path, "/")
	if !strings.Contains(elem, ".") {
		return
	}
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range files {
			warnLogosAliasFile(name)
		}
	}
}

// warnLogosAliasFile warns once for an alias-extension source file.
func warnLogosAliasFile(name string) {
	if !cfg.IsLogosFile(name) || logosIsCanonicalLgo(name) {
		return
	}
	fmt.Fprintf(os.Stderr, "go: warning: [portability::alias_extension] %s: loGo/s alias extension file; rename to .lgo before publishing\n", name)
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
