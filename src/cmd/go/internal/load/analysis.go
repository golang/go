// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

// This file supports analysis in go test: the configuration of the
// analyzers that apply to each root package (one for which go test
// reports diagnostics, as opposed to a dependency) as references to
// analyzer and suite variables, and the synthesis of the analysis
// tool that contains them. Planning and running the analysis is in
// cmd/go/internal/work/analysis.go.
//
// Resolution of the configuration has two stages. The static stage,
// here, determines which variables the tool links, from references
// to them, before the tool exists. The dynamic stage (see
// work.Builder.planAnalysis) uses the tool's metadata to expand
// suites into their members and select the analyzers that apply to
// each package.
//
// The configuration of go test's analysis is currently fixed: the
// vet suite (internal/analysis/suite/vet), or the members of it
// selected by the -vet flag.

import (
	"bytes"
	"context"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/build"
	"cmd/go/internal/modload"
)

// AnalysisToolDriver is the import path of the driver package linked
// into the synthesized analysis tool.
const AnalysisToolDriver = "internal/analysis/gotestchecker"

// analysisVetPackage is the package whose Suite variable holds the
// vet suite, from which go test's analyzers are selected.
const analysisVetPackage = "internal/analysis/suite/vet"

// An AnalysisVar identifies an exported analyzer or suite variable.
type AnalysisVar struct {
	PkgPath string
	Name    string
}

func (v AnalysisVar) String() string { return v.PkgPath + "." + v.Name }

// An AnalysisRef is a reference to an analyzer variable (pkg.Var) or
// to members of a suite variable (pkg.Var[*] or pkg.Var[A,B]), or (if
// Negative) a negation of analyzers of a given Name.
type AnalysisRef struct {
	Negative bool        // a negation (disable)
	Var      AnalysisVar // the referenced variable
	Members  []string    // nil for an analyzer (pkg.Var); ["*"] or names for a suite (pkg.Var[...])
	Name     string      // for a negation by name: the Analyzer.Name (Var and Members are empty)
}

// Ref returns the reference as written, e.g. "pkgpath.Var[A,B]",
// or the quoted analyzer name of a negation by name.
func (a *AnalysisRef) Ref() string {
	if a.ByName() {
		return strconv.Quote(a.Name)
	}
	ref := a.Var.String()
	if a.Members != nil {
		ref += "[" + strings.Join(a.Members, ",") + "]"
	}
	return ref
}

// ByName reports whether the reference is a negation that names
// analyzers by Analyzer.Name rather than by reference. Such a negation
// is applied only in the dynamic stage, to every analyzer of that name;
// it is not linked into the tool.
func (a *AnalysisRef) ByName() bool { return a.Name != "" }

// IsSuite reports whether the reference is to a suite (pkg.Var[...]).
func (a *AnalysisRef) IsSuite() bool { return a.Members != nil }

// AllMembers reports whether the reference selects all members of a
// suite (pkg.Var[*]).
func (a *AnalysisRef) AllMembers() bool {
	return len(a.Members) == 1 && a.Members[0] == "*"
}

// Errorf returns an error about the reference.
func (a *AnalysisRef) Errorf(format string, args ...any) error {
	return fmt.Errorf("%s: %s", a.Ref(), fmt.Sprintf(format, args...))
}

// CompareAnalysisRefs orders references deterministically.
func CompareAnalysisRefs(x, y *AnalysisRef) int {
	return strings.Compare(x.Ref(), y.Ref())
}

// An AnalysisConfig records the references that apply to a root
// package.
type AnalysisConfig struct {
	// Positive holds the references that enable analyzers for the
	// package, in order of decreasing specificity.
	Positive []*AnalysisRef

	// Negative holds the negations that apply to the package,
	// which are applied once the analyzers are known.
	Negative []*AnalysisRef
}

// VetAnalysisConfig returns go test's built-in analysis configuration:
// the members of the vet suite with the specified names, or all of
// them if names is nil.
func VetAnalysisConfig(names []string) *AnalysisConfig {
	var members []string
	if names == nil {
		members = []string{"*"}
	}
	for _, name := range names {
		// cmd/vet accepts some legacy names (see analysisflags).
		if legacy, ok := vetLegacyNames[name]; ok {
			name = legacy
		}
		if !slices.Contains(members, name) {
			members = append(members, name)
		}
	}
	return &AnalysisConfig{
		Positive: []*AnalysisRef{{
			Var:     AnalysisVar{analysisVetPackage, "Suite"},
			Members: members,
		}},
	}
}

// vetLegacyNames maps the legacy names of vet analyzers, which the
// -vet flag of go test accepts for compatibility with cmd/vet, to
// their names.
var vetLegacyNames = map[string]string{
	"bool":       "bools",
	"buildtags":  "buildtag",
	"methods":    "stdmethods",
	"rangeloops": "loopclosure",
}

// AnalysisToolPackage returns a synthesized main package for an
// analysis tool and the source of its sole file. The tool contains
// the variables referenced by the specified references (those that
// enable analyzers for some root package).
//
// The package of each reference must load. A reference whose package
// does not load is reported, and dropped. (Whether the package
// declares the variable is discovered only if the tool fails to
// compile.)
//
// It also returns errors about the offending references.
//
// The package's imports are loaded without regard to the rule for
// internal packages, like the generated testmain: the tool must
// import analyzers from internal packages, and the driver itself is
// an internal package of the standard library.
func AnalysisToolPackage(ld *modload.Loader, ctx context.Context, opts PackageOpts, refs []*AnalysisRef) (pmain *Package, src []byte, errs []error) {
	pmain = &Package{
		PackagePublic: PackagePublic{
			Name:       "main",
			GoFiles:    []string{"analysistool.go"},
			ImportPath: "analysistool",
		},
		Internal: PackageInternal{
			Build:     &build.Package{Name: "main"},
			OmitDebug: true,
		},
	}
	pmain.Internal.ExeName = "analysistool"
	setToolFlags(ld, pmain)

	var stk ImportStack
	stk.Push(ImportInfo{Pkg: "analysistool"})
	defer stk.Pop()

	// Load the driver and the packages of the references.
	// (For now, all are standard packages.)
	pkgs := make(map[string]*Package)
	load := func(paths []string) {
		for _, path := range paths {
			if _, ok := pkgs[path]; !ok {
				pkgs[path], _ = loadImport(ld, ctx, opts, nil, path, "", nil, &stk, nil, 0)
			}
		}
	}
	refPkgs := []string{AnalysisToolDriver}
	for _, a := range refs {
		if !slices.Contains(refPkgs, a.Var.PkgPath) {
			refPkgs = append(refPkgs, a.Var.PkgPath)
		}
	}
	load(refPkgs)

	// checkRef checks that the package of the reference loaded
	// without error and is importable. It does not check that the
	// package declares the variable, which would require parsing
	// the whole package: a missing variable makes the analysis tool
	// fail to compile. Nor does it check the variable's type, which
	// is checked by the analysis tool.
	checkRef := func(a *AnalysisRef) error {
		p := pkgs[a.Var.PkgPath]
		if p == nil {
			return fmt.Errorf("cannot find package %s", a.Var.PkgPath)
		}
		if p.Error != nil {
			return fmt.Errorf("cannot load package %s: %v", a.Var.PkgPath, p.Error)
		}
		if p.Name == "main" {
			return fmt.Errorf("package %s is a program, not an importable package", a.Var.PkgPath)
		}
		return nil
	}

	// Check the references, and link the variables of the valid
	// ones. (checkRef depends only on the package, so all
	// references to one variable succeed or fail together.)
	linked := make(map[AnalysisVar]bool)
	deps := []string{AnalysisToolDriver} // imports of the main package
	depSet := map[string]bool{AnalysisToolDriver: true}
	for _, a := range refs {
		if err := checkRef(a); err != nil {
			errs = append(errs, a.Errorf("%v", err))
			continue
		}
		linked[a.Var] = true
		if !depSet[a.Var.PkgPath] {
			depSet[a.Var.PkgPath] = true
			deps = append(deps, a.Var.PkgPath)
		}
	}

	// Add the imports needed by the linker.
	ldDeps, err := LinkerDeps(ld, pmain)
	if err != nil && pmain.Error == nil {
		pmain.Error = &PackageError{Err: err}
	}
	deps = append(deps, ldDeps...)
	for _, dep := range deps {
		p1 := pkgs[dep]
		if p1 == nil {
			var err *PackageError
			p1, err = loadImport(ld, ctx, opts, nil, dep, "", nil, &stk, nil, 0)
			if err != nil && pmain.Error == nil {
				pmain.Error = err
				pmain.Incomplete = true
			}
			pkgs[dep] = p1
		}
		if p1.Error != nil && pmain.Error == nil {
			pmain.Error = p1.Error
			pmain.Incomplete = true
		}
		if !slices.Contains(pmain.Imports, dep) {
			pmain.Imports = append(pmain.Imports, dep)
			pmain.Internal.Imports = append(pmain.Internal.Imports, p1)
		}
	}

	return pmain, AnalysisToolSource(maps.Keys(linked)), errs
}

// AnalysisToolSource returns the source of the main package of an
// analysis tool that links the specified variables.
func AnalysisToolSource(vars iter.Seq[AnalysisVar]) []byte {
	var imports, entries bytes.Buffer
	pkgname := make(map[string]string) // pkgpath -> import name
	// The variables are in order of qualified name (see
	// gotestchecker.Main).
	byName := func(x, y AnalysisVar) int {
		return strings.Compare(x.String(), y.String())
	}
	for _, v := range slices.SortedFunc(vars, byName) {
		name, ok := pkgname[v.PkgPath]
		if !ok {
			name = fmt.Sprintf("a%d", len(pkgname))
			pkgname[v.PkgPath] = name
			fmt.Fprintf(&imports, "\t%s %q\n", name, v.PkgPath)
		}
		fmt.Fprintf(&entries, "\t\t{%q, %s.%s},\n", v.String(), name, v.Name)
	}
	return fmt.Appendf(nil, `// Code generated by go test. DO NOT EDIT.

package main

import (
	%q
%s)

func main() {
	gotestchecker.Main({
%s	})
}
`, AnalysisToolDriver, imports.Bytes(), entries.Bytes())
}
