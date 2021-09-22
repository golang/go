// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildssa defines an Analyzer that constructs the SSA
// representation of an error-free package and returns the set of all
// functions within it. It does not report any diagnostics itself but
// may be used as an input to other analyzers.
//
// THIS INTERFACE IS EXPERIMENTAL AND MAY BE SUBJECT TO INCOMPATIBLE CHANGE.
package buildssa

import (
	"go/ast"
	"go/types"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ssa"
)

var Analyzer = &analysis.Analyzer{
	Name:       "buildssa",
	Doc:        "build SSA-form IR for later passes",
	Run:        run,
	ResultType: reflect.TypeOf(new(SSA)),
}

// SSA provides SSA-form intermediate representation for all the
// non-blank source functions in the current package.
type SSA struct {
	Pkg      *ssa.Package
	SrcFuncs []*ssa.Function
}

func run(pass *analysis.Pass) (interface{}, error) {
	// Plundered from ssautil.BuildPackage.

	// We must create a new Program for each Package because the
	// analysis API provides no place to hang a Program shared by
	// all Packages. Consequently, SSA Packages and Functions do not
	// have a canonical representation across an analysis session of
	// multiple packages. This is unlikely to be a problem in
	// practice because the analysis API essentially forces all
	// packages to be analysed independently, so any given call to
	// Analysis.Run on a package will see only SSA objects belonging
	// to a single Program.

	// Some Analyzers may need GlobalDebug, in which case we'll have
	// to set it globally, but let's wait till we need it.
	// Monomorphize at least until type parameters are available.
	mode := ssa.InstantiateGenerics

	prog := ssa.NewProgram(pass.Fset, mode)

	// Create SSA packages for all imports.
	// Order is not significant.
	created := make(map[*types.Package]bool)
	var createAll func(pkgs []*types.Package)
	createAll = func(pkgs []*types.Package) {
		for _, p := range pkgs {
			if !created[p] {
				created[p] = true
				prog.CreatePackage(p, nil, nil, true)
				createAll(p.Imports())
			}
		}
	}
	createAll(pass.Pkg.Imports())

	// Create and build the primary package.
	ssapkg := prog.CreatePackage(pass.Pkg, pass.Files, pass.TypesInfo, false)
	ssapkg.Build()

	// Compute list of source functions, including literals,
	// in source order.
	var funcs []*ssa.Function
	for _, f := range pass.Files {
		for _, decl := range f.Decls {
			if fdecl, ok := decl.(*ast.FuncDecl); ok {

				// SSA will not build a Function
				// for a FuncDecl named blank.
				// That's arguably too strict but
				// relaxing it would break uniqueness of
				// names of package members.
				if fdecl.Name.Name == "_" {
					continue
				}

				// (init functions have distinct Func
				// objects named "init" and distinct
				// ssa.Functions named "init#1", ...)

				fn := pass.TypesInfo.Defs[fdecl.Name].(*types.Func)
				if fn == nil {
					panic(fn)
				}

				f := ssapkg.Prog.FuncValue(fn)
				if f == nil {
					panic(fn)
				}

				var addAnons func(f *ssa.Function)
				addAnons = func(f *ssa.Function) {
					funcs = append(funcs, f)
					for _, anon := range f.AnonFuncs {
						addAnons(anon)
					}
				}
				addAnons(f)
			}
		}
	}

	return &SSA{Pkg: ssapkg, SrcFuncs: funcs}, nil
}
