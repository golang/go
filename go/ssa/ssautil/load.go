// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssautil

// This file defines utility functions for constructing programs in SSA form.

import (
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types"
)

// CreateProgram returns a new program in SSA form, given a program
// loaded from source.  An SSA package is created for each transitively
// error-free package of lprog.
//
// Code for bodies of functions is not built until BuildAll() is called
// on the result.
//
// mode controls diagnostics and checking during SSA construction.
//
func CreateProgram(lprog *loader.Program, mode ssa.BuilderMode) *ssa.Program {
	// TODO(adonovan): inline and delete ssa.Create.
	return ssa.Create(lprog, mode)
}

// LoadPackage builds SSA code for a single package.
//
// It populates pkg by type-checking the specified file ASTs.  All
// dependencies are loaded using the importer specified by tc, which
// typically loads compiler export data; SSA code cannot be built for
// those packages.  LoadPackage then constructs an ssa.Program with all
// dependency packages created, and builds and returns the SSA package
// corresponding to pkg.
//
// The caller must have set pkg.Path() to the import path.
//
// The operation fails if there were any type-checking or import errors.
//
// LoadPackage modifies the tc.Import field.
//
func LoadPackage(tc *types.Config, fset *token.FileSet, pkg *types.Package, files []*ast.File, mode ssa.BuilderMode) (*ssa.Package, *types.Info, error) {
	if fset == nil {
		panic("no token.FileSet")
	}
	if pkg.Path() == "" {
		panic("no package path")
	}

	// client's effective import function
	clientImport := tc.Import
	if clientImport == nil {
		clientImport = types.DefaultImport
	}

	deps := make(map[*types.Package]bool)

	tc.Import = func(packages map[string]*types.Package, path string) (pkg *types.Package, err error) {
		pkg, err = clientImport(packages, path)
		if pkg != nil {
			deps[pkg] = true
		}
		return
	}

	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	if err := types.NewChecker(tc, fset, pkg, info).Files(files); err != nil {
		return nil, nil, err
	}

	// Create the SSA program and its packages.
	prog := ssa.NewProgram(fset, mode)
	for dep := range deps {
		prog.CreatePackage(dep, nil, nil, true)
	}
	ssapkg := prog.CreatePackage(pkg, files, info, false)
	ssapkg.Build()
	return ssapkg, info, nil
}
