// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package importer defines the Importer, which loads, parses and
// type-checks packages of Go code plus their transitive closure, and
// retains both the ASTs and the derived facts.
//
// TODO(adonovan): document and test this package better, with examples.
// Currently it's covered by the ssa/ tests.
//
package importer

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"os"
	"strconv"
	"sync"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// An Importer's methods are not thread-safe unless specified.
type Importer struct {
	config   *Config                 // the client configuration
	Fset     *token.FileSet          // position info for all files seen
	Packages map[string]*PackageInfo // successfully imported packages keyed by import path
	errors   map[string]error        // cache of errors by import path

	mu       sync.Mutex          // guards 'packages' map
	packages map[string]*pkgInfo // all attempted imported packages, keyed by import path
}

// pkgInfo holds internal per-package information.
type pkgInfo struct {
	info  *PackageInfo  // success info
	err   error         // failure info
	ready chan struct{} // channel close is notification of ready state
}

// Config specifies the configuration for the importer.
//
type Config struct {
	// TypeChecker contains options relating to the type checker.
	// The Importer will override any user-supplied values for its
	// Error and Import fields; other fields will be passed
	// through to the type checker.
	TypeChecker types.Config

	// If Build is non-nil, it is used to satisfy imports.
	//
	// If it is nil, binary object files produced by the gc
	// compiler will be loaded instead of source code for all
	// imported packages.  Such files supply only the types of
	// package-level declarations and values of constants, but no
	// code, so this mode will not yield a whole program.  It is
	// intended for analyses that perform intraprocedural analysis
	// of a single package.
	Build *build.Context
}

// New returns a new, empty Importer using configuration options
// specified by config.
//
func New(config *Config) *Importer {
	imp := &Importer{
		config:   config,
		Fset:     token.NewFileSet(),
		packages: make(map[string]*pkgInfo),
		Packages: make(map[string]*PackageInfo),
		errors:   make(map[string]error),
	}
	imp.config.TypeChecker.Error = func(e error) { fmt.Fprintln(os.Stderr, e) }
	imp.config.TypeChecker.Import = imp.doImport
	return imp
}

// doImport loads the typechecker package identified by path
// Implements the types.Importer prototype.
//
func (imp *Importer) doImport(imports map[string]*types.Package, path string) (pkg *types.Package, err error) {
	// Package unsafe is handled specially, and has no PackageInfo.
	if path == "unsafe" {
		return types.Unsafe, nil
	}

	// Load the source/binary for 'path', type-check it, construct
	// a PackageInfo and update our map (imp.Packages) and the
	// type-checker's map (imports).
	// TODO(adonovan): make it the type-checker's job to
	// consult/update the 'imports' map.  Then we won't need it.
	var info *PackageInfo
	if imp.config.Build != nil {
		info, err = imp.LoadPackage(path)
	} else {
		if info, ok := imp.Packages[path]; ok {
			imports[path] = info.Pkg
			pkg = info.Pkg
			return // positive cache hit
		}

		if err = imp.errors[path]; err != nil {
			return // negative cache hit
		}

		if pkg, err = types.GcImport(imports, path); err == nil {
			info = &PackageInfo{Pkg: pkg}
			imp.Packages[path] = info
		}
	}

	if err == nil {
		// Cache success.
		pkg = info.Pkg
		imports[path] = pkg
		return pkg, nil
	}

	// Cache failure
	imp.errors[path] = err
	return nil, err
}

// imports returns the set of paths imported by the specified files.
func imports(p string, files []*ast.File) map[string]bool {
	imports := make(map[string]bool)
outer:
	for _, file := range files {
		for _, decl := range file.Decls {
			if decl, ok := decl.(*ast.GenDecl); ok {
				if decl.Tok != token.IMPORT {
					break outer // stop at the first non-import
				}
				for _, spec := range decl.Specs {
					spec := spec.(*ast.ImportSpec)
					if path, _ := strconv.Unquote(spec.Path.Value); path != "C" {
						imports[path] = true
					}
				}
			} else {
				break outer // stop at the first non-import
			}
		}
	}
	return imports
}

// LoadPackage loads the package of the specified import path,
// performs type-checking, and returns the corresponding
// PackageInfo.
//
// Precondition: Importer.config.Build != nil.
// Idempotent and thread-safe.
//
// TODO(adonovan): fix: violated in call from CreatePackageFromArgs!
// TODO(adonovan): rethink this API.
//
func (imp *Importer) LoadPackage(importPath string) (*PackageInfo, error) {
	if imp.config.Build == nil {
		panic("Importer.LoadPackage without a go/build.Config")
	}

	imp.mu.Lock()
	pi, ok := imp.packages[importPath]
	if !ok {
		// First request for this pkgInfo:
		// create a new one in !ready state.
		pi = &pkgInfo{ready: make(chan struct{})}
		imp.packages[importPath] = pi
		imp.mu.Unlock()

		// Load/parse the package.
		if files, err := loadPackage(imp.config.Build, imp.Fset, importPath); err == nil {
			// Kick off asynchronous I/O and parsing for the imports.
			for path := range imports(importPath, files) {
				go func(path string) { imp.LoadPackage(path) }(path)
			}

			// Type-check the package.
			pi.info, pi.err = imp.CreateSourcePackage(importPath, files)
		} else {
			pi.err = err
		}

		close(pi.ready) // enter ready state and wake up waiters
	} else {
		imp.mu.Unlock()

		<-pi.ready // wait for ready condition.
	}

	// Invariant: pi is ready.

	if pi.err != nil {
		return nil, pi.err
	}
	return pi.info, nil
}

// CreateSourcePackage invokes the type-checker on files and returns a
// PackageInfo containing the resulting type-checker package, the
// ASTs, and other type information.
//
// The order of files determines the package initialization order.
//
// importPath is the full name under which this package is known, such
// as appears in an import declaration. e.g. "sync/atomic".
//
// Postcondition: for result (info, err), info.Err == err.
//
func (imp *Importer) CreateSourcePackage(importPath string, files []*ast.File) (*PackageInfo, error) {
	info := &PackageInfo{
		Files: files,
		Info: types.Info{
			Types:      make(map[ast.Expr]types.Type),
			Values:     make(map[ast.Expr]exact.Value),
			Objects:    make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Scopes:     make(map[ast.Node]*types.Scope),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
		},
	}
	info.Pkg, info.Err = imp.config.TypeChecker.Check(importPath, imp.Fset, files, &info.Info)
	imp.Packages[importPath] = info
	return info, info.Err
}
