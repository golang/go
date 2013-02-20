package ssa

// This file defines an implementation of the types.Importer interface
// (func) that loads the transitive closure of dependencies of a
// "main" package.

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
)

// Prototype of a function that locates, reads and parses a set of
// source files given an import path.
//
// fset is the fileset to which the ASTs should be added.
// path is the imported path, e.g. "sync/atomic".
//
// On success, the function returns files, the set of ASTs produced,
// or the first error encountered.
//
type SourceLoader func(fset *token.FileSet, path string) (files []*ast.File, err error)

// doImport loads the typechecker package identified by path
// Implements the types.Importer prototype.
//
func (b *Builder) doImport(imports map[string]*types.Package, path string) (typkg *types.Package, err error) {
	// Package unsafe is handled specially, and has no ssa.Package.
	if path == "unsafe" {
		return types.Unsafe, nil
	}

	if pkg := b.Prog.Packages[path]; pkg != nil {
		typkg = pkg.Types
		imports[path] = typkg
		return // positive cache hit
	}

	if err = b.importErrs[path]; err != nil {
		return // negative cache hit
	}
	var files []*ast.File
	if b.mode&UseGCImporter != 0 {
		typkg, err = types.GcImport(&b.typechecker, imports, path)
	} else {
		files, err = b.loader(b.Prog.Files, path)
		if err == nil {
			typkg, err = b.typechecker.Check(b.Prog.Files, files)
		}
	}
	if err != nil {
		// Cache failure
		b.importErrs[path] = err
		return nil, err
	}

	// Cache success
	imports[path] = typkg                                           // cache for just this package.
	b.Prog.Packages[path] = b.createPackageImpl(typkg, path, files) // cache across all packages

	return typkg, nil
}

// GorootLoader is an implementation of the SourceLoader function
// prototype that loads and parses Go source files from the package
// directory beneath $GOROOT/src/pkg.
//
// TODO(adonovan): get rsc and adg (go/build owners) to review this.
//
func GorootLoader(fset *token.FileSet, path string) (files []*ast.File, err error) {
	// TODO(adonovan): fix: Do we need cwd? Shouldn't ImportDir(path) / $GOROOT suffice?
	srcDir, err := os.Getwd()
	if err != nil {
		return // serious misconfiguration
	}
	bp, err := build.Import(path, srcDir, 0)
	if err != nil {
		return // import failed
	}
	files, err = ParseFiles(fset, bp.Dir, bp.GoFiles...)
	if err != nil {
		return nil, err
	}
	return
}

// ParseFiles parses the Go source files files within directory dir
// and returns their ASTs, or the first parse error if any.
//
// This utility function is provided to facilitate implementing a
// SourceLoader.
//
func ParseFiles(fset *token.FileSet, dir string, files ...string) (parsed []*ast.File, err error) {
	for _, file := range files {
		var f *ast.File
		if !filepath.IsAbs(file) {
			file = filepath.Join(dir, file)
		}
		f, err = parser.ParseFile(fset, file, nil, parser.DeclarationErrors)
		if err != nil {
			return // parsing failed
		}
		parsed = append(parsed, f)
	}
	return
}
