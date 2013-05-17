package ssa

// This file defines an implementation of the types.Importer interface
// (func) that loads the transitive closure of dependencies of a
// "main" package.

import (
	"errors"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"code.google.com/p/go.tools/go/types"
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
	var info *TypeInfo
	if b.Context.Mode&UseGCImporter != 0 {
		typkg, err = types.GcImport(imports, path)
	} else {
		files, err = b.Context.Loader(b.Prog.Files, path)
		if err == nil {
			typkg, info, err = b.typecheck(path, files)
		}
	}
	if err != nil {
		// Cache failure
		b.importErrs[path] = err
		return nil, err
	}

	// Cache success
	imports[path] = typkg                                                 // cache for just this package.
	b.Prog.Packages[path] = b.createPackageImpl(typkg, path, files, info) // cache across all packages

	return typkg, nil
}

// MakeGoBuildLoader returns an implementation of the SourceLoader
// function prototype that locates packages using the go/build
// libraries.  It may return nil upon gross misconfiguration
// (e.g. os.Getwd() failed).
//
// ctxt specifies the go/build.Context to use; if nil, the default
// Context is used.
//
func MakeGoBuildLoader(ctxt *build.Context) SourceLoader {
	srcDir, err := os.Getwd()
	if err != nil {
		return nil // serious misconfiguration
	}
	if ctxt == nil {
		ctxt = &build.Default
	}
	return func(fset *token.FileSet, path string) (files []*ast.File, err error) {
		// TODO(adonovan): fix: Do we need cwd? Shouldn't
		// ImportDir(path) / $GOROOT suffice?
		bp, err := ctxt.Import(path, srcDir, 0)
		if err != nil {
			return // import failed
		}
		files, err = ParseFiles(fset, bp.Dir, bp.GoFiles...)
		if err != nil {
			return nil, err
		}
		return
	}
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

// CreatePackageFromArgs builds an initial Package from a list of
// command-line arguments.
// If args is a list of *.go files, they are parsed and type-checked.
// If args is a Go package import path, that package is imported.
// rest is the suffix of args that were not consumed.
//
// This utility is provided to facilitate construction of command-line
// tools with a consistent user interface.
//
func CreatePackageFromArgs(builder *Builder, args []string) (pkg *Package, rest []string, err error) {
	var pkgname string
	var files []*ast.File

	switch {
	case len(args) == 0:
		err = errors.New("No *.go source files nor package name was specified.")

	case strings.HasSuffix(args[0], ".go"):
		// % tool a.go b.go ...
		// Leading consecutive *.go arguments constitute main package.
		pkgname = "main"
		i := 1
		for ; i < len(args) && strings.HasSuffix(args[i], ".go"); i++ {
		}
		files, err = ParseFiles(builder.Prog.Files, ".", args[:i]...)
		rest = args[i:]

	default:
		// % tool my/package ...
		// First argument is import path of main package.
		pkgname = args[0]
		rest = args[1:]
		files, err = builder.Context.Loader(builder.Prog.Files, pkgname)
	}
	if err == nil {
		pkg, err = builder.CreatePackage(pkgname, files)
	}
	return
}
