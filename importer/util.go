package importer

// This file defines various utility functions exposed by the package
// and used by it.

import (
	"errors"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// CreatePackageFromArgs builds an initial Package from a list of
// command-line arguments.
// If args is a list of *.go files, they are parsed and type-checked.
// If args is a Go package import path, that package is imported.
// The rest result contains the suffix of args that were not consumed.
//
// This utility is provided to facilitate construction of command-line
// tools with a consistent user interface.
//
func CreatePackageFromArgs(imp *Importer, args []string) (info *PackageInfo, rest []string, err error) {
	switch {
	case len(args) == 0:
		return nil, nil, errors.New("No *.go source files nor package name was specified.")

	case strings.HasSuffix(args[0], ".go"):
		// % tool a.go b.go ...
		// Leading consecutive *.go arguments constitute main package.
		i := 1
		for ; i < len(args) && strings.HasSuffix(args[i], ".go"); i++ {
		}
		var files []*ast.File
		files, err = ParseFiles(imp.Fset, ".", args[:i]...)
		rest = args[i:]
		if err == nil {
			info = imp.CreateSourcePackage("main", files)
			err = info.Err
		}

	default:
		// % tool my/package ...
		// First argument is import path of main package.
		pkgname := args[0]
		info, err = imp.LoadPackage(pkgname)
		rest = args[1:]
	}

	return
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

// ---------- Internal helpers ----------

// unparen returns e with any enclosing parentheses stripped.
func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			break
		}
		e = p.X
	}
	return e
}

func unreachable() {
	panic("unreachable")
}
