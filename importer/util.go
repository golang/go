// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"sync"
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
		return nil, nil, errors.New("no *.go source files nor package name was specified.")

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
			info, err = imp.CreateSourcePackage("main", files)
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

var cwd string

func init() {
	var err error
	cwd, err = os.Getwd()
	if err != nil {
		panic("getcwd failed: " + err.Error())
	}
}

// loadPackage ascertains which files belong to package path, then
// loads, parses and returns them.
func loadPackage(ctxt *build.Context, fset *token.FileSet, path string) (files []*ast.File, err error) {
	// TODO(adonovan): fix: Do we need cwd? Shouldn't
	// ImportDir(path) / $GOROOT suffice?
	bp, err := ctxt.Import(path, cwd, 0)
	if _, ok := err.(*build.NoGoError); ok {
		return nil, nil // empty directory
	}
	if err != nil {
		return // import failed
	}
	return ParseFiles(fset, bp.Dir, bp.GoFiles...)
}

// ParseFiles parses the Go source files files within directory dir
// and returns their ASTs, or the first parse error if any.
//
func ParseFiles(fset *token.FileSet, dir string, files ...string) ([]*ast.File, error) {
	var wg sync.WaitGroup
	n := len(files)
	parsed := make([]*ast.File, n, n)
	errors := make([]error, n, n)
	for i, file := range files {
		if !filepath.IsAbs(file) {
			file = filepath.Join(dir, file)
		}
		wg.Add(1)
		go func(i int, file string) {
			parsed[i], errors[i] = parser.ParseFile(fset, file, nil, parser.DeclarationErrors)
			wg.Done()
		}(i, file)
	}
	wg.Wait()

	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	return parsed, nil
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
