// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Importer implements the types.ImporterFrom interface.
// It looks for Go2 packages using GO2PATH.
// Imported Go2 packages are rewritten to normal Go packages.
// This type also tracks references across imported packages.
type Importer struct {
	// Temporary directory used to rewrite packages.
	tmpdir string

	// Aggregated info from go/types.
	info *types.Info

	// Map from import path to directory holding rewritten files.
	translated map[string]string

	// Map from import path to package information.
	packages map[string]*types.Package

	// Map from Object to AST function declaration for
	// parameterized functions.
	idToFunc map[types.Object]*ast.FuncDecl

	// Map from Object to AST type definition for parameterized types.
	idToTypeSpec map[types.Object]*ast.TypeSpec
}

var _ types.ImporterFrom = &Importer{}

// NewImporter returns a new Importer.
// The tmpdir will become a GOPATH with translated files.
func NewImporter(tmpdir string) *Importer {
	info := &types.Info{
		Types:    make(map[ast.Expr]types.TypeAndValue),
		Inferred: make(map[*ast.CallExpr]types.Inferred),
		Defs:     make(map[*ast.Ident]types.Object),
		Uses:     make(map[*ast.Ident]types.Object),
	}
	return &Importer{
		tmpdir:       tmpdir,
		info:         info,
		translated:   make(map[string]string),
		packages:     make(map[string]*types.Package),
		idToFunc:     make(map[types.Object]*ast.FuncDecl),
		idToTypeSpec: make(map[types.Object]*ast.TypeSpec),
	}
}

// defaultImporter is the default Go 1 Importer.
var defaultImporter = importer.Default().(types.ImporterFrom)

// Import should never be called. This is the old API; current code
// uses ImportFrom. This method still needs to be defined in order
// to implement the interface.
func (imp *Importer) Import(path string) (*types.Package, error) {
	log.Fatal("unexpected call to Import method")
	return nil, nil
}

// ImportFrom looks for a Go2 package, and if not found tries the
// default importer.
func (imp *Importer) ImportFrom(importPath, dir string, mode types.ImportMode) (*types.Package, error) {
	if build.IsLocalImport(importPath) {
		return imp.localImport(importPath, dir)
	}

	if imp.translated[importPath] != "" {
		tpkg, ok := imp.packages[importPath]
		if !ok {
			return nil, fmt.Errorf("circular import when processing %q", importPath)
		}
		return tpkg, nil
	}

	var pdir string
	if go2path := os.Getenv("GO2PATH"); go2path != "" {
		pdir = imp.findFromPath(go2path, importPath)
	}
	if pdir == "" {
		bpkg, err := build.Import(importPath, dir, build.FindOnly)
		if err != nil {
			return nil, err
		}
		pdir = bpkg.Dir
	}

	// If the directory holds .go2 files, we need to translate them.
	fdir, err := os.Open(pdir)
	if err != nil {
		return nil, err
	}
	defer fdir.Close()
	names, err := fdir.Readdirnames(-1)
	if err != nil {
		return nil, err
	}
	var gofiles, go2files []string
	for _, name := range names {
		switch filepath.Ext(name) {
		case ".go":
			gofiles = append(gofiles, name)
		case ".go2":
			go2files = append(go2files, name)
		}
	}

	if len(go2files) == 0 {
		// No .go2 files, so the default importer can handle it.
		return defaultImporter.ImportFrom(importPath, dir, mode)
	}

	if len(gofiles) > 0 {
		for _, gofile := range gofiles {
			if err := checkGoFile(pdir, gofile); err != nil {
				return nil, err
			}
		}
	}

	tdir := filepath.Join(imp.tmpdir, "src", importPath)
	if err := os.MkdirAll(tdir, 0755); err != nil {
		return nil, err
	}
	for _, name := range names {
		data, err := ioutil.ReadFile(filepath.Join(pdir, name))
		if err != nil {
			return nil, err
		}
		if err := ioutil.WriteFile(filepath.Join(tdir, name), data, 0644); err != nil {
			return nil, err
		}
	}

	imp.translated[importPath] = tdir

	tpkgs, err := rewriteToPkgs(imp, tdir)
	if err != nil {
		return nil, err
	}

	switch len(tpkgs) {
	case 1:
		return tpkgs[0], nil
	case 2:
		if strings.HasSuffix(tpkgs[0].Name(), "_test") {
			return tpkgs[1], nil
		} else if strings.HasSuffix(tpkgs[1].Name(), "_test") {
			return tpkgs[0], nil
		}
	}

	return nil, fmt.Errorf("unexpected number of packages (%d) for %q (directory %q)", len(tpkgs), importPath, pdir)
}

// findFromPath looks for a directory under gopath.
func (imp *Importer) findFromPath(gopath, dir string) string {
	if filepath.IsAbs(dir) || build.IsLocalImport(dir) {
		return ""
	}
	for _, pd := range strings.Split(gopath, ":") {
		d := filepath.Join(pd, "src", dir)
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			return d
		}
	}
	return ""
}

// localImport handles a local import such as
//     import "./a"
// This is for tests that use directives like //compiledir.
func (imp *Importer) localImport(importPath, dir string) (*types.Package, error) {
	tpkg, ok := imp.packages[strings.TrimPrefix(importPath, "./")]
	if !ok {
		return nil, fmt.Errorf("cannot find local import %q", importPath)
	}
	return tpkg, nil
}

// register records information for a package, for use when working
// with packages that import this one.
func (imp *Importer) register(pkgfiles []namedAST, tpkg *types.Package) {
	imp.packages[tpkg.Path()] = tpkg
	for _, nast := range pkgfiles {
		imp.addIDs(nast.ast)
	}
}

// addIDs finds IDs for generic functions and types and adds them to a map.
func (imp *Importer) addIDs(f *ast.File) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if isParameterizedFuncDecl(decl, imp.info) {
				obj, ok := imp.info.Defs[decl.Name]
				if !ok {
					panic(fmt.Sprintf("no types.Object for %q", decl.Name.Name))
				}
				imp.idToFunc[obj] = decl
			}
		case *ast.GenDecl:
			if decl.Tok == token.TYPE {
				for _, s := range decl.Specs {
					ts := s.(*ast.TypeSpec)
					obj, ok := imp.info.Defs[ts.Name]
					if !ok {
						panic(fmt.Sprintf("no types.Object for %q", ts.Name.Name))
					}
					imp.idToTypeSpec[obj] = ts
				}
			}
		}
	}
}

// lookupPackage looks up a package by path.
func (imp *Importer) lookupPackage(path string) (*types.Package, bool) {
	pkg, ok := imp.packages[strings.TrimPrefix(path, "./")]
	return pkg, ok
}

// lookupFunc looks up a function by Object.
func (imp *Importer) lookupFunc(obj types.Object) (*ast.FuncDecl, bool) {
	decl, ok := imp.idToFunc[obj]
	return decl, ok
}

// lookupTypeSpec looks up a type by Object.
func (imp *Importer) lookupTypeSpec(obj types.Object) (*ast.TypeSpec, bool) {
	ts, ok := imp.idToTypeSpec[obj]
	return ts, ok
}
