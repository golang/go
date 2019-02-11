// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"log"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config packages.Config

	files map[source.URI]*File
}

// NewView creates a new View, given a root path and go/packages configuration.
// If config is nil, one is created with the directory set to the rootPath.
func NewView(config *packages.Config) *View {
	return &View{
		Config: *config,
		files:  make(map[source.URI]*File),
	}
}

func (v *View) FileSet() *token.FileSet {
	return v.Config.Fset
}

// SetContent sets the overlay contents for a file. A nil content value will
// remove the file from the active set and revert it to its on-disk contents.
func (v *View) SetContent(ctx context.Context, uri source.URI, content []byte) (source.View, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	f := v.getFile(uri)
	f.content = content

	// Resetting the contents invalidates the ast, token, and pkg fields.
	f.ast = nil
	f.token = nil
	f.pkg = nil

	// We might need to update the overlay.
	switch {
	case f.active && content == nil:
		// The file was active, so we need to forget its content.
		f.active = false
		if filename, err := f.URI.Filename(); err == nil {
			delete(f.view.Config.Overlay, filename)
		}
		f.content = nil
	case content != nil:
		// This is an active overlay, so we update the map.
		f.active = true
		if filename, err := f.URI.Filename(); err == nil {
			f.view.Config.Overlay[filename] = f.content
		}
	}

	// TODO(rstambler): We should really return a new, updated view.
	return v, nil
}

// GetFile returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *View) GetFile(ctx context.Context, uri source.URI) (source.File, error) {
	v.mu.Lock()
	f := v.getFile(uri)
	v.mu.Unlock()
	return f, nil
}

// getFile is the unlocked internal implementation of GetFile.
func (v *View) getFile(uri source.URI) *File {
	f, found := v.files[uri]
	if !found {
		f = &File{
			URI:  uri,
			view: v,
		}
		v.files[uri] = f
	}
	return f
}

func (v *View) parse(uri source.URI) error {
	path, err := uri.Filename()
	if err != nil {
		return err
	}
	pkgs, err := packages.Load(&v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", path)
		}
		return err
	}
	for _, pkg := range pkgs {
		imp := &importer{
			entries:         make(map[string]*entry),
			packages:        make(map[string]*packages.Package),
			v:               v,
			topLevelPkgPath: pkg.PkgPath,
		}
		if err := imp.addImports(pkg); err != nil {
			return err
		}

		// TODO(rstambler): Get real TypeSizes from go/packages.
		pkg.TypesSizes = &types.StdSizes{}

		imp.importPackage(pkg.PkgPath)
	}
	return nil
}

type importer struct {
	mu              sync.Mutex
	entries         map[string]*entry
	packages        map[string]*packages.Package
	topLevelPkgPath string

	v *View
}

type entry struct {
	pkg   *types.Package
	err   error
	ready chan struct{}
}

func (imp *importer) addImports(pkg *packages.Package) error {
	imp.packages[pkg.PkgPath] = pkg
	for _, i := range pkg.Imports {
		if i.PkgPath == pkg.PkgPath {
			return fmt.Errorf("import cycle: [%v]", pkg.PkgPath)
		}
		if err := imp.addImports(i); err != nil {
			return err
		}
	}
	return nil
}

func (imp *importer) Import(path string) (*types.Package, error) {
	if path == imp.topLevelPkgPath {
		return nil, fmt.Errorf("import cycle: [%v]", path)
	}
	imp.mu.Lock()
	e, ok := imp.entries[path]
	if ok {
		// cache hit
		imp.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		imp.entries[path] = e
		imp.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = imp.importPackage(path)
		close(e.ready)
	}
	return e.pkg, e.err
}

func (imp *importer) importPackage(pkgPath string) (*types.Package, error) {
	imp.mu.Lock()
	pkg, ok := imp.packages[pkgPath]
	imp.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no metadata for %v", pkgPath)
	}
	pkg.Fset = imp.v.Config.Fset
	pkg.Syntax = make([]*ast.File, len(pkg.GoFiles))
	for i, filename := range pkg.GoFiles {
		var src interface{}
		overlay, ok := imp.v.Config.Overlay[filename]
		if ok {
			src = overlay
		}
		file, err := parser.ParseFile(imp.v.Config.Fset, filename, src, parser.AllErrors|parser.ParseComments)
		if file == nil {
			return nil, err
		}
		if err != nil {
			switch err := err.(type) {
			case *scanner.Error:
				pkg.Errors = append(pkg.Errors, packages.Error{
					Pos:  err.Pos.String(),
					Msg:  err.Msg,
					Kind: packages.ParseError,
				})
			case scanner.ErrorList:
				// The first parser error is likely the root cause of the problem.
				if err.Len() > 0 {
					pkg.Errors = append(pkg.Errors, packages.Error{
						Pos:  err[0].Pos.String(),
						Msg:  err[0].Msg,
						Kind: packages.ParseError,
					})
				}
			}
		}
		pkg.Syntax[i] = file
	}
	cfg := &types.Config{
		Error: func(err error) {
			if err, ok := err.(types.Error); ok {
				pkg.Errors = append(pkg.Errors, packages.Error{
					Pos:  imp.v.Config.Fset.Position(err.Pos).String(),
					Msg:  err.Msg,
					Kind: packages.TypeError,
				})
			}
		},
		Importer: imp,
	}
	pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)
	pkg.TypesInfo = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
	}
	check := types.NewChecker(cfg, imp.v.Config.Fset, pkg.Types, pkg.TypesInfo)
	check.Files(pkg.Syntax)

	// Add every file in this package to our cache.
	for _, file := range pkg.Syntax {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			log.Printf("invalid position for file %v", file.Name)
			continue
		}
		tok := imp.v.Config.Fset.File(file.Pos())
		if tok == nil {
			log.Printf("no token.File for %v", file.Name)
			continue
		}
		fURI := source.ToURI(tok.Name())
		f := imp.v.getFile(fURI)
		f.token = tok
		f.ast = file
		f.pkg = pkg
	}
	return pkg.Types, nil
}
