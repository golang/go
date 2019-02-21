// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config packages.Config

	files map[source.URI]*File

	analysisCache *source.AnalysisCache
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

func (v *View) GetAnalysisCache() *source.AnalysisCache {
	v.analysisCache = source.NewAnalysisCache()
	return v.analysisCache
}

// SetContent sets the overlay contents for a file. A nil content value will
// remove the file from the active set and revert it to its on-disk contents.
func (v *View) SetContent(ctx context.Context, uri source.URI, content []byte) (source.View, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	newView := NewView(&v.Config)

	for fURI, f := range v.files {
		newView.files[fURI] = &File{
			URI:     fURI,
			view:    newView,
			active:  f.active,
			content: f.content,
			ast:     f.ast,
			token:   f.token,
			pkg:     f.pkg,
		}
	}

	f := newView.getFile(uri)
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

	return newView, nil
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
	// TODO(rstambler): Enforce here that LoadMode is LoadImports?
	pkgs, err := packages.Load(&v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", path)
		}
		return err
	}
	var foundPkg bool // true if we found the package for uri
	for _, pkg := range pkgs {
		// TODO(rstambler): Get real TypeSizes from go/packages (golang.org/issues/30139).
		pkg.TypesSizes = &types.StdSizes{}

		imp := &importer{
			entries:         make(map[string]*entry),
			packages:        make(map[string]*packages.Package),
			v:               v,
			topLevelPkgPath: pkg.PkgPath,
		}
		if err := imp.addImports(pkg.PkgPath, pkg); err != nil {
			return err
		}
		// Start prefetching direct imports.
		for importPath := range pkg.Imports {
			go imp.Import(importPath)
		}
		imp.importPackage(pkg.PkgPath)

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
			if fURI == uri {
				foundPkg = true
			}
			f := imp.v.getFile(fURI)
			f.token = tok
			f.ast = file
			f.pkg = pkg
		}
	}
	if !foundPkg {
		return fmt.Errorf("no package found for %v", uri)
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

func (imp *importer) addImports(path string, pkg *packages.Package) error {
	if _, ok := imp.packages[path]; ok {
		return nil
	}
	imp.packages[path] = pkg
	for importPath, importPkg := range pkg.Imports {
		if err := imp.addImports(importPath, importPkg); err != nil {
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
	appendError := func(err error) {
		imp.appendPkgError(pkg, err)
	}
	files, errs := imp.parseFiles(pkg.CompiledGoFiles)
	for _, err := range errs {
		appendError(err)
	}
	pkg.Syntax = files
	cfg := &types.Config{
		Error:    appendError,
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

	return pkg.Types, nil
}

func (imp *importer) appendPkgError(pkg *packages.Package, err error) {
	if err == nil {
		return
	}
	var errs []packages.Error
	switch err := err.(type) {
	case *scanner.Error:
		errs = append(errs, packages.Error{
			Pos:  err.Pos.String(),
			Msg:  err.Msg,
			Kind: packages.ParseError,
		})
	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if err.Len() > 0 {
			errs = append(errs, packages.Error{
				Pos:  err[0].Pos.String(),
				Msg:  err[0].Msg,
				Kind: packages.ParseError,
			})
		}
	case types.Error:
		errs = append(errs, packages.Error{
			Pos:  imp.v.Config.Fset.Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})
	}
	pkg.Errors = append(pkg.Errors, errs...)
}

// We use a counting semaphore to limit
// the number of parallel I/O calls per process.
var ioLimit = make(chan bool, 20)

// parseFiles reads and parses the Go source files and returns the ASTs
// of the ones that could be at least partially parsed, along with a
// list of I/O and parse errors encountered.
//
// Because files are scanned in parallel, the token.Pos
// positions of the resulting ast.Files are not ordered.
//
func (imp *importer) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, filename := range filenames {
		if imp.v.Config.Context.Err() != nil {
			parsed[i] = nil
			errors[i] = imp.v.Config.Context.Err()
			continue
		}

		// First, check if we have already cached an AST for this file.
		f := imp.v.files[source.ToURI(filename)]
		var fAST *ast.File
		if f != nil {
			fAST = f.ast
		}

		wg.Add(1)
		go func(i int, filename string) {
			ioLimit <- true // wait

			if fAST != nil {
				parsed[i], errors[i] = fAST, nil
			} else {
				// We don't have a cached AST for this file.
				var src []byte
				// Check for an available overlay.
				for f, contents := range imp.v.Config.Overlay {
					if sameFile(f, filename) {
						src = contents
					}
				}
				var err error
				// We don't have an overlay, so we must read the file's contents.
				if src == nil {
					src, err = ioutil.ReadFile(filename)
				}
				if err != nil {
					parsed[i], errors[i] = nil, err
				} else {
					// ParseFile may return both an AST and an error.
					parsed[i], errors[i] = imp.v.Config.ParseFile(imp.v.Config.Fset, filename, src)
				}
			}

			<-ioLimit // signal
			wg.Done()
		}(i, filename)
	}
	wg.Wait()

	// Eliminate nils, preserving order.
	var o int
	for _, f := range parsed {
		if f != nil {
			parsed[o] = f
			o++
		}
	}
	parsed = parsed[:o]

	o = 0
	for _, err := range errors {
		if err != nil {
			errors[o] = err
			o++
		}
	}
	errors = errors[:o]

	return parsed, errors
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if x == y {
		// It could be the case that y doesn't exist.
		// For instance, it may be an overlay file that
		// hasn't been written to disk. To handle that case
		// let x == y through. (We added the exact absolute path
		// string to the CompiledGoFiles list, so the unwritten
		// overlay case implies x==y.)
		return true
	}
	if strings.EqualFold(filepath.Base(x), filepath.Base(y)) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}
