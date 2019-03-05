package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
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

func (v *View) parse(ctx context.Context, uri source.URI) error {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// Apply any queued-up content changes.
	if err := v.applyContentChanges(ctx); err != nil {
		return err
	}

	f := v.files[uri]

	// This should never happen.
	if f == nil {
		return fmt.Errorf("no file for %v", uri)
	}
	// If the package for the file has not been invalidated by the application
	// of the pending changes, there is no need to continue.
	if f.isPopulated() {
		return nil
	}
	// Check if the file's imports have changed. If they have, update the
	// metadata by calling packages.Load.
	if err := v.checkMetadata(ctx, f); err != nil {
		return err
	}
	if f.meta == nil {
		return fmt.Errorf("no metadata found for %v", uri)
	}
	// Start prefetching direct imports.
	for importPath := range f.meta.children {
		go v.Import(importPath)
	}
	// Type-check package.
	pkg, err := v.typeCheck(f.meta.pkgPath)
	if pkg == nil || pkg.Types == nil {
		return err
	}
	// Add every file in this package to our cache.
	v.cachePackage(pkg)

	// If we still have not found the package for the file, something is wrong.
	if f.pkg == nil {
		return fmt.Errorf("no package found for %v", uri)
	}
	return nil
}

func (v *View) cachePackage(pkg *packages.Package) {
	for _, file := range pkg.Syntax {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			log.Printf("invalid position for file %v", file.Name)
			continue
		}
		tok := v.Config.Fset.File(file.Pos())
		if tok == nil {
			log.Printf("no token.File for %v", file.Name)
			continue
		}
		fURI := source.ToURI(tok.Name())
		f := v.getFile(fURI)
		f.token = tok
		f.ast = file
		f.imports = f.ast.Imports
		f.pkg = pkg
	}
}

func (v *View) checkMetadata(ctx context.Context, f *File) error {
	filename, err := f.URI.Filename()
	if err != nil {
		return err
	}
	if v.reparseImports(ctx, f, filename) {
		cfg := v.Config
		cfg.Mode = packages.LoadImports
		pkgs, err := packages.Load(&cfg, fmt.Sprintf("file=%s", filename))
		if len(pkgs) == 0 {
			if err == nil {
				err = fmt.Errorf("no packages found for %s", filename)
			}
			return err
		}
		for _, pkg := range pkgs {
			// If the package comes back with errors from `go list`, don't bother
			// type-checking it.
			for _, err := range pkg.Errors {
				switch err.Kind {
				case packages.UnknownError, packages.ListError:
					return err
				}
			}
			v.link(pkg.PkgPath, pkg, nil)
		}
	}
	return nil
}

// reparseImports reparses a file's import declarations to determine if they
// have changed.
func (v *View) reparseImports(ctx context.Context, f *File, filename string) bool {
	if f.meta == nil {
		return true
	}
	// Get file content in case we don't already have it?
	f.read(ctx)
	parsed, _ := parser.ParseFile(v.Config.Fset, filename, f.content, parser.ImportsOnly)
	if parsed == nil {
		return true
	}
	if len(f.imports) != len(parsed.Imports) {
		return true
	}
	for i, importSpec := range f.imports {
		if importSpec.Path.Value != f.imports[i].Path.Value {
			return true
		}
	}
	return false
}

func (v *View) link(pkgPath string, pkg *packages.Package, parent *metadata) *metadata {
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		m = &metadata{
			pkgPath:  pkgPath,
			id:       pkg.ID,
			parents:  make(map[string]bool),
			children: make(map[string]bool),
		}
		v.mcache.packages[pkgPath] = m
	}
	// Reset any field that could have changed across calls to packages.Load.
	m.name = pkg.Name
	m.files = pkg.CompiledGoFiles
	for _, filename := range m.files {
		if f, ok := v.files[source.ToURI(filename)]; ok {
			f.meta = m
		}
	}
	// Connect the import graph.
	if parent != nil {
		m.parents[parent.pkgPath] = true
		parent.children[pkgPath] = true
	}
	for importPath, importPkg := range pkg.Imports {
		if _, ok := m.children[importPath]; !ok {
			v.link(importPath, importPkg, m)
		}
	}
	// Clear out any imports that have been removed.
	for importPath := range m.children {
		if _, ok := pkg.Imports[importPath]; !ok {
			delete(m.children, importPath)
			if child, ok := v.mcache.packages[importPath]; ok {
				delete(child.parents, pkgPath)
			}
		}
	}
	return m
}

func (v *View) Import(pkgPath string) (*types.Package, error) {
	v.pcache.mu.Lock()
	e, ok := v.pcache.packages[pkgPath]
	if ok {
		// cache hit
		v.pcache.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		v.pcache.packages[pkgPath] = e
		v.pcache.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = v.typeCheck(pkgPath)
		close(e.ready)
	}
	if e.err != nil {
		return nil, e.err
	}
	return e.pkg.Types, nil
}

func (v *View) typeCheck(pkgPath string) (*packages.Package, error) {
	meta, ok := v.mcache.packages[pkgPath]
	if !ok {
		return nil, fmt.Errorf("no metadata for %v", pkgPath)
	}
	// Use the default type information for the unsafe package.
	var typ *types.Package
	if meta.pkgPath == "unsafe" {
		typ = types.Unsafe
	} else {
		typ = types.NewPackage(meta.pkgPath, meta.name)
	}
	pkg := &packages.Package{
		ID:              meta.id,
		Name:            meta.name,
		PkgPath:         meta.pkgPath,
		CompiledGoFiles: meta.files,
		Imports:         make(map[string]*packages.Package),
		Fset:            v.Config.Fset,
		Types:           typ,
		TypesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		// TODO(rstambler): Get real TypeSizes from go/packages (golang.org/issues/30139).
		TypesSizes: &types.StdSizes{},
	}
	appendError := func(err error) {
		v.appendPkgError(pkg, err)
	}
	files, errs := v.parseFiles(meta.files)
	for _, err := range errs {
		appendError(err)
	}
	pkg.Syntax = files
	cfg := &types.Config{
		Error:    appendError,
		Importer: v,
	}
	check := types.NewChecker(cfg, v.Config.Fset, pkg.Types, pkg.TypesInfo)
	check.Files(pkg.Syntax)

	// Set imports of package to correspond to cached packages. This is
	// necessary for go/analysis, but once we merge its approach with the
	// current caching system, we can eliminate this.
	v.pcache.mu.Lock()
	for importPath := range meta.children {
		if importEntry, ok := v.pcache.packages[importPath]; ok {
			pkg.Imports[importPath] = importEntry.pkg
		}
	}
	v.pcache.mu.Unlock()

	return pkg, nil
}

func (v *View) appendPkgError(pkg *packages.Package, err error) {
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
			Pos:  v.Config.Fset.Position(err.Pos).String(),
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
func (v *View) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, filename := range filenames {
		if v.Config.Context.Err() != nil {
			parsed[i] = nil
			errors[i] = v.Config.Context.Err()
			continue
		}

		// First, check if we have already cached an AST for this file.
		f := v.files[source.ToURI(filename)]
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
				for f, contents := range v.Config.Overlay {
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
					parsed[i], errors[i] = v.Config.ParseFile(v.Config.Fset, filename, src)
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
