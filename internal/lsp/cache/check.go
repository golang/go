package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/span"
)

func (v *View) parse(ctx context.Context, f *File) ([]packages.Error, error) {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// Apply any queued-up content changes.
	if err := v.applyContentChanges(ctx); err != nil {
		return nil, err
	}

	// If the package for the file has not been invalidated by the application
	// of the pending changes, there is no need to continue.
	if f.isPopulated() {
		return nil, nil
	}
	// Check if the file's imports have changed. If they have, update the
	// metadata by calling packages.Load.
	if errs, err := v.checkMetadata(ctx, f); err != nil {
		return errs, err
	}
	if f.meta == nil {
		return nil, fmt.Errorf("no metadata found for %v", f.filename)
	}
	imp := &importer{
		view:     v,
		circular: make(map[string]struct{}),
		ctx:      ctx,
	}
	// Start prefetching direct imports.
	for importPath := range f.meta.children {
		go imp.Import(importPath)
	}
	// Type-check package.
	pkg, err := imp.typeCheck(f.meta.pkgPath)
	if pkg == nil || pkg.GetTypes() == nil {
		return nil, err
	}

	// If we still have not found the package for the file, something is wrong.
	if f.pkg == nil {
		return nil, fmt.Errorf("parse: no package found for %v", f.filename)
	}
	return nil, nil
}

func (v *View) cachePackage(ctx context.Context, pkg *Package) {
	for _, file := range pkg.GetSyntax() {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			v.Logger().Errorf(ctx, "invalid position for file %v", file.Name)
			continue
		}
		tok := v.Config.Fset.File(file.Pos())
		if tok == nil {
			v.Logger().Errorf(ctx, "no token.File for %v", file.Name)
			continue
		}
		fURI := span.FileURI(tok.Name())
		f, err := v.getFile(fURI)
		if err != nil {
			v.Logger().Errorf(ctx, "no file: %v", err)
			continue
		}
		f.token = tok
		f.ast = file
		f.imports = f.ast.Imports
		f.pkg = pkg
	}
}

func (v *View) checkMetadata(ctx context.Context, f *File) ([]packages.Error, error) {
	if v.reparseImports(ctx, f, f.filename) {
		cfg := v.Config
		cfg.Mode = packages.LoadImports | packages.NeedTypesSizes
		pkgs, err := packages.Load(&cfg, fmt.Sprintf("file=%s", f.filename))
		if len(pkgs) == 0 {
			if err == nil {
				err = fmt.Errorf("no packages found for %s", f.filename)
			}
			return nil, err
		}
		for _, pkg := range pkgs {
			// If the package comes back with errors from `go list`, don't bother
			// type-checking it.
			if len(pkg.Errors) > 0 {
				return pkg.Errors, fmt.Errorf("package %s has errors, skipping type-checking", pkg.PkgPath)
			}
			v.link(pkg.PkgPath, pkg, nil)
		}
	}
	return nil, nil
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
			pkgPath:    pkgPath,
			id:         pkg.ID,
			typesSizes: pkg.TypesSizes,
			parents:    make(map[string]bool),
			children:   make(map[string]bool),
		}
		v.mcache.packages[pkgPath] = m
	}
	// Reset any field that could have changed across calls to packages.Load.
	m.name = pkg.Name
	m.files = pkg.CompiledGoFiles
	for _, filename := range m.files {
		if f, _ := v.getFile(span.FileURI(filename)); f != nil {
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

type importer struct {
	view *View

	// circular maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	circular map[string]struct{}

	ctx context.Context
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	if _, ok := imp.circular[pkgPath]; ok {
		return nil, fmt.Errorf("circular import detected")
	}
	imp.view.pcache.mu.Lock()
	e, ok := imp.view.pcache.packages[pkgPath]
	if ok {
		// cache hit
		imp.view.pcache.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		imp.view.pcache.packages[pkgPath] = e
		imp.view.pcache.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = imp.typeCheck(pkgPath)
		close(e.ready)
	}
	if e.err != nil {
		return nil, e.err
	}
	return e.pkg.types, nil
}

func (imp *importer) typeCheck(pkgPath string) (*Package, error) {
	meta, ok := imp.view.mcache.packages[pkgPath]
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
	pkg := &Package{
		id:         meta.id,
		pkgPath:    meta.pkgPath,
		files:      meta.files,
		imports:    make(map[string]*Package),
		types:      typ,
		typesSizes: meta.typesSizes,
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		analyses: make(map[*analysis.Analyzer]*analysisEntry),
	}
	appendError := func(err error) {
		imp.view.appendPkgError(pkg, err)
	}
	files, errs := imp.view.parseFiles(meta.files)
	for _, err := range errs {
		appendError(err)
	}
	pkg.syntax = files

	// Handle circular imports by copying previously seen imports.
	newCircular := copySet(imp.circular)
	newCircular[pkgPath] = struct{}{}

	cfg := &types.Config{
		Error: appendError,
		Importer: &importer{
			view:     imp.view,
			circular: newCircular,
			ctx:      imp.ctx,
		},
	}
	check := types.NewChecker(cfg, imp.view.Config.Fset, pkg.types, pkg.typesInfo)
	check.Files(pkg.syntax)

	// Add every file in this package to our cache.
	imp.view.cachePackage(imp.ctx, pkg)

	// Set imports of package to correspond to cached packages.
	// We lock the package cache, but we shouldn't get any inconsistencies
	// because we are still holding the lock on the view.
	imp.view.pcache.mu.Lock()
	defer imp.view.pcache.mu.Unlock()

	for importPath := range meta.children {
		if importEntry, ok := imp.view.pcache.packages[importPath]; ok {
			pkg.imports[importPath] = importEntry.pkg
		}
	}

	return pkg, nil
}

func copySet(m map[string]struct{}) map[string]struct{} {
	result := make(map[string]struct{})
	for k, v := range m {
		result[k] = v
	}
	return result
}

func (v *View) appendPkgError(pkg *Package, err error) {
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
	pkg.errors = append(pkg.errors, errs...)
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
		f, err := v.findFile(span.FileURI(filename))
		if err != nil {
			parsed[i], errors[i] = nil, err
		}
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
