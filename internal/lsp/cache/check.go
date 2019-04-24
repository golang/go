package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/types"

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
		view: v,
		seen: make(map[string]struct{}),
		ctx:  ctx,
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
			// Return this error as a diagnostic to the user.
			return []packages.Error{
				{
					Msg:  err.Error(),
					Kind: packages.ListError,
				},
			}, err
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

	// seen maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	seen map[string]struct{}

	ctx context.Context
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	if _, ok := imp.seen[pkgPath]; ok {
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
	files, errs := imp.parseFiles(meta.files)
	for _, err := range errs {
		appendError(err)
	}
	pkg.syntax = files

	// Handle circular imports by copying previously seen imports.
	seen := make(map[string]struct{})
	for k, v := range imp.seen {
		seen[k] = v
	}
	seen[pkgPath] = struct{}{}

	cfg := &types.Config{
		Error: appendError,
		Importer: &importer{
			view: imp.view,
			seen: seen,
			ctx:  imp.ctx,
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
