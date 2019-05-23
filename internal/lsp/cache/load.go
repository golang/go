package cache

import (
	"context"
	"fmt"
	"go/parser"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/span"
)

func (v *view) loadParseTypecheck(ctx context.Context, f *goFile) ([]packages.Error, error) {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// Apply any queued-up content changes.
	if err := v.applyContentChanges(ctx); err != nil {
		return nil, err
	}

	// If the package for the file has not been invalidated by the application
	// of the pending changes, there is no need to continue.
	if !f.isDirty() {
		return nil, nil
	}

	// Check if we need to run go/packages.Load for this file's package.
	if errs, err := v.checkMetadata(ctx, f); err != nil {
		return errs, err
	}

	if f.meta == nil {
		return nil, fmt.Errorf("loadParseTypecheck: no metadata found for %v", f.filename())
	}

	imp := &importer{
		view:            v,
		seen:            make(map[string]struct{}),
		ctx:             ctx,
		fset:            f.FileSet(),
		topLevelPkgPath: f.meta.pkgPath,
	}

	// Start prefetching direct imports.
	for importPath := range f.meta.children {
		go imp.Import(importPath)
	}
	// Type-check package.
	pkg, err := imp.getPkg(f.meta.pkgPath)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, err
	}
	// If we still have not found the package for the file, something is wrong.
	if f.pkg == nil {
		return nil, fmt.Errorf("parse: no package found for %v", f.filename())
	}
	return nil, nil
}

// checkMetadata determines if we should run go/packages.Load for this file.
// If yes, update the metadata for the file and its package.
func (v *view) checkMetadata(ctx context.Context, f *goFile) ([]packages.Error, error) {
	if !v.reparseImports(ctx, f) {
		return nil, nil
	}
	pkgs, err := packages.Load(v.buildConfig(), fmt.Sprintf("file=%s", f.filename()))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("%s: no packages found", f.filename())
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
		// If the package comes back with errors from `go list`,
		// don't bother type-checking it.
		if len(pkg.Errors) > 0 {
			return pkg.Errors, fmt.Errorf("package %s has errors, skipping type-checking", pkg.PkgPath)
		}
		// Build the import graph for this package.
		v.link(ctx, pkg.PkgPath, pkg, nil)
	}
	return nil, nil
}

// reparseImports reparses a file's package and import declarations to
// determine if they have changed.
func (v *view) reparseImports(ctx context.Context, f *goFile) bool {
	if f.meta == nil {
		return true
	}
	// Get file content in case we don't already have it.
	f.read(ctx)
	if f.fc.Error != nil {
		return true
	}
	parsed, _ := parser.ParseFile(f.FileSet(), f.filename(), f.fc.Data, parser.ImportsOnly)
	if parsed == nil {
		return true
	}
	// If the package name has changed, re-run `go list`.
	if f.meta.name != parsed.Name.Name {
		return true
	}
	// If the package's imports have changed, re-run `go list`.
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

func (v *view) link(ctx context.Context, pkgPath string, pkg *packages.Package, parent *metadata) *metadata {
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
			if gof, ok := f.(*goFile); ok {
				gof.meta = m
			} else {
				v.Session().Logger().Errorf(ctx, "not a Go file: %s", f.URI())
			}
		}
	}
	// Connect the import graph.
	if parent != nil {
		m.parents[parent.pkgPath] = true
		parent.children[pkgPath] = true
	}
	for importPath, importPkg := range pkg.Imports {
		if _, ok := m.children[importPath]; !ok {
			v.link(ctx, importPath, importPkg, m)
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
