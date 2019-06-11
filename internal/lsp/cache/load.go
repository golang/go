package cache

import (
	"context"
	"fmt"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (v *view) loadParseTypecheck(ctx context.Context, f *goFile) ([]packages.Error, error) {
	v.mcache.mu.Lock()
	defer v.mcache.mu.Unlock()

	// If the AST for this file is trimmed, and we are explicitly type-checking it,
	// don't ignore function bodies.
	if f.astIsTrimmed() {
		f.invalidateAST()
	}

	// Save the metadata's current missing imports, if any.
	var originalMissingImports map[packagePath]struct{}
	if f.meta != nil {
		originalMissingImports = f.meta.missingImports
	}
	// Check if we need to run go/packages.Load for this file's package.
	if errs, err := v.checkMetadata(ctx, f); err != nil {
		return errs, err
	}
	// If `go list` failed to get data for the file in question (this should never happen).
	if f.meta == nil {
		return nil, fmt.Errorf("loadParseTypecheck: no metadata found for %v", f.filename())
	}
	// If we have already seen these missing imports before, and we still have type information,
	// there is no need to continue.
	if sameSet(originalMissingImports, f.meta.missingImports) && f.pkg != nil {
		return nil, nil
	}

	imp := &importer{
		view:          v,
		seen:          make(map[packagePath]struct{}),
		ctx:           ctx,
		fset:          f.FileSet(),
		topLevelPkgID: f.meta.id,
	}

	// Start prefetching direct imports.
	for importPath := range f.meta.children {
		go imp.Import(string(importPath))
	}
	// Type-check package.
	pkg, err := imp.getPkg(f.meta.pkgPath)
	if err != nil {
		return nil, err
	}
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("loadParseTypecheck: %s is ill typed", f.meta.pkgPath)
	}
	// If we still have not found the package for the file, something is wrong.
	if f.pkg == nil {
		return nil, fmt.Errorf("loadParseTypeCheck: no package found for %v", f.filename())
	}
	return nil, nil
}

func sameSet(x, y map[packagePath]struct{}) bool {
	if len(x) != len(y) {
		return false
	}
	for k := range x {
		if _, ok := y[k]; !ok {
			return false
		}
	}
	return true
}

// checkMetadata determines if we should run go/packages.Load for this file.
// If yes, update the metadata for the file and its package.
func (v *view) checkMetadata(ctx context.Context, f *goFile) ([]packages.Error, error) {
	if !v.parseImports(ctx, f) {
		return nil, nil
	}
	pkgs, err := packages.Load(v.buildConfig(), fmt.Sprintf("file=%s", f.filename()))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", f.filename())
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
		v.link(ctx, packagePath(pkg.PkgPath), pkg, nil)
	}
	return nil, nil
}

// reparseImports reparses a file's package and import declarations to
// determine if they have changed.
func (v *view) parseImports(ctx context.Context, f *goFile) bool {
	if f.meta == nil || len(f.meta.missingImports) > 0 {
		return true
	}
	// Get file content in case we don't already have it.
	parsed, _ := v.session.cache.ParseGo(f.Handle(ctx), source.ParseHeader).Parse(ctx)
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
		if importSpec.Path.Value != parsed.Imports[i].Path.Value {
			return true
		}
	}
	return false
}

func (v *view) link(ctx context.Context, pkgPath packagePath, pkg *packages.Package, parent *metadata) *metadata {
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		m = &metadata{
			pkgPath:        pkgPath,
			id:             packageID(pkg.ID),
			typesSizes:     pkg.TypesSizes,
			parents:        make(map[packagePath]bool),
			children:       make(map[packagePath]bool),
			missingImports: make(map[packagePath]struct{}),
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
		if len(importPkg.Errors) > 0 {
			m.missingImports[pkgPath] = struct{}{}
		}
		importPkgPath := packagePath(importPath)
		if _, ok := m.children[importPkgPath]; !ok {
			v.link(ctx, importPkgPath, importPkg, m)
		}
	}
	// Clear out any imports that have been removed.
	for importPath := range m.children {
		if _, ok := pkg.Imports[string(importPath)]; !ok {
			delete(m.children, importPath)
			if child, ok := v.mcache.packages[importPath]; ok {
				delete(child.parents, pkgPath)
			}
		}
	}
	return m
}
