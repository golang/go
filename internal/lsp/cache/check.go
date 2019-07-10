// Copyright 2019 The Go Authors. All rights reserved.
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
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

type importer struct {
	view *view

	// seen maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	seen map[packageID]struct{}

	// topLevelPkgID is the ID of the package from which type-checking began.
	topLevelPkgID packageID

	ctx  context.Context
	fset *token.FileSet
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	ctx := imp.ctx
	id, ok := imp.view.mcache.ids[packagePath(pkgPath)]
	if !ok {
		return nil, fmt.Errorf("no known ID for %s", pkgPath)
	}
	pkg, err := imp.getPkg(ctx, id)
	if err != nil {
		return nil, err
	}
	return pkg.types, nil
}

func (imp *importer) getPkg(ctx context.Context, id packageID) (*pkg, error) {
	if _, ok := imp.seen[id]; ok {
		return nil, fmt.Errorf("circular import detected")
	}
	imp.view.pcache.mu.Lock()
	e, ok := imp.view.pcache.packages[id]

	if ok {
		// cache hit
		imp.view.pcache.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		imp.view.pcache.packages[id] = e
		imp.view.pcache.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = imp.typeCheck(ctx, id)
		if e.err != nil {
			// Don't cache failed packages. If we didn't successfully cache the package
			// in each file, then this pcache entry won't get invalidated as those files
			// change.
			imp.view.pcache.mu.Lock()
			if imp.view.pcache.packages[id] == e {
				delete(imp.view.pcache.packages, id)
			}
			imp.view.pcache.mu.Unlock()
		}
		close(e.ready)
	}

	if e.err != nil {
		// If the import had been previously canceled, and that error cached, try again.
		if e.err == context.Canceled && ctx.Err() == nil {
			return imp.getPkg(ctx, id)
		}
		return nil, e.err
	}

	return e.pkg, nil
}

func (imp *importer) typeCheck(ctx context.Context, id packageID) (*pkg, error) {
	ctx, done := trace.StartSpan(ctx, "cache.importer.typeCheck", telemetry.Package.Of(id))
	defer done()
	meta, ok := imp.view.mcache.packages[id]
	if !ok {
		return nil, fmt.Errorf("no metadata for %v", id)
	}
	pkg := &pkg{
		id:         meta.id,
		pkgPath:    meta.pkgPath,
		imports:    make(map[packagePath]*pkg),
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

	// Ignore function bodies for any dependency packages.
	mode := source.ParseFull
	if imp.topLevelPkgID != pkg.id {
		mode = source.ParseExported
	}
	var (
		files  = make([]*ast.File, len(meta.files))
		errors = make([]error, len(meta.files))
		wg     sync.WaitGroup
	)
	for _, filename := range meta.files {
		uri := span.FileURI(filename)
		f, err := imp.view.getFile(ctx, uri)
		if err != nil {
			log.Error(ctx, "unable to get file", err, telemetry.File.Of(f.URI()))
			continue
		}
		pkg.files = append(pkg.files, imp.view.session.cache.ParseGoHandle(f.Handle(ctx), mode))
	}
	for i, ph := range pkg.files {
		wg.Add(1)
		go func(i int, ph source.ParseGoHandle) {
			defer wg.Done()

			files[i], errors[i] = ph.Parse(ctx)
		}(i, ph)
	}
	wg.Wait()

	var i int
	for _, f := range files {
		if f != nil {
			files[i] = f
			i++
		}
	}
	for _, err := range errors {
		if err == context.Canceled {
			return nil, err
		}
		if err != nil {
			imp.view.session.cache.appendPkgError(pkg, err)
		}
	}

	// Use the default type information for the unsafe package.
	if meta.pkgPath == "unsafe" {
		pkg.types = types.Unsafe
	} else if len(files) == 0 { // not the unsafe package, no parsed files
		return nil, fmt.Errorf("no parsed files for package %s", pkg.pkgPath)
	} else {
		pkg.types = types.NewPackage(string(meta.pkgPath), meta.name)
	}

	// Handle circular imports by copying previously seen imports.
	seen := make(map[packageID]struct{})
	for k, v := range imp.seen {
		seen[k] = v
	}
	seen[id] = struct{}{}

	cfg := &types.Config{
		Error: func(err error) {
			imp.view.session.cache.appendPkgError(pkg, err)
		},
		IgnoreFuncBodies: mode == source.ParseExported,
		Importer: &importer{
			view:          imp.view,
			ctx:           ctx,
			fset:          imp.fset,
			topLevelPkgID: imp.topLevelPkgID,
			seen:          seen,
		},
	}
	check := types.NewChecker(cfg, imp.fset, pkg.types, pkg.typesInfo)

	// Ignore type-checking errors.
	check.Files(files)

	// Add every file in this package to our cache.
	if err := imp.cachePackage(ctx, pkg, meta, mode); err != nil {
		return nil, err
	}

	return pkg, nil
}

func (imp *importer) cachePackage(ctx context.Context, pkg *pkg, meta *metadata, mode source.ParseMode) error {
	for _, ph := range pkg.files {
		uri := ph.File().Identity().URI
		f, err := imp.view.getFile(ctx, uri)
		if err != nil {
			return fmt.Errorf("no such file %s: %v", uri, err)
		}
		gof, ok := f.(*goFile)
		if !ok {
			return fmt.Errorf("non Go file %s", uri)
		}
		if err := imp.cachePerFile(gof, ph, pkg); err != nil {
			return fmt.Errorf("failed to cache file %s: %v", gof.URI(), err)
		}
	}

	// Set imports of package to correspond to cached packages.
	// We lock the package cache, but we shouldn't get any inconsistencies
	// because we are still holding the lock on the view.
	for importPath := range meta.children {
		importPkg, err := imp.getPkg(ctx, importPath)
		if err != nil {
			continue
		}
		pkg.imports[importPkg.pkgPath] = importPkg
	}

	return nil
}

func (imp *importer) cachePerFile(gof *goFile, ph source.ParseGoHandle, p *pkg) error {
	gof.mu.Lock()
	defer gof.mu.Unlock()

	// Set the package even if we failed to parse the file.
	if gof.pkgs == nil {
		gof.pkgs = make(map[packageID]*pkg)
	}
	gof.pkgs[p.id] = p

	file, err := ph.Parse(imp.ctx)
	if file == nil {
		return fmt.Errorf("no AST for %s: %v", ph.File().Identity().URI, err)
	}
	gof.imports = file.Imports
	return nil
}

func (c *cache) appendPkgError(pkg *pkg, err error) {
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
			Pos:  c.FileSet().Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})
	}
	pkg.errors = append(pkg.errors, errs...)
}
