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

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/span"
)

type importer struct {
	view *view

	// seen maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	seen map[string]struct{}

	// topLevelPkgPath is the path of the package from which type-checking began.
	topLevelPkgPath string

	ctx  context.Context
	fset *token.FileSet
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	pkg, err := imp.getPkg(pkgPath)
	if err != nil {
		return nil, err
	}
	return pkg.types, nil
}

func (imp *importer) getPkg(pkgPath string) (*pkg, error) {
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
	return e.pkg, nil
}

func (imp *importer) typeCheck(pkgPath string) (*pkg, error) {
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
	pkg := &pkg{
		id:         meta.id,
		pkgPath:    meta.pkgPath,
		files:      meta.files,
		imports:    make(map[string]*pkg),
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

	// Don't type-check function bodies if we are not in the top-level package.
	files, errs := imp.parseFiles(meta.files, imp.ignoreFuncBodies(pkg.pkgPath))
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
			view:            imp.view,
			ctx:             imp.ctx,
			fset:            imp.fset,
			topLevelPkgPath: imp.topLevelPkgPath,
			seen:            seen,
		},
	}
	check := types.NewChecker(cfg, imp.fset, pkg.types, pkg.typesInfo)
	check.Files(pkg.syntax)

	// Add every file in this package to our cache.
	imp.cachePackage(imp.ctx, pkg, meta)

	return pkg, nil
}

func (imp *importer) cachePackage(ctx context.Context, pkg *pkg, meta *metadata) {
	for _, file := range pkg.GetSyntax() {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			imp.view.Session().Logger().Errorf(ctx, "invalid position for file %v", file.Name)
			continue
		}
		tok := imp.view.Session().Cache().FileSet().File(file.Pos())
		if tok == nil {
			imp.view.Session().Logger().Errorf(ctx, "no token.File for %v", file.Name)
			continue
		}
		fURI := span.FileURI(tok.Name())
		f, err := imp.view.getFile(fURI)
		if err != nil {
			imp.view.Session().Logger().Errorf(ctx, "no file: %v", err)
			continue
		}
		gof, ok := f.(*goFile)
		if !ok {
			imp.view.Session().Logger().Errorf(ctx, "not a go file: %v", f.URI())
			continue
		}
		gof.token = tok
		gof.ast = &astFile{
			file:      file,
			isTrimmed: imp.ignoreFuncBodies(pkg.pkgPath),
		}
		gof.imports = file.Imports
		gof.pkg = pkg
	}

	// Set imports of package to correspond to cached packages.
	// We lock the package cache, but we shouldn't get any inconsistencies
	// because we are still holding the lock on the view.
	for importPath := range meta.children {
		importPkg, err := imp.getPkg(importPath)
		if err != nil {
			continue
		}
		pkg.imports[importPath] = importPkg
	}
}

func (v *view) appendPkgError(pkg *pkg, err error) {
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
			Pos:  v.Session().Cache().FileSet().Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})
	}
	pkg.errors = append(pkg.errors, errs...)
}

func (imp *importer) ignoreFuncBodies(pkgPath string) bool {
	return imp.topLevelPkgPath != pkgPath
}
