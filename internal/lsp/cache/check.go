// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"go/ast"
	"go/scanner"
	"go/types"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type importer struct {
	view   *view
	ctx    context.Context
	config *packages.Config

	// seen maintains the set of previously imported packages.
	// If we have seen a package that is already in this map, we have a circular import.
	seen map[packageID]struct{}

	// topLevelPackageID is the ID of the package from which type-checking began.
	topLevelPackageID packageID

	// parentPkg is the package that imports the current package.
	parentPkg *pkg

	// parentCheckPackageHandle is the check package handle that imports the current package.
	parentCheckPackageHandle *checkPackageHandle
}

// checkPackageKey uniquely identifies a package and its config.
type checkPackageKey struct {
	id     string
	files  string
	config string

	// TODO: For now, we don't include dependencies in the key.
	// This will be necessary when we change the cache invalidation logic.
}

// checkPackageHandle implements source.CheckPackageHandle.
type checkPackageHandle struct {
	handle *memoize.Handle

	files   []source.ParseGoHandle
	imports map[packagePath]*checkPackageHandle

	m      *metadata
	config *packages.Config
}

// checkPackageData contains the data produced by type-checking a package.
type checkPackageData struct {
	memoize.NoCopy

	pkg *pkg
	err error
}

func (pkg *pkg) GetImport(ctx context.Context, pkgPath string) (source.Package, error) {
	if imp := pkg.imports[packagePath(pkgPath)]; imp != nil {
		return imp, nil
	}
	// Don't return a nil pointer because that still satisfies the interface.
	return nil, errors.Errorf("no imported package for %s", pkgPath)
}

// checkPackageHandle returns a source.CheckPackageHandle for a given package and config.
func (imp *importer) checkPackageHandle(ctx context.Context, m *metadata) (*checkPackageHandle, error) {
	phs, err := imp.parseGoHandles(ctx, m)
	if err != nil {
		log.Error(ctx, "no ParseGoHandles", err, telemetry.Package.Of(m.id))
		return nil, err
	}
	key := checkPackageKey{
		id:     string(m.id),
		files:  hashParseKeys(phs),
		config: hashConfig(imp.config),
	}
	cph := &checkPackageHandle{
		m:       m,
		files:   phs,
		config:  imp.config,
		imports: make(map[packagePath]*checkPackageHandle),
	}
	h := imp.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		data := &checkPackageData{}
		data.pkg, data.err = imp.typeCheck(ctx, cph, m)
		return data
	})
	cph.handle = h
	return cph, nil
}

// hashConfig returns the hash for the *packages.Config.
func hashConfig(config *packages.Config) string {
	b := bytes.NewBuffer(nil)

	// Dir, Mode, Env, BuildFlags are the parts of the config that can change.
	b.WriteString(config.Dir)
	b.WriteString(string(config.Mode))

	for _, e := range config.Env {
		b.WriteString(e)
	}
	for _, f := range config.BuildFlags {
		b.WriteString(f)
	}
	return hashContents(b.Bytes())
}

func (cph *checkPackageHandle) Check(ctx context.Context) (source.Package, error) {
	return cph.check(ctx)
}

func (cph *checkPackageHandle) check(ctx context.Context) (*pkg, error) {
	ctx, done := trace.StartSpan(ctx, "cache.checkPackageHandle.check", telemetry.Package.Of(cph.m.id))
	defer done()

	v := cph.handle.Get(ctx)
	if v == nil {
		return nil, ctx.Err()
	}
	data := v.(*checkPackageData)
	return data.pkg, data.err
}

func (cph *checkPackageHandle) Config() *packages.Config {
	return cph.config
}

func (cph *checkPackageHandle) Files() []source.ParseGoHandle {
	return cph.files
}

func (cph *checkPackageHandle) ID() string {
	return string(cph.m.id)
}

func (cph *checkPackageHandle) Cached(ctx context.Context) (source.Package, error) {
	return cph.cached(ctx)
}

func (cph *checkPackageHandle) cached(ctx context.Context) (*pkg, error) {
	v := cph.handle.Cached()
	if v == nil {
		return nil, errors.Errorf("no cached type information for %s", cph.m.pkgPath)
	}
	data := v.(*checkPackageData)
	return data.pkg, data.err
}

func (imp *importer) parseGoHandles(ctx context.Context, m *metadata) ([]source.ParseGoHandle, error) {
	phs := make([]source.ParseGoHandle, 0, len(m.files))
	for _, uri := range m.files {
		f, err := imp.view.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		fh := f.Handle(ctx)
		mode := source.ParseExported
		if imp.topLevelPackageID == m.id {
			mode = source.ParseFull
		}
		phs = append(phs, imp.view.session.cache.ParseGoHandle(fh, mode))
	}
	return phs, nil
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	ctx, done := trace.StartSpan(imp.ctx, "cache.importer.Import", telemetry.PackagePath.Of(pkgPath))
	defer done()

	// We need to set the parent package's imports, so there should always be one.
	if imp.parentPkg == nil {
		return nil, errors.Errorf("no parent package for import %s", pkgPath)
	}

	// Get the CheckPackageHandle from the importing package.
	cph, ok := imp.parentCheckPackageHandle.imports[packagePath(pkgPath)]
	if !ok {
		return nil, errors.Errorf("no package data for import path %s", pkgPath)
	}
	for _, ph := range cph.Files() {
		if ph.Mode() != source.ParseExported {
			panic("dependency parsed in full mode")
		}
	}
	pkg, err := cph.check(ctx)
	if err != nil {
		return nil, err
	}
	imp.parentPkg.imports[packagePath(pkgPath)] = pkg

	// Add every file in this package to our cache.
	if err := imp.cachePackage(ctx, cph); err != nil {
		return nil, err
	}
	return pkg.GetTypes(), nil
}

func (imp *importer) typeCheck(ctx context.Context, cph *checkPackageHandle, m *metadata) (*pkg, error) {
	ctx, done := trace.StartSpan(ctx, "cache.importer.typeCheck", telemetry.Package.Of(m.id))
	defer done()

	pkg := &pkg{
		view:       imp.view,
		id:         m.id,
		pkgPath:    m.pkgPath,
		files:      cph.Files(),
		imports:    make(map[packagePath]*pkg),
		typesSizes: m.typesSizes,
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
	// If the package comes back with errors from `go list`,
	// don't bother type-checking it.
	for _, err := range m.errors {
		pkg.errors = append(m.errors, err)
	}
	// Set imports of package to correspond to cached packages.
	cimp := imp.child(ctx, pkg, cph)
	for _, child := range m.children {
		childHandle, err := cimp.checkPackageHandle(ctx, child)
		if err != nil {
			log.Error(ctx, "no check package handle", err, telemetry.Package.Of(child.id))
			continue
		}
		cph.imports[child.pkgPath] = childHandle
	}
	var (
		files       = make([]*ast.File, len(pkg.files))
		parseErrors = make([]error, len(pkg.files))
		wg          sync.WaitGroup
	)
	for i, ph := range pkg.files {
		wg.Add(1)
		go func(i int, ph source.ParseGoHandle) {
			defer wg.Done()

			files[i], _, parseErrors[i], _ = ph.Parse(ctx)
		}(i, ph)
	}
	wg.Wait()

	for _, err := range parseErrors {
		if err != nil {
			imp.view.session.cache.appendPkgError(pkg, err)
		}
	}

	var i int
	for _, f := range files {
		if f != nil {
			files[i] = f
			i++
		}
	}
	files = files[:i]

	// Use the default type information for the unsafe package.
	if m.pkgPath == "unsafe" {
		pkg.types = types.Unsafe
	} else if len(files) == 0 { // not the unsafe package, no parsed files
		return nil, errors.Errorf("no parsed files for package %s", pkg.pkgPath)
	} else {
		pkg.types = types.NewPackage(string(m.pkgPath), m.name)
	}

	cfg := &types.Config{
		Error: func(err error) {
			imp.view.session.cache.appendPkgError(pkg, err)
		},
		Importer: cimp,
	}
	check := types.NewChecker(cfg, imp.view.session.cache.FileSet(), pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files)

	return pkg, nil
}

func (imp *importer) child(ctx context.Context, pkg *pkg, cph *checkPackageHandle) *importer {
	// Handle circular imports by copying previously seen imports.
	seen := make(map[packageID]struct{})
	for k, v := range imp.seen {
		seen[k] = v
	}
	seen[pkg.id] = struct{}{}
	return &importer{
		view:                     imp.view,
		ctx:                      ctx,
		config:                   imp.config,
		seen:                     seen,
		topLevelPackageID:        imp.topLevelPackageID,
		parentPkg:                pkg,
		parentCheckPackageHandle: cph,
	}
}

func (imp *importer) cachePackage(ctx context.Context, cph *checkPackageHandle) error {
	for _, ph := range cph.files {
		uri := ph.File().Identity().URI
		f, err := imp.view.GetFile(ctx, uri)
		if err != nil {
			return errors.Errorf("no such file %s: %v", uri, err)
		}
		gof, ok := f.(*goFile)
		if !ok {
			return errors.Errorf("%s is not a Go file", uri)
		}
		if err := imp.cachePerFile(ctx, gof, ph, cph); err != nil {
			return errors.Errorf("failed to cache file %s: %v", gof.URI(), err)
		}
	}
	return nil
}

func (imp *importer) cachePerFile(ctx context.Context, gof *goFile, ph source.ParseGoHandle, cph *checkPackageHandle) error {
	gof.mu.Lock()
	defer gof.mu.Unlock()

	// Set the package even if we failed to parse the file.
	if gof.cphs == nil {
		gof.cphs = make(map[packageID]source.CheckPackageHandle)
	}
	gof.cphs[cph.m.id] = cph

	file, _, _, err := ph.Parse(ctx)
	if err != nil {
		return err
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
