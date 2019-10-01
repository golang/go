// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/scanner"
	"go/types"
	"sort"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type importer struct {
	snapshot *snapshot
	ctx      context.Context

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

// checkPackageHandle implements source.CheckPackageHandle.
type checkPackageHandle struct {
	handle *memoize.Handle

	// files are the ParseGoHandles that compose the package.
	files []source.ParseGoHandle

	// mode is the mode the the files were parsed in.
	mode source.ParseMode

	// imports is the map of the package's imports.
	imports map[packagePath]packageID

	// m is the metadata associated with the package.
	m *metadata

	// key is the hashed key for the package.
	key []byte
}

func (cph *checkPackageHandle) packageKey() packageKey {
	return packageKey{
		id:   cph.m.id,
		mode: cph.mode,
	}
}

// checkPackageData contains the data produced by type-checking a package.
type checkPackageData struct {
	memoize.NoCopy

	pkg *pkg
	err error
}

// checkPackageHandle returns a source.CheckPackageHandle for a given package and config.
func (imp *importer) checkPackageHandle(ctx context.Context, id packageID) (*checkPackageHandle, error) {
	// Determine the mode that the files should be parsed in.
	mode := imp.mode(id)

	// Check if we already have this CheckPackageHandle cached.
	if cph := imp.snapshot.getPackage(id, mode); cph != nil {
		return cph, nil
	}

	// Build the CheckPackageHandle for this ID and its dependencies.
	cph, err := imp.buildKey(ctx, id, mode)
	if err != nil {
		return nil, err
	}

	h := imp.snapshot.view.session.cache.store.Bind(string(cph.key), func(ctx context.Context) interface{} {
		data := &checkPackageData{}
		data.pkg, data.err = imp.typeCheck(ctx, cph)
		return data
	})
	cph.handle = h

	return cph, nil
}

// buildKey computes the checkPackageKey for a given checkPackageHandle.
func (imp *importer) buildKey(ctx context.Context, id packageID, mode source.ParseMode) (*checkPackageHandle, error) {
	m := imp.snapshot.getMetadata(id)
	if m == nil {
		return nil, errors.Errorf("no metadata for %s", id)
	}

	phs, err := imp.parseGoHandles(ctx, m, mode)
	if err != nil {
		return nil, err
	}
	cph := &checkPackageHandle{
		m:       m,
		files:   phs,
		imports: make(map[packagePath]packageID),
		mode:    mode,
	}

	// Make sure all of the deps are sorted.
	deps := append([]packageID{}, m.deps...)
	sort.Slice(deps, func(i, j int) bool {
		return deps[i] < deps[j]
	})

	// Create the dep importer for use on the dependency handles.
	depImporter := &importer{
		snapshot:          imp.snapshot,
		topLevelPackageID: imp.topLevelPackageID,
	}
	// Begin computing the key by getting the depKeys for all dependencies.
	var depKeys [][]byte
	for _, dep := range deps {
		depHandle, err := depImporter.checkPackageHandle(ctx, dep)
		if err != nil {
			return nil, errors.Errorf("no dep handle for %s: %+v", dep, err)
		}
		cph.imports[depHandle.m.pkgPath] = depHandle.m.id
		depKeys = append(depKeys, depHandle.key)
	}
	cph.key = checkPackageKey(cph.m.id, cph.files, m.config, depKeys)

	// Cache the CheckPackageHandle in the snapshot.
	imp.snapshot.addPackage(cph)

	return cph, nil
}

func checkPackageKey(id packageID, phs []source.ParseGoHandle, cfg *packages.Config, deps [][]byte) []byte {
	return []byte(hashContents([]byte(fmt.Sprintf("%s%s%s%s", id, hashParseKeys(phs), hashConfig(cfg), hashContents(bytes.Join(deps, nil))))))
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
		return nil, errors.Errorf("no package for %s", cph.m.id)
	}
	data := v.(*checkPackageData)
	return data.pkg, data.err
}

func (cph *checkPackageHandle) Files() []source.ParseGoHandle {
	return cph.files
}

func (cph *checkPackageHandle) ID() string {
	return string(cph.m.id)
}

func (cph *checkPackageHandle) MissingDependencies() []string {
	var md []string
	for i := range cph.m.missingDeps {
		md = append(md, string(i))
	}
	return md
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

func (imp *importer) parseGoHandles(ctx context.Context, m *metadata, mode source.ParseMode) ([]source.ParseGoHandle, error) {
	phs := make([]source.ParseGoHandle, 0, len(m.files))
	for _, uri := range m.files {
		f, err := imp.snapshot.view.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		fh := imp.snapshot.Handle(ctx, f)
		phs = append(phs, imp.snapshot.view.session.cache.ParseGoHandle(fh, mode))
	}
	return phs, nil
}

func (imp *importer) mode(id packageID) source.ParseMode {
	if imp.topLevelPackageID == id {
		return source.ParseFull
	}
	return source.ParseExported
}

func (imp *importer) Import(pkgPath string) (*types.Package, error) {
	ctx, done := trace.StartSpan(imp.ctx, "cache.importer.Import", telemetry.PackagePath.Of(pkgPath))
	defer done()

	// We need to set the parent package's imports, so there should always be one.
	if imp.parentPkg == nil {
		return nil, errors.Errorf("no parent package for import %s", pkgPath)
	}
	// Get the CheckPackageHandle from the importing package.
	id, ok := imp.parentCheckPackageHandle.imports[packagePath(pkgPath)]
	if !ok {
		return nil, errors.Errorf("no package data for import path %s", pkgPath)
	}
	cph := imp.snapshot.getPackage(id, source.ParseExported)
	if cph == nil {
		return nil, errors.Errorf("no package for %s", id)
	}
	pkg, err := cph.check(ctx)
	if err != nil {
		return nil, err
	}
	imp.parentPkg.imports[packagePath(pkgPath)] = pkg
	return pkg.GetTypes(), nil
}

func (imp *importer) typeCheck(ctx context.Context, cph *checkPackageHandle) (*pkg, error) {
	ctx, done := trace.StartSpan(ctx, "cache.importer.typeCheck", telemetry.Package.Of(cph.m.id))
	defer done()

	pkg := &pkg{
		view:       imp.snapshot.view,
		id:         cph.m.id,
		pkgPath:    cph.m.pkgPath,
		files:      cph.Files(),
		imports:    make(map[packagePath]*pkg),
		typesSizes: cph.m.typesSizes,
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
	for _, err := range cph.m.errors {
		pkg.errors = append(cph.m.errors, err)
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
			imp.snapshot.view.session.cache.appendPkgError(pkg, err)
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
	if cph.m.pkgPath == "unsafe" {
		pkg.types = types.Unsafe
	} else if len(files) == 0 { // not the unsafe package, no parsed files
		return nil, errors.Errorf("no parsed files for package %s", pkg.pkgPath)
	} else {
		pkg.types = types.NewPackage(string(cph.m.pkgPath), cph.m.name)
	}

	cfg := &types.Config{
		Error: func(err error) {
			imp.snapshot.view.session.cache.appendPkgError(pkg, err)
		},
		Importer: imp.depImporter(ctx, cph, pkg),
	}
	check := types.NewChecker(cfg, imp.snapshot.view.session.cache.FileSet(), pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files)

	return pkg, nil
}

func (imp *importer) depImporter(ctx context.Context, cph *checkPackageHandle, pkg *pkg) *importer {
	// Handle circular imports by copying previously seen imports.
	seen := make(map[packageID]struct{})
	for k, v := range imp.seen {
		seen[k] = v
	}
	seen[pkg.id] = struct{}{}
	return &importer{
		snapshot:                 imp.snapshot,
		topLevelPackageID:        imp.topLevelPackageID,
		parentPkg:                pkg,
		parentCheckPackageHandle: cph,
		seen:                     seen,
		ctx:                      ctx,
	}
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
