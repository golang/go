// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type packageHandleKey string

// packageHandle implements source.PackageHandle.
type packageHandle struct {
	handle *memoize.Handle

	goFiles []source.ParseGoHandle

	// compiledGoFiles are the ParseGoHandles that compose the package.
	compiledGoFiles []source.ParseGoHandle

	// mode is the mode the the files were parsed in.
	mode source.ParseMode

	// m is the metadata associated with the package.
	m *metadata

	// key is the hashed key for the package.
	key packageHandleKey
}

func (ph *packageHandle) packageKey() packageKey {
	return packageKey{
		id:   ph.m.id,
		mode: ph.mode,
	}
}

// packageData contains the data produced by type-checking a package.
type packageData struct {
	memoize.NoCopy

	pkg *pkg
	err error
}

// buildPackageHandle returns a source.PackageHandle for a given package and config.
func (s *snapshot) buildPackageHandle(ctx context.Context, id packageID, mode source.ParseMode) (*packageHandle, error) {
	if ph := s.getPackage(id, mode); ph != nil {
		return ph, nil
	}

	// Build the PackageHandle for this ID and its dependencies.
	ph, deps, err := s.buildKey(ctx, id, mode)
	if err != nil {
		return nil, err
	}

	// Do not close over the packageHandle or the snapshot in the Bind function.
	// This creates a cycle, which causes the finalizers to never run on the handles.
	// The possible cycles are:
	//
	//     packageHandle.h.function -> packageHandle
	//     packageHandle.h.function -> snapshot -> packageHandle
	//

	m := ph.m
	goFiles := ph.goFiles
	compiledGoFiles := ph.compiledGoFiles
	key := ph.key
	fset := s.view.session.cache.fset

	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		// Begin loading the direct dependencies, in parallel.
		for _, dep := range deps {
			go func(dep *packageHandle) {
				dep.check(ctx)
			}(dep)
		}
		data := &packageData{}
		data.pkg, data.err = typeCheck(ctx, fset, m, mode, goFiles, compiledGoFiles, deps)
		return data
	})
	ph.handle = h

	// Cache the PackageHandle in the snapshot.
	s.addPackage(ph)

	return ph, nil
}

// buildKey computes the key for a given packageHandle.
func (s *snapshot) buildKey(ctx context.Context, id packageID, mode source.ParseMode) (*packageHandle, map[packagePath]*packageHandle, error) {
	m := s.getMetadata(id)
	if m == nil {
		return nil, nil, errors.Errorf("no metadata for %s", id)
	}
	goFiles, err := s.parseGoHandles(ctx, m.goFiles, mode)
	if err != nil {
		return nil, nil, err
	}
	compiledGoFiles, err := s.parseGoHandles(ctx, m.compiledGoFiles, mode)
	if err != nil {
		return nil, nil, err
	}
	ph := &packageHandle{
		m:               m,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		mode:            mode,
	}
	// Make sure all of the depList are sorted.
	depList := append([]packageID{}, m.deps...)
	sort.Slice(depList, func(i, j int) bool {
		return depList[i] < depList[j]
	})

	deps := make(map[packagePath]*packageHandle)

	// Begin computing the key by getting the depKeys for all dependencies.
	var depKeys []packageHandleKey
	for _, depID := range depList {
		mode := source.ParseExported
		if _, ok := s.isWorkspacePackage(depID); ok {
			mode = source.ParseFull
		}
		depHandle, err := s.buildPackageHandle(ctx, depID, mode)
		if err != nil {
			log.Error(ctx, "no dep handle", err, telemetry.Package.Of(depID))

			// One bad dependency should not prevent us from checking the entire package.
			// Add a special key to mark a bad dependency.
			depKeys = append(depKeys, packageHandleKey(fmt.Sprintf("%s import not found", id)))
			continue
		}
		deps[depHandle.m.pkgPath] = depHandle
		depKeys = append(depKeys, depHandle.key)
	}
	ph.key = checkPackageKey(ph.m.id, ph.compiledGoFiles, m.config, depKeys)
	return ph, deps, nil
}

func checkPackageKey(id packageID, pghs []source.ParseGoHandle, cfg *packages.Config, deps []packageHandleKey) packageHandleKey {
	var depBytes []byte
	for _, dep := range deps {
		depBytes = append(depBytes, []byte(dep)...)
	}
	return packageHandleKey(hashContents([]byte(fmt.Sprintf("%s%s%s%s", id, hashParseKeys(pghs), hashConfig(cfg), hashContents(depBytes)))))
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

func (ph *packageHandle) Check(ctx context.Context) (source.Package, error) {
	return ph.check(ctx)
}

func (ph *packageHandle) check(ctx context.Context) (*pkg, error) {
	v := ph.handle.Get(ctx)
	if v == nil {
		return nil, ctx.Err()
	}
	data := v.(*packageData)
	return data.pkg, data.err
}

func (ph *packageHandle) CompiledGoFiles() []source.ParseGoHandle {
	return ph.compiledGoFiles
}

func (ph *packageHandle) ID() string {
	return string(ph.m.id)
}

func (ph *packageHandle) MissingDependencies() []string {
	var md []string
	for i := range ph.m.missingDeps {
		md = append(md, string(i))
	}
	return md
}

func hashImports(ctx context.Context, wsPackages []source.PackageHandle) (string, error) {
	results := make(map[string]bool)
	var imports []string
	for _, ph := range wsPackages {
		// Check package since we do not always invalidate the metadata.
		pkg, err := ph.Check(ctx)
		if err != nil {
			return "", err
		}
		for _, path := range pkg.Imports() {
			imp := path.PkgPath()
			if _, ok := results[imp]; !ok {
				results[imp] = true
				imports = append(imports, imp)
			}
		}
	}
	sort.Strings(imports)
	hashed := strings.Join(imports, ",")
	return hashContents([]byte(hashed)), nil
}

func (ph *packageHandle) Cached() (source.Package, error) {
	return ph.cached()
}

func (ph *packageHandle) cached() (*pkg, error) {
	v := ph.handle.Cached()
	if v == nil {
		return nil, errors.Errorf("no cached type information for %s", ph.m.pkgPath)
	}
	data := v.(*packageData)
	return data.pkg, data.err
}

func (s *snapshot) parseGoHandles(ctx context.Context, files []span.URI, mode source.ParseMode) ([]source.ParseGoHandle, error) {
	phs := make([]source.ParseGoHandle, 0, len(files))
	for _, uri := range files {
		fh, err := s.GetFile(uri)
		if err != nil {
			return nil, err
		}
		phs = append(phs, s.view.session.cache.ParseGoHandle(fh, mode))
	}
	return phs, nil
}

func typeCheck(ctx context.Context, fset *token.FileSet, m *metadata, mode source.ParseMode, goFiles []source.ParseGoHandle, compiledGoFiles []source.ParseGoHandle, deps map[packagePath]*packageHandle) (*pkg, error) {
	ctx, done := trace.StartSpan(ctx, "cache.importer.typeCheck", telemetry.Package.Of(m.id))
	defer done()

	var rawErrors []error
	for _, err := range m.errors {
		rawErrors = append(rawErrors, err)
	}

	pkg := &pkg{
		id:              m.id,
		pkgPath:         m.pkgPath,
		mode:            mode,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		imports:         make(map[packagePath]*pkg),
		typesSizes:      m.typesSizes,
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		forTest: m.forTest,
	}
	var (
		files        = make([]*ast.File, len(pkg.compiledGoFiles))
		parseErrors  = make([]error, len(pkg.compiledGoFiles))
		actualErrors = make([]error, len(pkg.compiledGoFiles))
		wg           sync.WaitGroup
	)
	for i, ph := range pkg.compiledGoFiles {
		wg.Add(1)
		go func(i int, ph source.ParseGoHandle) {
			files[i], _, _, parseErrors[i], actualErrors[i] = ph.Parse(ctx)
			wg.Done()
		}(i, ph)
	}
	for _, ph := range pkg.goFiles {
		wg.Add(1)
		// We need to parse the non-compiled go files, but we don't care about their errors.
		go func(ph source.ParseGoHandle) {
			ph.Parse(ctx)
			wg.Done()
		}(ph)
	}
	wg.Wait()

	for _, e := range parseErrors {
		if e != nil {
			rawErrors = append(rawErrors, e)
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
	if pkg.pkgPath == "unsafe" {
		pkg.types = types.Unsafe
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		return pkg, nil
	} else if len(files) == 0 { // not the unsafe package, no parsed files
		return nil, errors.Errorf("no parsed files for package %s, expected: %s, errors: %v, list errors: %v", pkg.pkgPath, pkg.compiledGoFiles, actualErrors, rawErrors)
	} else {
		pkg.types = types.NewPackage(string(m.pkgPath), m.name)
	}

	cfg := &types.Config{
		Error: func(e error) {
			rawErrors = append(rawErrors, e)
		},
		Importer: importerFunc(func(pkgPath string) (*types.Package, error) {
			// If the context was cancelled, we should abort.
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			dep := deps[packagePath(pkgPath)]
			if dep == nil {
				// We may be in GOPATH mode, in which case we need to check vendor dirs.
				searchDir := path.Dir(pkg.PkgPath())
				for {
					vdir := packagePath(path.Join(searchDir, "vendor", pkgPath))
					if vdep := deps[vdir]; vdep != nil {
						dep = vdep
						break
					}

					// Search until Dir doesn't take us anywhere new, e.g. "." or "/".
					next := path.Dir(searchDir)
					if searchDir == next {
						break
					}
					searchDir = next
				}
			}
			if dep == nil {
				return nil, errors.Errorf("no package for import %s", pkgPath)
			}
			depPkg, err := dep.check(ctx)
			if err != nil {
				return nil, err
			}
			pkg.imports[depPkg.pkgPath] = depPkg
			return depPkg.types, nil
		}),
	}
	check := types.NewChecker(cfg, fset, pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files)
	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// We don't care about a package's errors unless we have parsed it in full.
	if mode == source.ParseFull {
		for _, e := range rawErrors {
			srcErr, err := sourceError(ctx, fset, pkg, e)
			if err != nil {
				log.Error(ctx, "unable to compute error positions", err, telemetry.Package.Of(pkg.ID()))
				continue
			}
			pkg.errors = append(pkg.errors, srcErr)
		}
	}
	return pkg, nil
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
