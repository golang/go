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
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/typesinternal"
	errors "golang.org/x/xerrors"
)

type packageHandleKey string

// packageHandle implements source.PackageHandle.
type packageHandle struct {
	handle *memoize.Handle

	goFiles, compiledGoFiles []*parseGoHandle

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

	// Cache the PackageHandle in the snapshot. If a package handle has already
	// been cached, addPackage will return the cached value. This is fine,
	// since the original package handle above will have no references and be
	// garbage collected.
	ph = s.addPackageHandle(ph)

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
			event.Error(ctx, "no dep handle", err, tag.Package.Of(string(depID)))
			if ctx.Err() != nil {
				return nil, nil, ctx.Err()
			}
			// One bad dependency should not prevent us from checking the entire package.
			// Add a special key to mark a bad dependency.
			depKeys = append(depKeys, packageHandleKey(fmt.Sprintf("%s import not found", id)))
			continue
		}
		deps[depHandle.m.pkgPath] = depHandle
		depKeys = append(depKeys, depHandle.key)
	}
	ph.key = checkPackageKey(ctx, ph.m.id, ph.compiledGoFiles, m.config, depKeys)
	return ph, deps, nil
}

func checkPackageKey(ctx context.Context, id packageID, pghs []*parseGoHandle, cfg *packages.Config, deps []packageHandleKey) packageHandleKey {
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
	b.WriteString(string(rune(config.Mode)))

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
	v, err := ph.handle.Get(ctx)
	if err != nil {
		return nil, err
	}
	data := v.(*packageData)
	return data.pkg, data.err
}

func (ph *packageHandle) ID() string {
	return string(ph.m.id)
}

func (ph *packageHandle) CompiledGoFiles() []source.ParseGoHandle {
	var files []source.ParseGoHandle
	for _, f := range ph.compiledGoFiles {
		files = append(files, f)
	}
	return files
}

func (ph *packageHandle) MissingDependencies() []string {
	var md []string
	for i := range ph.m.missingDeps {
		md = append(md, string(i))
	}
	return md
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

func (s *snapshot) parseGoHandles(ctx context.Context, files []span.URI, mode source.ParseMode) ([]*parseGoHandle, error) {
	pghs := make([]*parseGoHandle, 0, len(files))
	for _, uri := range files {
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		pghs = append(pghs, s.view.session.cache.parseGoHandle(ctx, fh, mode))
	}
	return pghs, nil
}

func typeCheck(ctx context.Context, fset *token.FileSet, m *metadata, mode source.ParseMode, goFiles, compiledGoFiles []*parseGoHandle, deps map[packagePath]*packageHandle) (*pkg, error) {
	ctx, done := event.Start(ctx, "cache.importer.typeCheck", tag.Package.Of(string(m.id)))
	defer done()

	var rawErrors []error
	for _, err := range m.errors {
		rawErrors = append(rawErrors, err)
	}

	pkg := &pkg{
		id:              m.id,
		name:            m.name,
		pkgPath:         m.pkgPath,
		mode:            mode,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		module:          m.module,
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

		mu             sync.Mutex
		skipTypeErrors bool
	)
	for i, ph := range pkg.compiledGoFiles {
		wg.Add(1)
		go func(i int, ph *parseGoHandle) {
			defer wg.Done()
			data, err := ph.parse(ctx)
			if err != nil {
				actualErrors[i] = err
				return
			}
			files[i], parseErrors[i], actualErrors[i] = data.ast, data.parseError, data.err

			mu.Lock()
			skipTypeErrors = skipTypeErrors || data.fixed
			mu.Unlock()
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
		pkg.types = types.NewPackage(string(m.pkgPath), string(m.name))
	}

	cfg := &types.Config{
		Error: func(e error) {
			// If we have fixed parse errors in any of the files,
			// we should hide type errors, as they may be completely nonsensical.
			if skipTypeErrors {
				return
			}
			rawErrors = append(rawErrors, e)
		},
		Importer: importerFunc(func(pkgPath string) (*types.Package, error) {
			// If the context was cancelled, we should abort.
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			dep := resolveImportPath(pkgPath, pkg, deps)
			if dep == nil {
				return nil, errors.Errorf("no package for import %s", pkgPath)
			}
			if !isValidImport(m.pkgPath, dep.m.pkgPath) {
				return nil, errors.Errorf("invalid use of internal package %s", pkgPath)
			}
			depPkg, err := dep.check(ctx)
			if err != nil {
				return nil, err
			}
			pkg.imports[depPkg.pkgPath] = depPkg
			return depPkg.types, nil
		}),
	}
	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	typesinternal.SetUsesCgo(cfg)

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
				event.Error(ctx, "unable to compute error positions", err, tag.Package.Of(pkg.ID()))
				continue
			}
			pkg.errors = append(pkg.errors, srcErr)
			if err, ok := e.(types.Error); ok {
				pkg.typeErrors = append(pkg.typeErrors, err)
			}
		}
	}

	return pkg, nil
}

// resolveImportPath resolves an import path in pkg to a package from deps.
// It should produce the same results as resolveImportPath:
// https://cs.opensource.google/go/go/+/master:src/cmd/go/internal/load/pkg.go;drc=641918ee09cb44d282a30ee8b66f99a0b63eaef9;l=990.
func resolveImportPath(importPath string, pkg *pkg, deps map[packagePath]*packageHandle) *packageHandle {
	if dep := deps[packagePath(importPath)]; dep != nil {
		return dep
	}
	// We may be in GOPATH mode, in which case we need to check vendor dirs.
	searchDir := path.Dir(pkg.PkgPath())
	for {
		vdir := packagePath(path.Join(searchDir, "vendor", importPath))
		if vdep := deps[vdir]; vdep != nil {
			return vdep
		}

		// Search until Dir doesn't take us anywhere new, e.g. "." or "/".
		next := path.Dir(searchDir)
		if searchDir == next {
			break
		}
		searchDir = next
	}

	// Vendor didn't work. Let's try minimal module compatibility mode.
	// In MMC, the packagePath is the canonical (.../vN/...) path, which
	// is hard to calculate. But the go command has already resolved the ID
	// to the non-versioned path, and we can take advantage of that.
	for _, dep := range deps {
		if dep.ID() == importPath {
			return dep
		}
	}
	return nil
}

func isValidImport(pkgPath, importPkgPath packagePath) bool {
	i := strings.LastIndex(string(importPkgPath), "/internal/")
	if i == -1 {
		return true
	}
	if pkgPath == "command-line-arguments" {
		return true
	}
	return strings.HasPrefix(string(pkgPath), string(importPkgPath[:i]))
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
