// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/types"
	"sort"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

// pkg contains the type information needed by the source package.
type pkg struct {
	id, pkgPath string
	files       []string
	syntax      []*ast.File
	errors      []packages.Error
	imports     map[string]*pkg
	types       *types.Package
	typesInfo   *types.Info
	typesSizes  types.Sizes

	// The analysis cache holds analysis information for all the packages in a view.
	// Each graph node (action) is one unit of analysis.
	// Edges express package-to-package (vertical) dependencies,
	// and analysis-to-analysis (horizontal) dependencies.
	mu       sync.Mutex
	analyses map[*analysis.Analyzer]*analysisEntry
}

type analysisEntry struct {
	done      chan struct{}
	succeeded bool
	*source.Action
}

func (pkg *pkg) GetActionGraph(ctx context.Context, a *analysis.Analyzer) (*source.Action, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	pkg.mu.Lock()
	e, ok := pkg.analyses[a]
	if ok {
		// cache hit
		pkg.mu.Unlock()

		// wait for entry to become ready or the context to be cancelled
		select {
		case <-e.done:
			// If the goroutine we are waiting on was cancelled, we should retry.
			// If errors other than cancelation/timeout become possible, it may
			// no longer be appropriate to always retry here.
			if !e.succeeded {
				return pkg.GetActionGraph(ctx, a)
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	} else {
		// cache miss
		e = &analysisEntry{
			done: make(chan struct{}),
			Action: &source.Action{
				Analyzer: a,
				Pkg:      pkg,
			},
		}
		pkg.analyses[a] = e
		pkg.mu.Unlock()

		defer func() {
			// If we got an error, clear out our defunct cache entry. We don't cache
			// errors since they could depend on our dependencies, which can change.
			// Currently the only possible error is context.Canceled, though, which
			// should also not be cached.
			if !e.succeeded {
				pkg.mu.Lock()
				delete(pkg.analyses, a)
				pkg.mu.Unlock()
			}

			// Always close done so waiters don't get stuck.
			close(e.done)
		}()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.

		// Add a dependency on each required analyzers.
		for _, req := range a.Requires {
			act, err := pkg.GetActionGraph(ctx, req)
			if err != nil {
				return nil, err
			}
			e.Deps = append(e.Deps, act)
		}

		// An analysis that consumes/produces facts
		// must run on the package's dependencies too.
		if len(a.FactTypes) > 0 {
			importPaths := make([]string, 0, len(pkg.imports))
			for importPath := range pkg.imports {
				importPaths = append(importPaths, importPath)
			}
			sort.Strings(importPaths) // for determinism
			for _, importPath := range importPaths {
				dep, ok := pkg.imports[importPath]
				if !ok {
					continue
				}
				act, err := dep.GetActionGraph(ctx, a)
				if err != nil {
					return nil, err
				}
				e.Deps = append(e.Deps, act)
			}
		}
		e.succeeded = true
	}
	return e.Action, nil
}

func (pkg *pkg) PkgPath() string {
	return pkg.pkgPath
}

func (pkg *pkg) GetFilenames() []string {
	return pkg.files
}

func (pkg *pkg) GetSyntax() []*ast.File {
	return pkg.syntax
}

func (pkg *pkg) GetErrors() []packages.Error {
	return pkg.errors
}

func (pkg *pkg) GetTypes() *types.Package {
	return pkg.types
}

func (pkg *pkg) GetTypesInfo() *types.Info {
	return pkg.typesInfo
}

func (pkg *pkg) GetTypesSizes() types.Sizes {
	return pkg.typesSizes
}

func (pkg *pkg) IsIllTyped() bool {
	return pkg.types == nil && pkg.typesInfo == nil
}

func (pkg *pkg) GetImport(pkgPath string) source.Package {
	if imp := pkg.imports[pkgPath]; imp != nil {
		return imp
	}
	// Don't return a nil pointer because that still satisfies the interface.
	return nil
}
