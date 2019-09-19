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
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// pkg contains the type information needed by the source package.
type pkg struct {
	view *view

	// ID and package path have their own types to avoid being used interchangeably.
	id      packageID
	pkgPath packagePath

	files      []source.ParseGoHandle
	errors     []packages.Error
	imports    map[packagePath]*pkg
	types      *types.Package
	typesInfo  *types.Info
	typesSizes types.Sizes

	// The analysis cache holds analysis information for all the packages in a view.
	// Each graph node (action) is one unit of analysis.
	// Edges express package-to-package (vertical) dependencies,
	// and analysis-to-analysis (horizontal) dependencies.
	mu       sync.Mutex
	analyses map[*analysis.Analyzer]*analysisEntry

	diagMu      sync.Mutex
	diagnostics map[*analysis.Analyzer][]source.Diagnostic
}

// Declare explicit types for package paths and IDs to ensure that we never use
// an ID where a path belongs, and vice versa. If we confused the two, it would
// result in confusing errors because package IDs often look like package paths.
type packageID string
type packagePath string

type analysisEntry struct {
	done      chan struct{}
	succeeded bool
	*source.Action
}

func (pkg *pkg) GetActionGraph(ctx context.Context, a *analysis.Analyzer) (*source.Action, error) {
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
				importPaths = append(importPaths, string(importPath))
			}
			sort.Strings(importPaths) // for determinism
			for _, importPath := range importPaths {
				dep, err := pkg.GetImport(ctx, importPath)
				if err != nil {
					return nil, err
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

func (pkg *pkg) ID() string {
	return string(pkg.id)
}

func (pkg *pkg) PkgPath() string {
	return string(pkg.pkgPath)
}

func (pkg *pkg) Files() []source.ParseGoHandle {
	return pkg.files
}

func (pkg *pkg) File(uri span.URI) (source.ParseGoHandle, error) {
	for _, ph := range pkg.Files() {
		if ph.File().Identity().URI == uri {
			return ph, nil
		}
	}
	return nil, errors.Errorf("no ParseGoHandle for %s", uri)
}

func (pkg *pkg) GetSyntax(ctx context.Context) []*ast.File {
	var syntax []*ast.File
	for _, ph := range pkg.files {
		file, _, _, err := ph.Cached(ctx)
		if err == nil {
			syntax = append(syntax, file)
		}
	}
	return syntax
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
	return pkg.types == nil || pkg.typesInfo == nil || pkg.typesSizes == nil
}

func (pkg *pkg) SetDiagnostics(a *analysis.Analyzer, diags []source.Diagnostic) {
	pkg.diagMu.Lock()
	defer pkg.diagMu.Unlock()
	if pkg.diagnostics == nil {
		pkg.diagnostics = make(map[*analysis.Analyzer][]source.Diagnostic)
	}
	pkg.diagnostics[a] = diags
}

func (pkg *pkg) GetDiagnostics() []source.Diagnostic {
	pkg.diagMu.Lock()
	defer pkg.diagMu.Unlock()

	var diags []source.Diagnostic
	for _, d := range pkg.diagnostics {
		diags = append(diags, d...)
	}
	return diags
}

func (p *pkg) FindFile(ctx context.Context, uri span.URI) (source.ParseGoHandle, source.Package, error) {
	// Special case for ignored files.
	if p.view.Ignore(uri) {
		return p.view.findIgnoredFile(ctx, uri)
	}

	queue := []*pkg{p}
	seen := make(map[string]bool)

	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		seen[pkg.ID()] = true

		for _, ph := range pkg.files {
			if ph.File().Identity().URI == uri {
				return ph, pkg, nil
			}
		}
		for _, dep := range pkg.imports {
			if !seen[dep.ID()] {
				queue = append(queue, dep)
			}
		}
	}
	return nil, nil, errors.Errorf("no file for %s", uri)
}
