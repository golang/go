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
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// pkg contains the type information needed by the source package.
type pkg struct {
	view *view

	// ID and package path have their own types to avoid being used interchangeably.
	id         packageID
	pkgPath    packagePath
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

func (p *pkg) GetActionGraph(ctx context.Context, a *analysis.Analyzer) (*source.Action, error) {
	p.mu.Lock()
	e, ok := p.analyses[a]
	if ok {
		// cache hit
		p.mu.Unlock()

		// wait for entry to become ready or the context to be cancelled
		select {
		case <-e.done:
			// If the goroutine we are waiting on was cancelled, we should retry.
			// If errors other than cancelation/timeout become possible, it may
			// no longer be appropriate to always retry here.
			if !e.succeeded {
				return p.GetActionGraph(ctx, a)
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
				Pkg:      p,
			},
		}
		p.analyses[a] = e
		p.mu.Unlock()

		defer func() {
			// If we got an error, clear out our defunct cache entry. We don't cache
			// errors since they could depend on our dependencies, which can change.
			// Currently the only possible error is context.Canceled, though, which
			// should also not be cached.
			if !e.succeeded {
				p.mu.Lock()
				delete(p.analyses, a)
				p.mu.Unlock()
			}

			// Always close done so waiters don't get stuck.
			close(e.done)
		}()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.

		// Add a dependency on each required analyzers.
		for _, req := range a.Requires {
			act, err := p.GetActionGraph(ctx, req)
			if err != nil {
				return nil, err
			}
			e.Deps = append(e.Deps, act)
		}

		// An analysis that consumes/produces facts
		// must run on the package's dependencies too.
		if len(a.FactTypes) > 0 {
			importPaths := make([]string, 0, len(p.imports))
			for importPath := range p.imports {
				importPaths = append(importPaths, string(importPath))
			}
			sort.Strings(importPaths) // for determinism
			for _, importPath := range importPaths {
				dep, err := p.GetImport(ctx, importPath)
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

func (p *pkg) ID() string {
	return string(p.id)
}

func (p *pkg) PkgPath() string {
	return string(p.pkgPath)
}

func (p *pkg) Files() []source.ParseGoHandle {
	return p.files
}

func (p *pkg) File(uri span.URI) (source.ParseGoHandle, error) {
	for _, ph := range p.Files() {
		if ph.File().Identity().URI == uri {
			return ph, nil
		}
	}
	return nil, errors.Errorf("no ParseGoHandle for %s", uri)
}

func (p *pkg) GetSyntax(ctx context.Context) []*ast.File {
	var syntax []*ast.File
	for _, ph := range p.files {
		file, _, _, err := ph.Cached(ctx)
		if err == nil {
			syntax = append(syntax, file)
		}
	}
	return syntax
}

func (p *pkg) GetErrors() []packages.Error {
	return p.errors
}

func (p *pkg) GetTypes() *types.Package {
	return p.types
}

func (p *pkg) GetTypesInfo() *types.Info {
	return p.typesInfo
}

func (p *pkg) GetTypesSizes() types.Sizes {
	return p.typesSizes
}

func (p *pkg) IsIllTyped() bool {
	return p.types == nil || p.typesInfo == nil || p.typesSizes == nil
}

func (p *pkg) GetImport(ctx context.Context, pkgPath string) (source.Package, error) {
	if imp := p.imports[packagePath(pkgPath)]; imp != nil {
		return imp, nil
	}
	// Don't return a nil pointer because that still satisfies the interface.
	return nil, errors.Errorf("no imported package for %s", pkgPath)
}

func (p *pkg) SetDiagnostics(a *analysis.Analyzer, diags []source.Diagnostic) {
	p.diagMu.Lock()
	defer p.diagMu.Unlock()
	if p.diagnostics == nil {
		p.diagnostics = make(map[*analysis.Analyzer][]source.Diagnostic)
	}
	p.diagnostics[a] = diags
}

func (pkg *pkg) FindDiagnostic(pdiag protocol.Diagnostic) (*source.Diagnostic, error) {
	pkg.diagMu.Lock()
	defer pkg.diagMu.Unlock()

	for a, diagnostics := range pkg.diagnostics {
		if a.Name != pdiag.Source {
			continue
		}
		for _, d := range diagnostics {
			if d.Message != pdiag.Message {
				continue
			}
			if protocol.CompareRange(d.Range, pdiag.Range) != 0 {
				continue
			}
			return &d, nil
		}
	}
	return nil, errors.Errorf("no matching diagnostic for %v", pdiag)
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
