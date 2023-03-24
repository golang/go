// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"context"
	"go/token"
	"runtime"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

const (
	// trace enables additional trace output to stdout, for debugging.
	//
	// Warning: produces a lot of output! Best to run with small package queries.
	trace = false
)

// A Package holds reference information for a single package.
type Package struct {
	idx packageIdx // memoized index of this package's ID, to save map lookups

	// Metadata holds metadata about this package and its dependencies.
	Metadata *source.Metadata

	// Refs records syntactic edges between declarations in this package and
	// declarations in this package or another package. See the package
	// documentation for a detailed description of what these edges do (and do
	// not) represent.
	Refs map[string][]Ref

	// TransitiveRefs records, for each declaration in the package, the
	// transitive set of packages within the containing graph that are
	// transitively reachable through references, starting with the given decl.
	TransitiveRefs map[string]*PackageSet

	// ReachesViaDeps records the set of packages in the containing graph whose
	// syntax may affect the current package's types. See the package
	// documentation for more details of what this means.
	ReachesByDeps *PackageSet
}

// A Ref is a referenced declaration.
//
// Unpack it using the Unpack method, with the PackageIndex instance that was
// used to construct the references.
type Ref struct {
	pkg  packageIdx
	name string
}

// UnpackRef unpacks the actual PackageID an name encoded in ref.
func (r Ref) Unpack(index *PackageIndex) (PackageID source.PackageID, name string) {
	return index.id(r.pkg), r.name
}

// A PackageGraph represents a fully analyzed graph of packages and their
// dependencies.
type PackageGraph struct {
	pkgIndex *PackageIndex
	meta     source.MetadataSource
	parse    func(context.Context, span.URI) (*source.ParsedGoFile, error)

	mu       sync.Mutex
	packages map[source.PackageID]*futurePackage
}

// BuildPackageGraph analyzes the package graph for the requested ids, whose
// metadata is described by meta.
//
// The provided parse function is used to parse the CompiledGoFiles of each package.
//
// The resulting PackageGraph is fully evaluated, and may be investigated using
// the Package method.
//
// See the package documentation for more information on the package reference
// algorithm.
func BuildPackageGraph(ctx context.Context, meta source.MetadataSource, ids []source.PackageID, parse func(context.Context, span.URI) (*source.ParsedGoFile, error)) (*PackageGraph, error) {
	g := &PackageGraph{
		pkgIndex: NewPackageIndex(),
		meta:     meta,
		parse:    parse,
		packages: make(map[source.PackageID]*futurePackage),
	}
	source.SortPostOrder(meta, ids)

	workers := runtime.GOMAXPROCS(0)
	if trace {
		workers = 1
	}

	var eg errgroup.Group
	eg.SetLimit(workers)
	for _, id := range ids {
		id := id
		eg.Go(func() error {
			_, err := g.Package(ctx, id)
			return err
		})
	}
	return g, eg.Wait()
}

// futurePackage is a future result of analyzing a package, for use from Package only.
type futurePackage struct {
	done chan struct{}
	pkg  *Package
	err  error
}

// Package gets the result of analyzing references for a single package.
func (g *PackageGraph) Package(ctx context.Context, id source.PackageID) (*Package, error) {
	g.mu.Lock()
	fut, ok := g.packages[id]
	if ok {
		g.mu.Unlock()
		select {
		case <-fut.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	} else {
		fut = &futurePackage{done: make(chan struct{})}
		g.packages[id] = fut
		g.mu.Unlock()
		fut.pkg, fut.err = g.buildPackage(ctx, id)
		close(fut.done)
	}
	return fut.pkg, fut.err
}

// buildPackage parses a package and extracts its reference graph. It should
// only be called from Package.
func (g *PackageGraph) buildPackage(ctx context.Context, id source.PackageID) (*Package, error) {
	p := &Package{
		idx:            g.pkgIndex.idx(id),
		Metadata:       g.meta.Metadata(id),
		Refs:           make(map[string][]Ref),
		TransitiveRefs: make(map[string]*PackageSet),
	}
	var files []*source.ParsedGoFile
	for _, filename := range p.Metadata.CompiledGoFiles {
		f, err := g.parse(ctx, filename)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	imports := make(map[source.ImportPath]*source.Metadata)
	for impPath, depID := range p.Metadata.DepsByImpPath {
		if depID != "" {
			imports[impPath] = g.meta.Metadata(depID)
		}
	}
	p.Refs = Refs(files, id, imports, g.pkgIndex)

	// Compute packages reachable from each exported symbol of this package.
	for name := range p.Refs {
		if token.IsExported(name) {
			set := g.pkgIndex.New()
			g.reachableByName(ctx, p, name, set, make(map[string]bool))
			p.TransitiveRefs[name] = set
		}
	}

	var err error
	p.ReachesByDeps, err = g.reachesByDeps(ctx, p.Metadata)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// reachableByName computes the set of packages that are reachable through
// references, starting with the declaration for name in package p.
func (g *PackageGraph) reachableByName(ctx context.Context, p *Package, name string, set *PackageSet, seen map[string]bool) error {
	if seen[name] {
		return nil
	}
	seen[name] = true

	// Opt: when we compact reachable edges inside the Refs algorithm, we handle
	// all edges to a given package in a batch, so they should be adjacent to
	// each other in the resulting slice. Therefore remembering the last P here
	// can save on lookups.
	depP := p
	for _, node := range p.Refs[name] {
		if node.pkg == p.idx {
			// same package
			g.reachableByName(ctx, p, node.name, set, seen)
		} else {
			// cross-package ref
			if depP.idx != node.pkg {
				id := g.pkgIndex.id(node.pkg)
				var err error
				depP, err = g.Package(ctx, id)
				if err != nil {
					return err
				}
			}
			set.add(node.pkg)
			set.Union(depP.TransitiveRefs[node.name])
		}
	}

	return nil
}

// reachesByDeps computes the set of packages that are reachable through
// dependencies of the package m.
func (g *PackageGraph) reachesByDeps(ctx context.Context, m *source.Metadata) (*PackageSet, error) {
	transitive := g.pkgIndex.New()
	for _, depID := range m.DepsByPkgPath {
		dep, err := g.Package(ctx, depID)
		if err != nil {
			return nil, err
		}
		transitive.add(dep.idx)
		for name, set := range dep.TransitiveRefs {
			if token.IsExported(name) {
				transitive.Union(set)
			}
		}
	}
	return transitive, nil
}
