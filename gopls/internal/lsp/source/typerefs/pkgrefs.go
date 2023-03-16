// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"context"
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

	// debug enables additional assertions.
	debug = false
)

// A Package holds reference information for a single package.
type Package struct {
	// metadata holds metadata about this package and its dependencies.
	metadata *source.Metadata

	// transitiveRefs records, for each exported declaration in the package, the
	// transitive set of packages within the containing graph that are
	// transitively reachable through references, starting with the given decl.
	transitiveRefs map[string]*PackageSet

	// ReachesViaDeps records the set of packages in the containing graph whose
	// syntax may affect the current package's types. See the package
	// documentation for more details of what this means.
	ReachesByDeps *PackageSet
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
		metadata:       g.meta.Metadata(id),
		transitiveRefs: make(map[string]*PackageSet),
	}
	var files []*source.ParsedGoFile
	for _, filename := range p.metadata.CompiledGoFiles {
		f, err := g.parse(ctx, filename)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	imports := make(map[source.ImportPath]*source.Metadata)
	for impPath, depID := range p.metadata.DepsByImpPath {
		if depID != "" {
			imports[impPath] = g.meta.Metadata(depID)
		}
	}

	// Compute the symbol-level dependencies through this package.
	// TODO(adonovan): opt: persist this in the filecache, keyed
	// by hash(id, CompiledGoFiles, imports).
	data := Encode(files, id, imports)

	//      This point separates the local preprocessing
	//  --  of a single package (above) from the global   --
	//      transitive reachability query (below).

	// classes records syntactic edges between declarations in this
	// package and declarations in this package or another
	// package. See the package documentation for a detailed
	// description of what these edges do (and do not) represent.
	classes := Decode(g.pkgIndex, id, data)

	idx := g.pkgIndex.idx(id)

	// Now compute the transitive closure of packages reachable
	// from any exported symbol of this package.
	for _, class := range classes {
		set := g.pkgIndex.NewSet()

		// The Refs slice is sorted by (PackageID, name),
		// so we can economize by calling g.Package only
		// when the package id changes.
		depP := p
		for _, sym := range class.Refs {
			assert(sym.pkgIdx != idx, "intra-package edge")
			symPkgID := g.pkgIndex.id(sym.pkgIdx)
			if depP.metadata.ID != symPkgID {
				// package changed
				var err error
				depP, err = g.Package(ctx, symPkgID)
				if err != nil {
					return nil, err
				}
			}
			set.add(sym.pkgIdx)
			set.Union(depP.transitiveRefs[sym.Name])
		}
		for _, name := range class.Decls {
			p.transitiveRefs[name] = set
		}
	}

	// Finally compute the union of transitiveRefs
	// across the direct deps of this package.
	byDeps, err := g.reachesByDeps(ctx, p.metadata)
	if err != nil {
		return nil, err
	}
	p.ReachesByDeps = byDeps

	return p, nil
}

// reachesByDeps computes the set of packages that are reachable through
// dependencies of the package m.
func (g *PackageGraph) reachesByDeps(ctx context.Context, m *source.Metadata) (*PackageSet, error) {
	transitive := g.pkgIndex.NewSet()
	for _, depID := range m.DepsByPkgPath {
		dep, err := g.Package(ctx, depID)
		if err != nil {
			return nil, err
		}
		transitive.Add(dep.metadata.ID)
		for _, set := range dep.transitiveRefs {
			transitive.Union(set)
		}
	}
	return transitive, nil
}
