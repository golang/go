// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package importgraph computes the forward and reverse import
// dependency graphs for all packages in a Go workspace.
package importgraph

import (
	"go/build"
	"sync"

	"code.google.com/p/go.tools/go/buildutil"
)

// A Graph is an import dependency graph, either forward or reverse.
//
// The graph maps each node (a package import path) to the set of its
// successors in the graph.  For a forward graph, this is the set of
// imported packages (prerequisites); for a reverse graph, it is the set
// of importing packages (clients).
//
type Graph map[string]map[string]bool

func (g Graph) addEdge(from, to string) {
	edges := g[from]
	if edges == nil {
		edges = make(map[string]bool)
		g[from] = edges
	}
	edges[to] = true
}

// Search returns all the nodes of the graph reachable from
// any of the specified roots, by following edges forwards.
// Relationally, this is the reflexive transitive closure.
func (g Graph) Search(roots ...string) map[string]bool {
	seen := make(map[string]bool)
	var visit func(x string)
	visit = func(x string) {
		if !seen[x] {
			seen[x] = true
			for y := range g[x] {
				visit(y)
			}
		}
	}
	for _, root := range roots {
		visit(root)
	}
	return seen
}

// Builds scans the specified Go workspace and builds the forward and
// reverse import dependency graphs for all its packages.
// It also returns a mapping from import paths to errors for packages
// that could not be loaded.
func Build(ctxt *build.Context) (forward, reverse Graph, errors map[string]error) {
	// The (sole) graph builder goroutine receives a stream of import
	// edges from the package loading goroutine.
	forward = make(Graph)
	reverse = make(Graph)
	edgec := make(chan [2]string)
	go func() {
		for edge := range edgec {
			if edge[1] == "C" {
				continue // "C" is fake
			}
			forward.addEdge(edge[0], edge[1])
			reverse.addEdge(edge[1], edge[0])
		}
	}()

	// The (sole) error goroutine receives a stream of ReadDir and
	// Import errors.
	type pathError struct {
		path string
		err  error
	}
	errorc := make(chan pathError)
	go func() {
		for e := range errorc {
			if errors == nil {
				errors = make(map[string]error)
			}
			errors[e.path] = e.err
		}
	}()

	var wg sync.WaitGroup
	buildutil.ForEachPackage(ctxt, func(path string, err error) {
		if err != nil {
			errorc <- pathError{path, err}
			return
		}
		wg.Add(1)
		// The import goroutines load the metadata for each package.
		go func(path string) {
			defer wg.Done()
			bp, err := ctxt.Import(path, "", 0)
			if _, ok := err.(*build.NoGoError); ok {
				return // empty directory is not an error
			}
			if err != nil {
				errorc <- pathError{path, err}
				return
			}
			for _, imp := range bp.Imports {
				edgec <- [2]string{path, imp}
			}
			for _, imp := range bp.TestImports {
				edgec <- [2]string{path, imp}
			}
			for _, imp := range bp.XTestImports {
				edgec <- [2]string{path, imp}
			}
		}(path)
	})
	wg.Wait()

	close(edgec)
	close(errorc)

	return forward, reverse, errors
}
