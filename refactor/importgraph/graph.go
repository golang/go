// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package importgraph computes the forward and reverse import
// dependency graphs for all packages in a Go workspace.
package importgraph // import "golang.org/x/tools/refactor/importgraph"

import (
	"go/build"
	"sync"

	"golang.org/x/tools/go/buildutil"
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

// Build scans the specified Go workspace and builds the forward and
// reverse import dependency graphs for all its packages.
// It also returns a mapping from canonical import paths to errors for packages
// whose loading was not entirely successful.
// A package may appear in the graph and in the errors mapping.
// All package paths are canonical and may contain "/vendor/".
func Build(ctxt *build.Context) (forward, reverse Graph, errors map[string]error) {
	type importEdge struct {
		from, to string
	}
	type pathError struct {
		path string
		err  error
	}

	ch := make(chan interface{})

	go func() {
		sema := make(chan int, 20) // I/O concurrency limiting semaphore
		var wg sync.WaitGroup
		buildutil.ForEachPackage(ctxt, func(path string, err error) {
			if err != nil {
				ch <- pathError{path, err}
				return
			}

			wg.Add(1)
			go func() {
				defer wg.Done()

				sema <- 1
				bp, err := ctxt.Import(path, "", 0)
				<-sema

				if err != nil {
					if _, ok := err.(*build.NoGoError); ok {
						// empty directory is not an error
					} else {
						ch <- pathError{path, err}
					}
					// Even in error cases, Import usually returns a package.
				}

				// absolutize resolves an import path relative
				// to the current package bp.
				// The absolute form may contain "vendor".
				//
				// The vendoring feature slows down Build by 3×.
				// Here are timings from a 1400 package workspace:
				//    1100ms: current code (with vendor check)
				//     880ms: with a nonblocking cache around ctxt.IsDir
				//     840ms: nonblocking cache with duplicate suppression
				//     340ms: original code (no vendor check)
				// TODO(adonovan): optimize, somehow.
				memo := make(map[string]string)
				absolutize := func(path string) string {
					canon, ok := memo[path]
					if !ok {
						sema <- 1
						bp2, _ := ctxt.Import(path, bp.Dir, build.FindOnly)
						<-sema

						if bp2 != nil {
							canon = bp2.ImportPath
						} else {
							canon = path
						}
						memo[path] = canon
					}
					return canon
				}

				if bp != nil {
					for _, imp := range bp.Imports {
						ch <- importEdge{path, absolutize(imp)}
					}
					for _, imp := range bp.TestImports {
						ch <- importEdge{path, absolutize(imp)}
					}
					for _, imp := range bp.XTestImports {
						ch <- importEdge{path, absolutize(imp)}
					}
				}

			}()
		})
		wg.Wait()
		close(ch)
	}()

	forward = make(Graph)
	reverse = make(Graph)

	for e := range ch {
		switch e := e.(type) {
		case pathError:
			if errors == nil {
				errors = make(map[string]error)
			}
			errors[e.path] = e.err

		case importEdge:
			if e.to == "C" {
				continue // "C" is fake
			}
			forward.addEdge(e.from, e.to)
			reverse.addEdge(e.to, e.from)
		}
	}

	return forward, reverse, errors
}
