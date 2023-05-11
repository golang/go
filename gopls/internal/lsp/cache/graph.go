// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"sort"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

// A metadataGraph is an immutable and transitively closed import
// graph of Go packages, as obtained from go/packages.
type metadataGraph struct {
	// metadata maps package IDs to their associated metadata.
	metadata map[PackageID]*source.Metadata

	// importedBy maps package IDs to the list of packages that import them.
	importedBy map[PackageID][]PackageID

	// ids maps file URIs to package IDs, sorted by (!valid, cli, packageID).
	// A single file may belong to multiple packages due to tests packages.
	//
	// Invariant: all IDs present in the ids map exist in the metadata map.
	ids map[span.URI][]PackageID
}

// Metadata implements the source.MetadataSource interface.
func (g *metadataGraph) Metadata(id PackageID) *source.Metadata {
	return g.metadata[id]
}

// Clone creates a new metadataGraph, applying the given updates to the
// receiver. A nil map value represents a deletion.
func (g *metadataGraph) Clone(updates map[PackageID]*source.Metadata) *metadataGraph {
	if len(updates) == 0 {
		// Optimization: since the graph is immutable, we can return the receiver.
		return g
	}

	// Copy metadata map then apply updates.
	metadata := make(map[PackageID]*source.Metadata, len(g.metadata))
	for id, m := range g.metadata {
		metadata[id] = m
	}
	for id, m := range updates {
		if m == nil {
			delete(metadata, id)
		} else {
			metadata[id] = m
		}
	}

	// Break import cycles involving updated nodes.
	breakImportCycles(metadata, updates)

	return newMetadataGraph(metadata)
}

// newMetadataGraph returns a new metadataGraph,
// deriving relations from the specified metadata.
func newMetadataGraph(metadata map[PackageID]*source.Metadata) *metadataGraph {
	// Build the import graph.
	importedBy := make(map[PackageID][]PackageID)
	for id, m := range metadata {
		for _, depID := range m.DepsByPkgPath {
			importedBy[depID] = append(importedBy[depID], id)
		}
	}

	// Collect file associations.
	uriIDs := make(map[span.URI][]PackageID)
	for id, m := range metadata {
		uris := map[span.URI]struct{}{}
		for _, uri := range m.CompiledGoFiles {
			uris[uri] = struct{}{}
		}
		for _, uri := range m.GoFiles {
			uris[uri] = struct{}{}
		}
		for uri := range uris {
			uriIDs[uri] = append(uriIDs[uri], id)
		}
	}

	// Sort and filter file associations.
	for uri, ids := range uriIDs {
		sort.Slice(ids, func(i, j int) bool {
			cli := source.IsCommandLineArguments(ids[i])
			clj := source.IsCommandLineArguments(ids[j])
			if cli != clj {
				return clj
			}

			// 2. packages appear in name order.
			return ids[i] < ids[j]
		})

		// Choose the best IDs for each URI, according to the following rules:
		//  - If there are any valid real packages, choose them.
		//  - Else, choose the first valid command-line-argument package, if it exists.
		//
		// TODO(rfindley): it might be better to track all IDs here, and exclude
		// them later when type checking, but this is the existing behavior.
		for i, id := range ids {
			// If we've seen *anything* prior to command-line arguments package, take
			// it. Note that ids[0] may itself be command-line-arguments.
			if i > 0 && source.IsCommandLineArguments(id) {
				uriIDs[uri] = ids[:i]
				break
			}
		}
	}

	return &metadataGraph{
		metadata:   metadata,
		importedBy: importedBy,
		ids:        uriIDs,
	}
}

// reverseReflexiveTransitiveClosure returns a new mapping containing the
// metadata for the specified packages along with any package that
// transitively imports one of them, keyed by ID, including all the initial packages.
func (g *metadataGraph) reverseReflexiveTransitiveClosure(ids ...PackageID) map[PackageID]*source.Metadata {
	seen := make(map[PackageID]*source.Metadata)
	var visitAll func([]PackageID)
	visitAll = func(ids []PackageID) {
		for _, id := range ids {
			if seen[id] == nil {
				if m := g.metadata[id]; m != nil {
					seen[id] = m
					visitAll(g.importedBy[id])
				}
			}
		}
	}
	visitAll(ids)
	return seen
}

// breakImportCycles breaks import cycles in the metadata by deleting
// Deps* edges. It modifies only metadata present in the 'updates'
// subset. This function has an internal test.
func breakImportCycles(metadata, updates map[PackageID]*source.Metadata) {
	// 'go list' should never report a cycle without flagging it
	// as such, but we're extra cautious since we're combining
	// information from multiple runs of 'go list'. Also, Bazel
	// may silently report cycles.
	cycles := detectImportCycles(metadata, updates)
	if len(cycles) > 0 {
		// There were cycles (uncommon). Break them.
		//
		// The naive way to break cycles would be to perform a
		// depth-first traversal and to detect and delete
		// cycle-forming edges as we encounter them.
		// However, we're not allowed to modify the existing
		// Metadata records, so we can only break edges out of
		// the 'updates' subset.
		//
		// Another possibility would be to delete not the
		// cycle forming edge but the topmost edge on the
		// stack whose tail is an updated node.
		// However, this would require that we retroactively
		// undo all the effects of the traversals that
		// occurred since that edge was pushed on the stack.
		//
		// We use a simpler scheme: we compute the set of cycles.
		// All cyclic paths necessarily involve at least one
		// updated node, so it is sufficient to break all
		// edges from each updated node to other members of
		// the strong component.
		//
		// This may result in the deletion of dominating
		// edges, causing some dependencies to appear
		// spuriously unreachable. Consider A <-> B -> C
		// where updates={A,B}. The cycle is {A,B} so the
		// algorithm will break both A->B and B->A, causing
		// A to no longer depend on B or C.
		//
		// But that's ok: any error in Metadata.Errors is
		// conservatively assumed by snapshot.clone to be a
		// potential import cycle error, and causes special
		// invalidation so that if B later drops its
		// cycle-forming import of A, both A and B will be
		// invalidated.
		for _, cycle := range cycles {
			cyclic := make(map[PackageID]bool)
			for _, m := range cycle {
				cyclic[m.ID] = true
			}
			for id := range cyclic {
				if m := updates[id]; m != nil {
					for path, depID := range m.DepsByImpPath {
						if cyclic[depID] {
							delete(m.DepsByImpPath, path)
						}
					}
					for path, depID := range m.DepsByPkgPath {
						if cyclic[depID] {
							delete(m.DepsByPkgPath, path)
						}
					}

					// Set m.Errors to enable special
					// invalidation logic in snapshot.clone.
					if len(m.Errors) == 0 {
						m.Errors = []packages.Error{{
							Msg:  "detected import cycle",
							Kind: packages.ListError,
						}}
					}
				}
			}
		}

		// double-check when debugging
		if false {
			if cycles := detectImportCycles(metadata, updates); len(cycles) > 0 {
				bug.Reportf("unbroken cycle: %v", cycles)
			}
		}
	}
}

// detectImportCycles reports cycles in the metadata graph. It returns a new
// unordered array of all cycles (nontrivial strong components) in the
// metadata graph reachable from a non-nil 'updates' value.
func detectImportCycles(metadata, updates map[PackageID]*source.Metadata) [][]*source.Metadata {
	// We use the depth-first algorithm of Tarjan.
	// https://doi.org/10.1137/0201010
	//
	// TODO(adonovan): when we can use generics, consider factoring
	// in common with the other implementation of Tarjan (in typerefs),
	// abstracting over the node and edge representation.

	// A node wraps a Metadata with its working state.
	// (Unfortunately we can't intrude on shared Metadata.)
	type node struct {
		rep            *node
		m              *source.Metadata
		index, lowlink int32
		scc            int8 // TODO(adonovan): opt: cram these 1.5 bits into previous word
	}
	nodes := make(map[PackageID]*node, len(metadata))
	nodeOf := func(id PackageID) *node {
		n, ok := nodes[id]
		if !ok {
			m := metadata[id]
			if m == nil {
				// Dangling import edge.
				// Not sure whether a go/packages driver ever
				// emits this, but create a dummy node in case.
				// Obviously it won't be part of any cycle.
				m = &source.Metadata{ID: id}
			}
			n = &node{m: m}
			n.rep = n
			nodes[id] = n
		}
		return n
	}

	// find returns the canonical node decl.
	// (The nodes form a disjoint set forest.)
	var find func(*node) *node
	find = func(n *node) *node {
		rep := n.rep
		if rep != n {
			rep = find(rep)
			n.rep = rep // simple path compression (no union-by-rank)
		}
		return rep
	}

	// global state
	var (
		index int32 = 1
		stack []*node
		sccs  [][]*source.Metadata // set of nontrivial strongly connected components
	)

	// visit implements the depth-first search of Tarjan's SCC algorithm
	// Precondition: x is canonical.
	var visit func(*node)
	visit = func(x *node) {
		x.index = index
		x.lowlink = index
		index++

		stack = append(stack, x) // push
		x.scc = -1

		for _, yid := range x.m.DepsByPkgPath {
			y := nodeOf(yid)
			// Loop invariant: x is canonical.
			y = find(y)
			if x == y {
				continue // nodes already combined (self-edges are impossible)
			}

			switch {
			case y.scc > 0:
				// y is already a collapsed SCC

			case y.scc < 0:
				// y is on the stack, and thus in the current SCC.
				if y.index < x.lowlink {
					x.lowlink = y.index
				}

			default:
				// y is unvisited; visit it now.
				visit(y)
				// Note: x and y are now non-canonical.
				x = find(x)
				if y.lowlink < x.lowlink {
					x.lowlink = y.lowlink
				}
			}
		}

		// Is x the root of an SCC?
		if x.lowlink == x.index {
			// Gather all metadata in the SCC (if nontrivial).
			var scc []*source.Metadata
			for {
				// Pop y from stack.
				i := len(stack) - 1
				y := stack[i]
				stack = stack[:i]
				if x != y || scc != nil {
					scc = append(scc, y.m)
				}
				if x == y {
					break // complete
				}
				// x becomes y's canonical representative.
				y.rep = x
			}
			if scc != nil {
				sccs = append(sccs, scc)
			}
			x.scc = 1
		}
	}

	// Visit only the updated nodes:
	// the existing metadata graph has no cycles,
	// so any new cycle must involve an updated node.
	for id, m := range updates {
		if m != nil {
			if n := nodeOf(id); n.index == 0 { // unvisited
				visit(n)
			}
		}
	}

	return sccs
}
