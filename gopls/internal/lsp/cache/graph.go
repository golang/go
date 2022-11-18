// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"sort"

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
	ids map[span.URI][]PackageID
}

// Clone creates a new metadataGraph, applying the given updates to the
// receiver.
func (g *metadataGraph) Clone(updates map[PackageID]*source.Metadata) *metadataGraph {
	if len(updates) == 0 {
		// Optimization: since the graph is immutable, we can return the receiver.
		return g
	}
	result := &metadataGraph{metadata: make(map[PackageID]*source.Metadata, len(g.metadata))}
	// Copy metadata.
	for id, m := range g.metadata {
		result.metadata[id] = m
	}
	for id, m := range updates {
		if m == nil {
			delete(result.metadata, id)
		} else {
			result.metadata[id] = m
		}
	}
	result.build()
	return result
}

// build constructs g.importedBy and g.uris from g.metadata.
func (g *metadataGraph) build() {
	// Build the import graph.
	g.importedBy = make(map[PackageID][]PackageID)
	for id, m := range g.metadata {
		for _, depID := range m.DepsByPkgPath {
			g.importedBy[depID] = append(g.importedBy[depID], id)
		}
	}

	// Collect file associations.
	g.ids = make(map[span.URI][]PackageID)
	for id, m := range g.metadata {
		uris := map[span.URI]struct{}{}
		for _, uri := range m.CompiledGoFiles {
			uris[uri] = struct{}{}
		}
		for _, uri := range m.GoFiles {
			uris[uri] = struct{}{}
		}
		for uri := range uris {
			g.ids[uri] = append(g.ids[uri], id)
		}
	}

	// Sort and filter file associations.
	for uri, ids := range g.ids {
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
		// them later in PackagesForFile, but this is the existing behavior.
		for i, id := range ids {
			// If we've seen *anything* prior to command-line arguments package, take
			// it. Note that ids[0] may itself be command-line-arguments.
			if i > 0 && source.IsCommandLineArguments(id) {
				g.ids[uri] = ids[:i]
				break
			}
		}
	}
}

// reverseTransitiveClosure calculates the set of packages that transitively
// import an id in ids. The result also includes given ids.
//
// If includeInvalid is false, the algorithm ignores packages with invalid
// metadata (including those in the given list of ids).
func (g *metadataGraph) reverseTransitiveClosure(ids ...PackageID) map[PackageID]bool {
	seen := make(map[PackageID]bool)
	var visitAll func([]PackageID)
	visitAll = func(ids []PackageID) {
		for _, id := range ids {
			if seen[id] {
				continue
			}
			m := g.metadata[id]
			// Only use invalid metadata if we support it.
			if m == nil {
				continue
			}
			seen[id] = true
			visitAll(g.importedBy[id])
		}
	}
	visitAll(ids)
	return seen
}
