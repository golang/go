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
	metadata map[PackageID]*KnownMetadata

	// importedBy maps package IDs to the list of packages that import them.
	importedBy map[PackageID][]PackageID

	// ids maps file URIs to package IDs, sorted by (!valid, cli, packageID).
	// A single file may belong to multiple packages due to tests packages.
	ids map[span.URI][]PackageID
}

// Clone creates a new metadataGraph, applying the given updates to the
// receiver.
func (g *metadataGraph) Clone(updates map[PackageID]*KnownMetadata) *metadataGraph {
	if len(updates) == 0 {
		// Optimization: since the graph is immutable, we can return the receiver.
		return g
	}
	result := &metadataGraph{metadata: make(map[PackageID]*KnownMetadata, len(g.metadata))}
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
		for _, importID := range uniqueDeps(m.Imports) {
			g.importedBy[importID] = append(g.importedBy[importID], id)
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
	//
	// We choose the first non-empty set of package associations out of the
	// following. For simplicity, call a non-command-line-arguments package a
	// "real" package.
	//
	// 1: valid real packages
	// 2: a valid command-line-arguments package
	// 3: invalid real packages
	// 4: an invalid command-line-arguments package
	for uri, ids := range g.ids {
		sort.Slice(ids, func(i, j int) bool {
			// 1. valid packages appear earlier.
			validi := g.metadata[ids[i]].Valid
			validj := g.metadata[ids[j]].Valid
			if validi != validj {
				return validi
			}

			// 2. command-line-args packages appear later.
			cli := source.IsCommandLineArguments(string(ids[i]))
			clj := source.IsCommandLineArguments(string(ids[j]))
			if cli != clj {
				return clj
			}

			// 3. packages appear in name order.
			return ids[i] < ids[j]
		})

		// Choose the best IDs for each URI, according to the following rules:
		//  - If there are any valid real packages, choose them.
		//  - Else, choose the first valid command-line-argument package, if it exists.
		//  - Else, keep using all the invalid metadata.
		//
		// TODO(rfindley): it might be better to track all IDs here, and exclude
		// them later in PackagesForFile, but this is the existing behavior.
		hasValidMetadata := false
		for i, id := range ids {
			m := g.metadata[id]
			if m.Valid {
				hasValidMetadata = true
			} else if hasValidMetadata {
				g.ids[uri] = ids[:i]
				break
			}
			// If we've seen *anything* prior to command-line arguments package, take
			// it. Note that ids[0] may itself be command-line-arguments.
			if i > 0 && source.IsCommandLineArguments(string(id)) {
				g.ids[uri] = ids[:i]
				break
			}
		}
	}
}

// uniqueDeps returns a new sorted and duplicate-free slice containing the
// IDs of the package's direct dependencies.
func uniqueDeps(imports map[ImportPath]PackageID) []PackageID {
	// TODO(adonovan): use generic maps.SortedUniqueValues(m.Imports) when available.
	ids := make([]PackageID, 0, len(imports))
	for _, id := range imports {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	// de-duplicate in place
	out := ids[:0]
	for _, id := range ids {
		if len(out) == 0 || id != out[len(out)-1] {
			out = append(out, id)
		}
	}
	return out
}

// reverseTransitiveClosure calculates the set of packages that transitively
// import an id in ids. The result also includes given ids.
//
// If includeInvalid is false, the algorithm ignores packages with invalid
// metadata (including those in the given list of ids).
func (g *metadataGraph) reverseTransitiveClosure(includeInvalid bool, ids ...PackageID) map[PackageID]bool {
	seen := make(map[PackageID]bool)
	var visitAll func([]PackageID)
	visitAll = func(ids []PackageID) {
		for _, id := range ids {
			if seen[id] {
				continue
			}
			m := g.metadata[id]
			// Only use invalid metadata if we support it.
			if m == nil || !(m.Valid || includeInvalid) {
				continue
			}
			seen[id] = true
			visitAll(g.importedBy[id])
		}
	}
	visitAll(ids)
	return seen
}
