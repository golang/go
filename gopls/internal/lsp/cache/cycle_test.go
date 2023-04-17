// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/source"
)

// This is an internal test of the breakImportCycles logic.
func TestBreakImportCycles(t *testing.T) {

	type Graph = map[PackageID]*source.Metadata

	// cyclic returns a description of a cycle,
	// if the graph is cyclic, otherwise "".
	cyclic := func(graph Graph) string {
		const (
			unvisited = 0
			visited   = 1
			onstack   = 2
		)
		color := make(map[PackageID]int)
		var visit func(id PackageID) string
		visit = func(id PackageID) string {
			switch color[id] {
			case unvisited:
				color[id] = onstack
			case onstack:
				return string(id) // cycle!
			case visited:
				return ""
			}
			if m := graph[id]; m != nil {
				for _, depID := range m.DepsByPkgPath {
					if cycle := visit(depID); cycle != "" {
						return string(id) + "->" + cycle
					}
				}
			}
			color[id] = visited
			return ""
		}
		for id := range graph {
			if cycle := visit(id); cycle != "" {
				return cycle
			}
		}
		return ""
	}

	// parse parses an import dependency graph.
	// The input is a semicolon-separated list of node descriptions.
	// Each node description is a package ID, optionally followed by
	// "->" and a comma-separated list of successor IDs.
	// Thus "a->b;b->c,d;e" represents the set of nodes {a,b,e}
	// and the set of edges {a->b, b->c, b->d}.
	parse := func(s string) Graph {
		m := make(Graph)
		makeNode := func(name string) *source.Metadata {
			id := PackageID(name)
			n, ok := m[id]
			if !ok {
				n = &source.Metadata{
					ID:            id,
					DepsByPkgPath: make(map[PackagePath]PackageID),
				}
				m[id] = n
			}
			return n
		}
		if s != "" {
			for _, item := range strings.Split(s, ";") {
				nodeID, succIDs, ok := cut(item, "->")
				node := makeNode(nodeID)
				if ok {
					for _, succID := range strings.Split(succIDs, ",") {
						node.DepsByPkgPath[PackagePath(succID)] = PackageID(succID)
					}
				}
			}
		}
		return m
	}

	// Sanity check of cycle detector.
	{
		got := cyclic(parse("a->b;b->c;c->a,d"))
		has := func(s string) bool { return strings.Contains(got, s) }
		if !(has("a->b") && has("b->c") && has("c->a") && !has("d")) {
			t.Fatalf("cyclic: got %q, want a->b->c->a or equivalent", got)
		}
	}

	// format formats an import graph, in lexicographic order,
	// in the notation of parse, but with a "!" after the name
	// of each node that has errors.
	format := func(graph Graph) string {
		var items []string
		for _, m := range graph {
			item := string(m.ID)
			if len(m.Errors) > 0 {
				item += "!"
			}
			var succs []string
			for _, depID := range m.DepsByPkgPath {
				succs = append(succs, string(depID))
			}
			if succs != nil {
				sort.Strings(succs)
				item += "->" + strings.Join(succs, ",")
			}
			items = append(items, item)
		}
		sort.Strings(items)
		return strings.Join(items, ";")
	}

	// We needn't test self-cycles as they are eliminated at Metadata construction.
	for _, test := range []struct {
		metadata, updates, want string
	}{
		// Simple 2-cycle.
		{"a->b", "b->a",
			"a->b;b!"}, // broke b->a

		{"a->b;b->c;c", "b->a,c",
			"a->b;b!->c;c"}, // broke b->a

		// Reversing direction of p->s edge creates pqrs cycle.
		{"a->p,q,r,s;p->q,s,z;q->r,z;r->s,z;s->z", "p->q,z;s->p,z",
			"a->p,q,r,s;p!->z;q->r,z;r->s,z;s!->z"}, // broke p->q, s->p

		// We break all intra-SCC edges from updated nodes,
		// which may be more than necessary (e.g. a->b).
		{"a->b;b->c;c;d->a", "a->b,e;c->d",
			"a!->e;b->c;c!;d->a"}, // broke a->b, c->d
	} {
		metadata := parse(test.metadata)
		updates := parse(test.updates)

		if cycle := cyclic(metadata); cycle != "" {
			t.Errorf("initial metadata %s has cycle %s: ", format(metadata), cycle)
			continue
		}

		t.Log("initial", format(metadata))

		// Apply updates.
		// (parse doesn't have a way to express node deletions,
		// but they aren't very interesting.)
		for id, m := range updates {
			metadata[id] = m
		}

		t.Log("updated", format(metadata))

		// breakImportCycles accesses only these fields of Metadata:
		//    DepsByImpPath, ID - read
		//    DepsByPkgPath     - read, updated
		//    Errors            - updated
		breakImportCycles(metadata, updates)

		t.Log("acyclic", format(metadata))

		if cycle := cyclic(metadata); cycle != "" {
			t.Errorf("resulting metadata %s has cycle %s: ", format(metadata), cycle)
		}

		got := format(metadata)
		if got != test.want {
			t.Errorf("test.metadata=%s test.updates=%s: got=%s want=%s",
				test.metadata, test.updates, got, test.want)
		}
	}
}
