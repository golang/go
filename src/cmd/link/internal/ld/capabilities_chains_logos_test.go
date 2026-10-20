// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

type capEdge [2]loader.Sym

func buildCapGraph(roots []loader.Sym, edges []capEdge) *capGraph {
	g := newCapGraph()
	adj := make(map[loader.Sym][]loader.Sym)
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], e[1])
	}
	var queue []loader.Sym
	for _, r := range roots {
		if g.visit(r, 0) {
			queue = append(queue, r)
		}
	}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for _, v := range adj[u] {
			if g.visit(v, u) {
				queue = append(queue, v)
			}
		}
	}
	return g
}

func capRoleTable(roles map[loader.Sym]capRole) func(loader.Sym) capRole {
	return func(s loader.Sym) capRole { return roles[s] }
}

func capChainNames(chain []loader.Sym, names map[loader.Sym]string) []string {
	var out []string
	for _, s := range chain {
		out = append(out, names[s])
	}
	return out
}

func TestCapGraphVisitDedupesEdges(t *testing.T) {
	g := newCapGraph()
	if !g.visit(1, 0) {
		t.Fatal("root must be new")
	}
	if !g.visit(2, 1) || g.visit(2, 1) {
		t.Fatal("second visit of the same edge must not be new")
	}
	g.visit(2, 1)
	g.visit(1, 2)
	g.visit(1, 2)
	if got := g.preds[2]; !slices.Equal(got, []loader.Sym{1}) {
		t.Errorf("preds[2] = %v, want [1]", got)
	}
	if got := g.preds[1]; !slices.Equal(got, []loader.Sym{2}) {
		t.Errorf("preds[1] = %v, want [2]: a root reached again keeps one pred", got)
	}
	if g.parent[1] != 0 {
		t.Errorf("root parent = %d, want 0", g.parent[1])
	}
}

func TestCapGraphChains(t *testing.T) {
	names := map[loader.Sym]string{
		1: "root", 2: "main", 3: "std.A", 4: "std.B", 5: "depx.F", 6: "depy.G",
		7: "sink", 8: "runtime.stack", 9: "depx.H", 10: "first.P", 11: "data",
	}
	dep := func(s string) capRole { return capRole{dep: s} }
	roles := map[loader.Sym]capRole{
		5: dep("depx"), 6: dep("depy"), 9: dep("depx"), 8: {wall: true},
	}
	tests := []struct {
		name  string
		roots []loader.Sym
		edges []capEdge
		sink  loader.Sym
		want  map[string][]string
	}{
		{
			name:  "exempt-only chain blames nobody",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 2}, {2, 3}, {3, 7}},
			sink:  7,
			want:  map[string][]string{},
		},
		{
			name:  "dependency chain survives a shorter exempt chain",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 3}, {3, 7}, {1, 2}, {2, 5}, {5, 4}, {4, 7}},
			sink:  7,
			want:  map[string][]string{"depx": {"root", "main", "depx.F", "std.B", "sink"}},
		},
		{
			name:  "first-party frames are exempt like standard library",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 2}, {2, 10}, {10, 7}},
			sink:  7,
			want:  map[string][]string{},
		},
		{
			name:  "each dependency is reported once with its own chain",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 5}, {1, 6}, {5, 3}, {6, 3}, {3, 7}},
			sink:  7,
			want: map[string][]string{
				"depx": {"root", "depx.F", "std.A", "sink"},
				"depy": {"root", "depy.G", "std.A", "sink"},
			},
		},
		{
			name:  "the dependency nearest the sink takes the blame",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 5}, {5, 6}, {6, 7}},
			sink:  7,
			want:  map[string][]string{"depy": {"root", "depx.F", "depy.G", "sink"}},
		},
		{
			name:  "a wall keeps implicit runtime edges out of the tail",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 2}, {2, 5}, {5, 8}, {8, 7}},
			sink:  7,
			want:  map[string][]string{},
		},
		{
			name:  "a wall before the blamed frame does not hide it",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 8}, {8, 5}, {5, 7}},
			sink:  7,
			want:  map[string][]string{"depx": {"root", "runtime.stack", "depx.F", "sink"}},
		},
		{
			name:  "a sink behind a wall still blames its direct caller",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 5}, {5, 8}},
			sink:  8,
			want:  map[string][]string{"depx": {"root", "depx.F", "runtime.stack"}},
		},
		{
			name:  "data symbols pass blame through",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 5}, {5, 11}, {11, 7}},
			sink:  7,
			want:  map[string][]string{"depx": {"root", "depx.F", "data", "sink"}},
		},
		{
			name:  "a sink the dependency owns blames it with an empty tail",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 2}, {2, 5}},
			sink:  5,
			want:  map[string][]string{"depx": {"root", "main", "depx.F"}},
		},
		{
			name:  "the shortest chain among a dependency's frames wins",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 2}, {2, 5}, {5, 9}, {9, 3}, {5, 3}, {3, 7}},
			sink:  7,
			want:  map[string][]string{"depx": {"root", "main", "depx.F", "std.A", "sink"}},
		},
		{
			name:  "a cycle through the tail repeats its frame",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 3}, {3, 5}, {5, 3}, {3, 7}},
			sink:  7,
			want:  map[string][]string{"depx": {"root", "std.A", "depx.F", "std.A", "sink"}},
		},
		{
			name:  "unreachable frames never blame",
			roots: []loader.Sym{1},
			edges: []capEdge{{1, 3}, {3, 7}, {6, 3}},
			sink:  7,
			want:  map[string][]string{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := buildCapGraph(tc.roots, tc.edges)
			got := make(map[string][]string)
			for _, b := range g.blamedChains(tc.sink, capRoleTable(roles)) {
				got[b.dep] = capChainNames(b.chain, names)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("blamed %v, want %v", got, tc.want)
			}
			for dep, want := range tc.want {
				if !slices.Equal(got[dep], want) {
					t.Errorf("chain for %s = %v, want %v", dep, got[dep], want)
				}
			}
		})
	}
}

func TestCapGraphChainsAreSortedByDependency(t *testing.T) {
	roles := map[loader.Sym]capRole{2: {dep: "c"}, 3: {dep: "a"}, 4: {dep: "b"}}
	g := buildCapGraph([]loader.Sym{1}, []capEdge{{1, 2}, {1, 3}, {1, 4}, {2, 5}, {3, 5}, {4, 5}})
	var got []string
	for _, b := range g.blamedChains(5, capRoleTable(roles)) {
		got = append(got, b.dep)
	}
	if !slices.Equal(got, []string{"a", "b", "c"}) {
		t.Errorf("dependencies = %v, want sorted", got)
	}
}

// capOracleChains computes the same answer by exhaustive search over
// (symbol, last blamed dependency, wall crossed since it) states: the
// length in symbols of the shortest walk from a root to the sink for each
// dependency, where the walk's last blamed frame carries that dependency
// and no wall follows it.
func capOracleChains(roots []loader.Sym, edges []capEdge, sink loader.Sym, role func(loader.Sym) capRole) map[string]int {
	type state struct {
		node loader.Sym
		dep  string
		wall bool
	}
	enter := func(s state, to loader.Sym) state {
		r := role(to)
		switch {
		case r.dep != "":
			return state{node: to, dep: r.dep}
		case r.wall && to != sink:
			return state{node: to, dep: s.dep, wall: true}
		}
		return state{node: to, dep: s.dep, wall: s.wall}
	}
	dist := make(map[state]int)
	var queue []state
	for _, r := range roots {
		s := enter(state{}, r)
		if _, seen := dist[s]; !seen {
			dist[s] = 1
			queue = append(queue, s)
		}
	}
	best := make(map[string]int)
	for len(queue) > 0 {
		s := queue[0]
		queue = queue[1:]
		if s.node == sink && s.dep != "" && !s.wall {
			if cur, ok := best[s.dep]; !ok || dist[s] < cur {
				best[s.dep] = dist[s]
			}
		}
		for _, e := range edges {
			if e[0] != s.node {
				continue
			}
			next := enter(s, e[1])
			if _, seen := dist[next]; !seen {
				dist[next] = dist[s] + 1
				queue = append(queue, next)
			}
		}
	}
	return best
}

func capValidChain(t *testing.T, g *capGraph, edges []capEdge, roots []loader.Sym, sink loader.Sym, role func(loader.Sym) capRole, b capBlame) {
	t.Helper()
	if len(b.chain) == 0 || b.chain[len(b.chain)-1] != sink {
		t.Fatalf("chain %v for %s does not end at sink %d", b.chain, b.dep, sink)
	}
	if !slices.Contains(roots, b.chain[0]) {
		t.Fatalf("chain %v for %s does not start at a root", b.chain, b.dep)
	}
	for i := 1; i < len(b.chain); i++ {
		if !slices.Contains(edges, capEdge{b.chain[i-1], b.chain[i]}) {
			t.Fatalf("chain %v for %s uses the missing edge %d -> %d", b.chain, b.dep, b.chain[i-1], b.chain[i])
		}
	}
	last := -1
	for i, s := range b.chain {
		if role(s).dep != "" {
			last = i
		}
	}
	if last < 0 || role(b.chain[last]).dep != b.dep {
		t.Fatalf("chain %v blames %q, want %q", b.chain, role(b.chain[max(last, 0)]).dep, b.dep)
	}
	for _, s := range b.chain[last+1:] {
		if role(s).wall && s != sink {
			t.Fatalf("chain %v crosses a wall after its blamed frame", b.chain)
		}
	}
}

func TestCapGraphChainsMatchOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for iter := range 3000 {
		n := 4 + rng.IntN(9)
		roles := make(map[loader.Sym]capRole)
		for s := 1; s <= n; s++ {
			switch rng.IntN(6) {
			case 0:
				roles[loader.Sym(s)] = capRole{dep: fmt.Sprintf("d%d", rng.IntN(3))}
			case 1:
				roles[loader.Sym(s)] = capRole{wall: true}
			}
		}
		var edges []capEdge
		for range n * (1 + rng.IntN(3)) {
			a, b := loader.Sym(1+rng.IntN(n)), loader.Sym(1+rng.IntN(n))
			if a != b {
				edges = append(edges, capEdge{a, b})
			}
		}
		roots := []loader.Sym{1}
		if rng.IntN(3) == 0 {
			roots = append(roots, loader.Sym(2+rng.IntN(n-1)))
		}
		sink := loader.Sym(1 + rng.IntN(n))
		role := capRoleTable(roles)
		g := buildCapGraph(roots, edges)
		want := capOracleChains(roots, edges, sink, role)
		got := make(map[string]int)
		for _, b := range g.blamedChains(sink, role) {
			capValidChain(t, g, edges, roots, sink, role, b)
			got[b.dep] = len(b.chain)
		}
		if len(got) != len(want) {
			t.Fatalf("iteration %d: blamed %v, oracle %v (roots %v edges %v sink %d roles %v)", iter, got, want, roots, edges, sink, roles)
		}
		for dep, n := range want {
			if got[dep] != n {
				t.Fatalf("iteration %d: chain length for %s = %d, oracle %d (roots %v edges %v sink %d roles %v)", iter, dep, got[dep], n, roots, edges, sink, roles)
			}
		}
	}
}
