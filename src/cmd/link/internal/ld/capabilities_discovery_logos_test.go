// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"context"
	"internal/testenv"
	"maps"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"
)

type capVisit struct{ node, parent loader.Sym }

const capDiscoveryHelper = "GO_LOGOS_CAP_DISCOVERY_HELPER"

// A relocation can name an inline owner before that owner's root path has
// been recorded. Discovery order must not become the ancestry of a finding.
func TestCapGraphDiscoveryOrder(t *testing.T) {
	dep := map[loader.Sym]capRole{2: {dep: "dep"}, 4: {dep: "other"}}
	tests := []struct {
		name   string
		visits []capVisit
		sink   loader.Sym
		roles  map[loader.Sym]capRole
	}{
		{"rooted_cycle", []capVisit{{1, 0}, {2, 3}, {3, 2}, {2, 1}, {3, 1}, {5, 2}}, 5, dep},
		{"disconnected_cycle", []capVisit{{1, 0}, {2, 3}, {3, 2}, {5, 2}}, 5, dep},
		{"unrooted_incoming_cycle", []capVisit{{1, 0}, {2, 3}, {3, 2}, {5, 1}, {5, 2}}, 5, dep},
		{"longer_cycle", []capVisit{{1, 0}, {2, 4}, {3, 2}, {4, 3}, {2, 1}, {5, 3}, {5, 4}}, 5, dep},
		{"late_shortcut", []capVisit{{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}, {4, 1}}, 5, dep},
		{"late_root", []capVisit{{1, 0}, {2, 1}, {5, 2}, {2, 0}}, 5, dep},
		{"multiple_roots", []capVisit{{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}, {6, 0}, {4, 6}}, 5, dep},
		{"unknown_parent", []capVisit{{1, 0}, {2, 9}, {5, 2}}, 5, dep},
		{"late_parent_control", []capVisit{{1, 0}, {3, 2}, {2, 1}, {5, 3}}, 5, dep},
		{"duplicate_edges_control", []capVisit{{1, 0}, {1, 0}, {2, 1}, {2, 1}, {5, 2}, {5, 2}}, 5, dep},
		{"capacity_growth_control", []capVisit{{1, 0}, {10, 11}, {11, 1}, {12, 10}}, 12, map[loader.Sym]capRole{10: {dep: "dep"}}},
		{"wall_before_control", []capVisit{{1, 0}, {3, 1}, {2, 3}, {5, 2}}, 5, map[loader.Sym]capRole{2: {dep: "dep"}, 3: {wall: true}}},
		{"wall_after_control", []capVisit{{1, 0}, {2, 1}, {3, 2}, {5, 3}}, 5, map[loader.Sym]capRole{2: {dep: "dep"}, 3: {wall: true}}},
		{"empty_graph_control", nil, 5, dep},
		{"root_sink_control", []capVisit{{2, 0}}, 2, dep},
		{"exempt_cycle_control", []capVisit{{1, 0}, {2, 3}, {3, 2}, {2, 1}, {5, 2}}, 5, nil},
	}
	if mode, ok := os.LookupEnv(capDiscoveryHelper); ok {
		i, err := strconv.Atoi(mode)
		if err != nil || i < 0 || i >= len(tests) {
			t.Fatalf("invalid discovery helper case %q", mode)
		}
		tc := tests[i]
		checkCapDiscovery(t, tc.visits, tc.sink, tc.roles)
		return
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runCapDiscoveryChild(t, "TestCapGraphDiscoveryOrder", strconv.Itoa(i))
		})
	}
}

func TestCapGraphShuffledDiscoveryOracle(t *testing.T) {
	if mode := os.Getenv(capDiscoveryHelper); mode != "oracle" {
		runCapDiscoveryChild(t, "TestCapGraphShuffledDiscoveryOracle", "oracle")
		return
	}
	rng := rand.New(rand.NewPCG(31, 47))
	for iter := range 1000 {
		n := 3 + rng.IntN(8)
		roles := make(map[loader.Sym]capRole)
		for s := 1; s <= n; s++ {
			switch rng.IntN(5) {
			case 0, 1:
				roles[loader.Sym(s)] = capRole{dep: "d" + strconv.Itoa(rng.IntN(3))}
			case 2:
				roles[loader.Sym(s)] = capRole{wall: true}
			}
		}
		visits := []capVisit{{1, 0}}
		if rng.IntN(3) == 0 {
			visits = append(visits, capVisit{loader.Sym(2 + rng.IntN(n-1)), 0})
		}
		for range 2 * n {
			visits = append(visits, capVisit{loader.Sym(1 + rng.IntN(n)), loader.Sym(1 + rng.IntN(n))})
		}
		rng.Shuffle(len(visits), func(i, j int) { visits[i], visits[j] = visits[j], visits[i] })
		t.Run(strconv.Itoa(iter), func(t *testing.T) {
			checkCapDiscovery(t, visits, loader.Sym(1+rng.IntN(n)), roles)
		})
	}
}

// An isolated process makes a cyclic parent walk a bounded test failure.
// CommandContext also checks that each process is reaped before cleanup.
func runCapDiscoveryChild(t *testing.T, name, mode string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	cmd := testenv.CommandContext(t, ctx, testenv.Executable(t), "-test.run=^"+name+"$")
	cmd.Env = append(cmd.Environ(), capDiscoveryHelper+"="+mode)
	output, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("capability ancestry traversal did not terminate: %v\n%s", ctx.Err(), output)
	}
	if err != nil {
		t.Fatalf("discovery-order check: %v\n%s", err, output)
	}
}

func checkCapDiscovery(t *testing.T, visits []capVisit, sink loader.Sym, roles map[loader.Sym]capRole) {
	t.Helper()
	g := newCapGraph(1)
	var roots []loader.Sym
	var edges []capEdge
	role := capRoleTable(roles)
	check := func(step int) {
		t.Helper()
		want := capOracleChains(roots, edges, sink, role)
		got := make(map[string]int)
		for _, b := range g.blamedChains(sink, role) {
			capValidChain(t, g, edges, roots, sink, role, b)
			if _, duplicate := got[b.dep]; duplicate {
				t.Fatalf("duplicate dependency %q at step %d", b.dep, step)
			}
			got[b.dep] = len(b.chain)
		}
		if !maps.Equal(got, want) {
			t.Fatalf("step %d: chain lengths %v, oracle %v; roots %v, edges %v, sink %d, roles %v", step, got, want, roots, edges, sink, roles)
		}
		for _, e := range edges {
			if !slices.Contains(g.preds[e[1]], e[0]) {
				t.Fatalf("lost predecessor edge %v at step %d", e, step)
			}
		}
	}
	check(-1)
	for i, v := range visits {
		g.visit(v.node, v.parent)
		if v.parent == 0 {
			roots = append(roots, v.node)
		} else {
			edges = append(edges, capEdge{v.parent, v.node})
		}
		check(i)
	}
}
