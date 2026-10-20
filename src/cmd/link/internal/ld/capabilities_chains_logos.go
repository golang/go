// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"maps"
	"slices"
)

// capGraph is the reachability graph the capabilities audit records while
// it floods from the link roots: the shortest-path parent of every visited
// symbol (roots carry parent 0) and every distinct reverse edge. The chain
// search walks the reverse edges outward from each sink.
type capGraph struct {
	parent map[loader.Sym]loader.Sym
	preds  map[loader.Sym][]loader.Sym
}

// newCapGraph returns an empty graph.
func newCapGraph() *capGraph {
	return &capGraph{
		parent: make(map[loader.Sym]loader.Sym),
		preds:  make(map[loader.Sym][]loader.Sym),
	}
}

// visit records the edge p -> s and reports whether s is new. Roots pass
// p == 0. A node's edges arrive in one run, so comparing against the last
// recorded predecessor drops every duplicate edge.
func (g *capGraph) visit(s, p loader.Sym) bool {
	if ps := g.preds[s]; p != 0 && (len(ps) == 0 || ps[len(ps)-1] != p) {
		g.preds[s] = append(ps, p)
	}
	if _, ok := g.parent[s]; ok {
		return false
	}
	g.parent[s] = p
	return true
}

// chainTo rebuilds the root-to-s chain through the parent map. A root's
// parent, and the parent of an unvisited symbol, is 0, which ends the walk.
func (g *capGraph) chainTo(s loader.Sym) []loader.Sym {
	var rev []loader.Sym
	for cur := s; cur != 0; cur = g.parent[cur] {
		rev = append(rev, cur)
	}
	slices.Reverse(rev)
	return rev
}

// depth returns the number of symbols on the root-to-s chain.
func (g *capGraph) depth(s loader.Sym) int {
	n := 0
	for cur := s; cur != 0; cur = g.parent[cur] {
		n++
	}
	return n
}

// capRole tells the chain search how to treat a symbol: dep names the
// package blamed for it ("" when the symbol is exempt), and wall marks
// symbols a tail never crosses.
type capRole struct {
	dep  string
	wall bool
}

// capBlame is one dependency blamed for reaching a sink, with the chain
// that shows it.
type capBlame struct {
	dep   string
	chain []loader.Sym
}

// capTail is a symbol's successor on its shortest exempt-only path to the
// sink, and that path's length in edges.
type capTail struct {
	next loader.Sym
	dist int
}

// capCandidate is the best blamed frame found so far for one dependency
// and the length of the chain through it.
type capCandidate struct {
	frame loader.Sym
	total int
}

// blamedChains returns one shortest chain for every dependency that
// reaches sink, sorted by dependency path. role classifies each symbol as
// blamed on a dependency, exempt (standard library, first-party, data), or
// a wall.
//
// The search walks reverse edges outward from the sink and stops at the
// first blamed symbol on each path, so a chain blames its frame nearest
// the sink that role does not exempt and every frame after the blamed one
// is exempt. A chain made only of exempt frames blames nobody, a
// dependency's chain is reported even when a shorter exempt-only chain
// reaches the sink first, and a sink a dependency owns blames that
// dependency with a zero-length tail. Walls keep implicit edges out of
// tails: every function reaches the runtime's stack growth and crash
// paths, so a tail through them would blame every dependency for
// whatever the runtime can reach. The sink itself may sit behind a wall.
func (g *capGraph) blamedChains(sink loader.Sym, role func(loader.Sym) capRole) []capBlame {
	if _, reached := g.parent[sink]; !reached {
		return nil
	}
	tails := map[loader.Sym]capTail{sink: {}}
	best := make(map[string]capCandidate)
	queue := []loader.Sym{sink}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		if dep := role(u).dep; dep != "" {
			g.consider(best, dep, u, tails[u].dist)
			continue
		}
		for _, p := range g.preds[u] {
			if _, seen := tails[p]; !seen && !role(p).wall {
				tails[p] = capTail{next: u, dist: tails[u].dist + 1}
				queue = append(queue, p)
			}
		}
	}
	return g.assemble(best, tails)
}

// consider keeps frame as dep's best blamed frame when the chain through it
// is shorter than the kept one; equal lengths keep the lower symbol so the
// choice is stable.
func (g *capGraph) consider(best map[string]capCandidate, dep string, frame loader.Sym, tailDist int) {
	total := g.depth(frame) + tailDist
	if cur, ok := best[dep]; ok && (cur.total < total || cur.total == total && cur.frame < frame) {
		return
	}
	best[dep] = capCandidate{frame: frame, total: total}
}

// assemble joins each dependency's root-to-frame chain with the frame's
// exempt tail to the sink.
func (g *capGraph) assemble(best map[string]capCandidate, tails map[loader.Sym]capTail) []capBlame {
	out := make([]capBlame, 0, len(best))
	for _, dep := range slices.Sorted(maps.Keys(best)) {
		frame := best[dep].frame
		chain := g.chainTo(frame)
		for cur := tails[frame].next; cur != 0; cur = tails[cur].next {
			chain = append(chain, cur)
		}
		out = append(out, capBlame{dep: dep, chain: chain})
	}
	return out
}
