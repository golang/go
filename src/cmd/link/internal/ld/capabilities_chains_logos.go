// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"maps"
	"slices"
)

type capGraph struct {
	parent map[loader.Sym]loader.Sym
	preds  map[loader.Sym][]loader.Sym
}

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

func (g *capGraph) chainTo(s loader.Sym) []loader.Sym {
	var rev []loader.Sym
	for cur := s; cur != 0; cur = g.parent[cur] {
		rev = append(rev, cur)
	}
	slices.Reverse(rev)
	return rev
}

func (g *capGraph) depth(s loader.Sym) int {
	n := 0
	for cur := s; cur != 0; cur = g.parent[cur] {
		n++
	}
	return n
}

type capRole struct {
	dep  string
	wall bool
}

type capBlame struct {
	dep   string
	chain []loader.Sym
}

type capTail struct {
	next loader.Sym
	dist int
}

type capCandidate struct {
	frame loader.Sym
	total int
}

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

func (g *capGraph) consider(best map[string]capCandidate, dep string, frame loader.Sym, tailDist int) {
	total := g.depth(frame) + tailDist
	if cur, ok := best[dep]; ok && (cur.total < total || cur.total == total && cur.frame < frame) {
		return
	}
	best[dep] = capCandidate{frame: frame, total: total}
}

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
