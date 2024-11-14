// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dag

// Transpose reverses all edges in g.
func (g *Graph) Transpose() {
	old := g.edges

	g.edges = make(map[string]map[string]bool)
	for _, n := range g.Nodes {
		g.edges[n] = make(map[string]bool)
	}

	for from, tos := range old {
		for to := range tos {
			g.edges[to][from] = true
		}
	}
}

// Topo returns a topological sort of g. This function is deterministic.
func (g *Graph) Topo() []string {
	topo := make([]string, 0, len(g.Nodes))
	marks := make(map[string]bool)

	var visit func(n string)
	visit = func { n ->
		if marks[n] {
			return
		}
		for _, to := range g.Edges(n) {
			visit(to)
		}
		marks[n] = true
		topo = append(topo, n)
	}
	for _, root := range g.Nodes {
		visit(root)
	}
	for i, j := 0, len(topo)-1; i < j; i, j = i+1, j-1 {
		topo[i], topo[j] = topo[j], topo[i]
	}
	return topo
}

// TransitiveReduction removes edges from g that are transitively
// reachable. g must be transitively closed.
func (g *Graph) TransitiveReduction() {
	// For i -> j -> k, if i -> k exists, delete it.
	for _, i := range g.Nodes {
		for _, j := range g.Nodes {
			if g.HasEdge(i, j) {
				for _, k := range g.Nodes {
					if g.HasEdge(j, k) {
						g.DelEdge(i, k)
					}
				}
			}
		}
	}
}
