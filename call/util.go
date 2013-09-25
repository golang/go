// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package call

// This file provides various representation-independent utilities
// over call graphs, such as visitation and path search.
//
// TODO(adonovan):
//
// Consider adding lookup functions such as:
//   FindSitesByPos(g Graph, lparen token.Pos) []Site
//   FindSitesByCallExpr(g Graph, expr *ast.CallExpr) []Site
//   FindSitesByInstr(g Graph, instr ssa.CallInstruction) []Site
//   FindNodesByFunc(g Graph, fn *ssa.Function) []GraphNode
//   (Counterargument: they're all inefficient linear scans; if the
//   caller does it explicitly there may be opportunities to optimize.
//
// Add a utility function to eliminate all context from a call graph.

// CalleesOf returns a new set containing all direct callees of the
// caller node.
//
func CalleesOf(caller GraphNode) map[GraphNode]bool {
	callees := make(map[GraphNode]bool)
	for _, e := range caller.Edges() {
		callees[e.Callee] = true
	}
	return callees
}

// GraphVisitEdges visits all the edges in graph g in depth-first order.
// The edge function is called for each edge in postorder.  If it
// returns non-nil, visitation stops and GraphVisitEdges returns that
// value.
//
func GraphVisitEdges(g Graph, edge func(Edge) error) error {
	seen := make(map[GraphNode]bool)
	var visit func(n GraphNode) error
	visit = func(n GraphNode) error {
		if !seen[n] {
			seen[n] = true
			for _, e := range n.Edges() {
				if err := visit(e.Callee); err != nil {
					return err
				}
				if err := edge(e); err != nil {
					return err
				}
			}
		}
		return nil
	}
	for _, n := range g.Nodes() {
		if err := visit(n); err != nil {
			return err
		}
	}
	return nil
}

// PathSearch finds an arbitrary path starting at node start and
// ending at some node for which isEnd() returns true.  On success,
// PathSearch returns the path as an ordered list of edges; on
// failure, it returns nil.
//
func PathSearch(start GraphNode, isEnd func(GraphNode) bool) []Edge {
	stack := make([]Edge, 0, 32)
	seen := make(map[GraphNode]bool)
	var search func(n GraphNode) []Edge
	search = func(n GraphNode) []Edge {
		if !seen[n] {
			seen[n] = true
			if isEnd(n) {
				return stack
			}
			for _, e := range n.Edges() {
				stack = append(stack, e) // push
				if found := search(e.Callee); found != nil {
					return found
				}
				stack = stack[:len(stack)-1] // pop
			}
		}
		return nil
	}
	return search(start)
}
