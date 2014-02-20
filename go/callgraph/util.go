// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package callgraph

// This file provides various utilities over call graphs, such as
// visitation and path search.

// CalleesOf returns a new set containing all direct callees of the
// caller node.
//
func CalleesOf(caller *Node) map[*Node]bool {
	callees := make(map[*Node]bool)
	for _, e := range caller.Out {
		callees[e.Callee] = true
	}
	return callees
}

// GraphVisitEdges visits all the edges in graph g in depth-first order.
// The edge function is called for each edge in postorder.  If it
// returns non-nil, visitation stops and GraphVisitEdges returns that
// value.
//
func GraphVisitEdges(g *Graph, edge func(*Edge) error) error {
	seen := make(map[*Node]bool)
	var visit func(n *Node) error
	visit = func(n *Node) error {
		if !seen[n] {
			seen[n] = true
			for _, e := range n.Out {
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
	for _, n := range g.Nodes {
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
func PathSearch(start *Node, isEnd func(*Node) bool) []*Edge {
	stack := make([]*Edge, 0, 32)
	seen := make(map[*Node]bool)
	var search func(n *Node) []*Edge
	search = func(n *Node) []*Edge {
		if !seen[n] {
			seen[n] = true
			if isEnd(n) {
				return stack
			}
			for _, e := range n.Out {
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
