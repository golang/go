// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package callgraph

import "golang.org/x/tools/go/ssa"

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

// DeleteSyntheticNodes removes from call graph g all nodes for
// synthetic functions (except g.Root and package initializers),
// preserving the topology.  In effect, calls to synthetic wrappers
// are "inlined".
//
func (g *Graph) DeleteSyntheticNodes() {
	// Measurements on the standard library and go.tools show that
	// resulting graph has ~15% fewer nodes and 4-8% fewer edges
	// than the input.
	//
	// Inlining a wrapper of in-degree m, out-degree n adds m*n
	// and removes m+n edges.  Since most wrappers are monomorphic
	// (n=1) this results in a slight reduction.  Polymorphic
	// wrappers (n>1), e.g. from embedding an interface value
	// inside a struct to satisfy some interface, cause an
	// increase in the graph, but they seem to be uncommon.

	// Hash all existing edges to avoid creating duplicates.
	edges := make(map[Edge]bool)
	for _, cgn := range g.Nodes {
		for _, e := range cgn.Out {
			edges[*e] = true
		}
	}
	for fn, cgn := range g.Nodes {
		if cgn == g.Root || fn.Synthetic == "" || isInit(cgn.Func) {
			continue // keep
		}
		for _, eIn := range cgn.In {
			for _, eOut := range cgn.Out {
				newEdge := Edge{eIn.Caller, eIn.Site, eOut.Callee}
				if edges[newEdge] {
					continue // don't add duplicate
				}
				AddEdge(eIn.Caller, eIn.Site, eOut.Callee)
				edges[newEdge] = true
			}
		}
		g.DeleteNode(cgn)
	}
}

func isInit(fn *ssa.Function) bool {
	return fn.Pkg != nil && fn.Pkg.Func("init") == fn
}

// DeleteNode removes node n and its edges from the graph g.
// (NB: not efficient for batch deletion.)
func (g *Graph) DeleteNode(n *Node) {
	n.deleteIns()
	n.deleteOuts()
	delete(g.Nodes, n.Func)
}

// deleteIns deletes all incoming edges to n.
func (n *Node) deleteIns() {
	for _, e := range n.In {
		removeOutEdge(e)
	}
	n.In = nil
}

// deleteOuts deletes all outgoing edges from n.
func (n *Node) deleteOuts() {
	for _, e := range n.Out {
		removeInEdge(e)
	}
	n.Out = nil
}

// removeOutEdge removes edge.Caller's outgoing edge 'edge'.
func removeOutEdge(edge *Edge) {
	caller := edge.Caller
	n := len(caller.Out)
	for i, e := range caller.Out {
		if e == edge {
			// Replace it with the final element and shrink the slice.
			caller.Out[i] = caller.Out[n-1]
			caller.Out[n-1] = nil // aid GC
			caller.Out = caller.Out[:n-1]
			return
		}
	}
	panic("edge not found: " + edge.String())
}

// removeInEdge removes edge.Callee's incoming edge 'edge'.
func removeInEdge(edge *Edge) {
	caller := edge.Callee
	n := len(caller.In)
	for i, e := range caller.In {
		if e == edge {
			// Replace it with the final element and shrink the slice.
			caller.In[i] = caller.In[n-1]
			caller.In[n-1] = nil // aid GC
			caller.In = caller.In[:n-1]
			return
		}
	}
	panic("edge not found: " + edge.String())
}
