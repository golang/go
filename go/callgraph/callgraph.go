// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Package callgraph defines the call graph and various algorithms
and utilities to operate on it.

A call graph is a labelled directed graph whose nodes represent
functions and whose edge labels represent syntactic function call
sites.  The presence of a labelled edge (caller, site, callee)
indicates that caller may call callee at the specified call site.

A call graph is a multigraph: it may contain multiple edges (caller,
*, callee) connecting the same pair of nodes, so long as the edges
differ by label; this occurs when one function calls another function
from multiple call sites.  Also, it may contain multiple edges
(caller, site, *) that differ only by callee; this indicates a
polymorphic call.

A SOUND call graph is one that overapproximates the dynamic calling
behaviors of the program in all possible executions.  One call graph
is more PRECISE than another if it is a smaller overapproximation of
the dynamic behavior.

All call graphs have a synthetic root node which is responsible for
calling main() and init().

Calls to built-in functions (e.g. panic, println) are not represented
in the call graph; they are treated like built-in operators of the
language.

*/
package callgraph // import "golang.org/x/tools/go/callgraph"

// TODO(adonovan): add a function to eliminate wrappers from the
// callgraph, preserving topology.
// More generally, we could eliminate "uninteresting" nodes such as
// nodes from packages we don't care about.

import (
	"fmt"
	"go/token"

	"golang.org/x/tools/go/ssa"
)

// A Graph represents a call graph.
//
// A graph may contain nodes that are not reachable from the root.
// If the call graph is sound, such nodes indicate unreachable
// functions.
//
type Graph struct {
	Root  *Node                   // the distinguished root node
	Nodes map[*ssa.Function]*Node // all nodes by function
}

// New returns a new Graph with the specified root node.
func New(root *ssa.Function) *Graph {
	g := &Graph{Nodes: make(map[*ssa.Function]*Node)}
	g.Root = g.CreateNode(root)
	return g
}

// CreateNode returns the Node for fn, creating it if not present.
func (g *Graph) CreateNode(fn *ssa.Function) *Node {
	n, ok := g.Nodes[fn]
	if !ok {
		n = &Node{Func: fn, ID: len(g.Nodes)}
		g.Nodes[fn] = n
	}
	return n
}

// A Node represents a node in a call graph.
type Node struct {
	Func *ssa.Function // the function this node represents
	ID   int           // 0-based sequence number
	In   []*Edge       // unordered set of incoming call edges (n.In[*].Callee == n)
	Out  []*Edge       // unordered set of outgoing call edges (n.Out[*].Caller == n)
}

func (n *Node) String() string {
	return fmt.Sprintf("n%d:%s", n.ID, n.Func)
}

// A Edge represents an edge in the call graph.
//
// Site is nil for edges originating in synthetic or intrinsic
// functions, e.g. reflect.Value.Call or the root of the call graph.
type Edge struct {
	Caller *Node
	Site   ssa.CallInstruction
	Callee *Node
}

func (e Edge) String() string {
	return fmt.Sprintf("%s --> %s", e.Caller, e.Callee)
}

func (e Edge) Description() string {
	var prefix string
	switch e.Site.(type) {
	case nil:
		return "synthetic call"
	case *ssa.Go:
		prefix = "concurrent "
	case *ssa.Defer:
		prefix = "deferred "
	}
	return prefix + e.Site.Common().Description()
}

func (e Edge) Pos() token.Pos {
	if e.Site == nil {
		return token.NoPos
	}
	return e.Site.Pos()
}

// AddEdge adds the edge (caller, site, callee) to the call graph.
// Elimination of duplicate edges is the caller's responsibility.
func AddEdge(caller *Node, site ssa.CallInstruction, callee *Node) {
	e := &Edge{caller, site, callee}
	callee.In = append(callee.In, e)
	caller.Out = append(caller.Out, e)
}
