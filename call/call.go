// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Package call defines the call graph abstraction and various algorithms
and utilities to operate on it.  It does not provide a concrete
implementation but permits other analyses (such as pointer analyses or
Rapid Type Analysis) to expose their own call graphs in a
representation-independent manner.

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

A call graph is called CONTEXT INSENSITIVE if no two nodes in N
represent the same syntactic function declaration, i.e. the set of
nodes and the set of syntactic functions are in one-to-one
correspondence.

A context-sensitive call graph may have multiple nodes corresponding
to the same function; this may yield a more precise approximation to
the calling behavior of the program.  Consider this program:

    func Apply(fn func(V), value V) { fn(value) }
    Apply(F, v1)
    ...
    Apply(G, v2)

A context-insensitive call graph would represent all calls to Apply by
the same node, so that node would have successors F and G.  A
context-sensitive call graph might represent the first and second
calls to Apply by distinct nodes, so that the first would have
successor F and the second would have successor G.  This is a more
precise representation of the possible behaviors of the program.

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
package call

import "code.google.com/p/go.tools/ssa"

// A Graph represents a call graph.
//
// A graph may contain nodes that are not reachable from the root.
// If the call graph is sound, such nodes indicate unreachable
// functions.
//
type Graph interface {
	Root() GraphNode    // the distinguished root node
	Nodes() []GraphNode // new unordered set of all nodes
}

// A GraphNode represents a node in a call graph.
//
// If the call graph is context sensitive, there may be multiple
// GraphNodes with the same Func(); the identity of the graph node
// indicates the context.
//
// Sites returns the set of syntactic call sites within this function.
//
// For nodes representing synthetic or intrinsic functions
// (e.g. reflect.Call, or the root of the call graph), Sites() returns
// a slice containing a single nil value to indicate the synthetic
// call site, and each edge in Edges() has a nil Site.
//
// All nodes "belong" to a single graph and must not be mixed with
// nodes belonging to another graph.
//
// A site may appear in Sites() but not in {e.Site | e ∈ Edges()}.
// This indicates that that caller node was unreachable, or that the
// call was dynamic yet no func or interface values flow to the call
// site.
//
// Clients should not mutate the node via the results of its methods.
//
type GraphNode interface {
	Func() *ssa.Function          // the function this node represents
	Sites() []ssa.CallInstruction // new unordered set of call sites within this function
	Edges() []Edge                // new unordered set of outgoing edges
}

// A Edge represents an edge in the call graph.
type Edge struct {
	Caller GraphNode
	Site   ssa.CallInstruction
	Callee GraphNode
}
