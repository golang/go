// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file defines our implementation of the call.Graph API.

import (
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/ssa"
)

// cgraph implements call.Graph.
type cgraph struct {
	root  *cgnode
	nodes []*cgnode
}

func (g *cgraph) Nodes() []call.GraphNode {
	nodes := make([]call.GraphNode, len(g.nodes))
	for i, node := range g.nodes {
		nodes[i] = node
	}
	return nodes
}

func (g *cgraph) Root() call.GraphNode {
	return g.root
}

// cgnode implements call.GraphNode.
type cgnode struct {
	fn         *ssa.Function
	obj        nodeid      // start of this contour's object block
	sites      []*callsite // ordered list of callsites within this function
	callersite *callsite   // where called from, if known; nil for shared contours
}

func (n *cgnode) Func() *ssa.Function {
	return n.fn
}

func (n *cgnode) Sites() []ssa.CallInstruction {
	sites := make([]ssa.CallInstruction, len(n.sites))
	for i, site := range n.sites {
		sites[i] = site.instr
	}
	return sites
}

func (n *cgnode) Edges() []call.Edge {
	var numEdges int
	for _, site := range n.sites {
		numEdges += len(site.callees)
	}
	edges := make([]call.Edge, 0, numEdges)

	for _, site := range n.sites {
		for _, callee := range site.callees {
			edges = append(edges, call.Edge{Caller: n, Site: site.instr, Callee: callee})
		}
	}
	return edges
}

func (n *cgnode) String() string {
	return fmt.Sprintf("cg%d:%s", n.obj, n.fn)
}

// A callsite represents a single call site within a cgnode;
// it is implicitly context-sensitive.
// callsites never represent calls to built-ins;
// they are handled as intrinsics.
//
type callsite struct {
	targets nodeid              // pts(·) contains objects for dynamically called functions
	instr   ssa.CallInstruction // the call instruction; nil for synthetic/intrinsic
	callees []*cgnode           // unordered set of callees of this site
}

func (c *callsite) String() string {
	if c.instr != nil {
		return c.instr.Common().Description()
	}
	return "synthetic function call"
}

// pos returns the source position of this callsite, or token.NoPos if implicit.
func (c *callsite) pos() token.Pos {
	if c.instr != nil {
		return c.instr.Pos()
	}
	return token.NoPos
}
