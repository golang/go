// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/ssa"
)

// TODO(adonovan): move the CallGraph, CallGraphNode, CallSite types
// into a separate package 'callgraph', and make them pure interfaces
// capable of supporting several implementations (context-sensitive
// and insensitive PTA, RTA, etc).

// ---------- CallGraphNode ----------

// A CallGraphNode is a context-sensitive representation of a node in
// the callgraph.  In other words, there may be multiple nodes
// representing a single *Function, depending on the contexts in which
// it is called.  The identity of the node is therefore important.
//
type CallGraphNode interface {
	Func() *ssa.Function // the function this node represents
	String() string      // diagnostic description of this callgraph node
}

type cgnode struct {
	fn  *ssa.Function
	obj nodeid // start of this contour's object block
}

func (n *cgnode) Func() *ssa.Function {
	return n.fn
}

func (n *cgnode) String() string {
	return fmt.Sprintf("cg%d:%s", n.obj, n.fn)
}

// ---------- CallSite ----------

// A CallSite is a context-sensitive representation of a function call
// site in the program.
//
type CallSite interface {
	Caller() CallGraphNode // the enclosing context of this call
	Pos() token.Pos        // source position; token.NoPos for synthetic calls
	Description() string   // UI description of call kind; see (*ssa.CallCommon).Description
	String() string        // diagnostic description of this callsite
}

// A callsite represents a single function or method callsite within a
// function.  callsites never represent calls to built-ins; they are
// handled as intrinsics.
//
type callsite struct {
	caller  *cgnode             // the origin of the call
	targets nodeid              // pts(targets) contains identities of all called functions.
	instr   ssa.CallInstruction // optional call instruction; provides IsInvoke, position, etc.
	pos     token.Pos           // position, if instr == nil, i.e. synthetic callsites.
}

// Caller returns the node in the callgraph from which this call originated.
func (c *callsite) Caller() CallGraphNode {
	return c.caller
}

// Description returns a description of this kind of call, in the
// manner of ssa.CallCommon.Description().
//
func (c *callsite) Description() string {
	if c.instr != nil {
		return c.instr.Common().Description()
	}
	return "synthetic function call"
}

// Pos returns the source position of this callsite, or token.NoPos if implicit.
func (c *callsite) Pos() token.Pos {
	if c.instr != nil {
		return c.instr.Pos()
	}
	return c.pos
}

func (c *callsite) String() string {
	// TODO(adonovan): provide more info, e.g. target of static
	// call, arguments, location.
	return c.Description()
}

// ---------- CallGraph ----------

// CallGraph is a forward directed graph of functions labelled by an
// arbitrary site within the caller.
//
// CallGraph.AddEdge may be used as the Context.Call callback for
// clients that wish to construct a call graph.
//
// TODO(adonovan): this is just a starting point.  Add options to
// control whether we record no callsite, an arbitrary callsite, or
// all callsites for a given graph edge.  Also, this could live in
// another package since it's just a client utility.
//
type CallGraph map[CallGraphNode]map[CallGraphNode]CallSite

func (cg CallGraph) AddEdge(site CallSite, callee CallGraphNode) {
	caller := site.Caller()
	callees := cg[caller]
	if callees == nil {
		callees = make(map[CallGraphNode]CallSite)
		cg[caller] = callees
	}
	callees[callee] = site // save an arbitrary site
}
