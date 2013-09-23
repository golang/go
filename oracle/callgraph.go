// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"
	"strings"

	"code.google.com/p/go.tools/oracle/json"
	"code.google.com/p/go.tools/pointer"
)

// callgraph displays the entire callgraph of the current program.
//
// Nodes may be seem to appear multiple times due to (limited)
// context sensitivity.
//
// TODO(adonovan): add options for restricting the display to a region
// of interest: function, package, subgraph, dirtree, goroutine, etc.
//
// TODO(adonovan): add an option to project away context sensitivity.
// The callgraph API should provide this feature.
//
// TODO(adonovan): elide nodes for synthetic functions?
//
func callgraph(o *Oracle, _ *QueryPos) (queryResult, error) {
	buildSSA(o)

	// Run the pointer analysis and build the complete callgraph.
	callgraph := make(pointer.CallGraph)
	o.config.Call = callgraph.AddEdge
	root := ptrAnalysis(o)

	// Assign (preorder) numbers to all the callgraph nodes.
	// TODO(adonovan): the callgraph API should do this for us.
	// (Actually, it does have unique numbers under the hood.)
	numbering := make(map[pointer.CallGraphNode]int)
	var number func(cgn pointer.CallGraphNode)
	number = func(cgn pointer.CallGraphNode) {
		if _, ok := numbering[cgn]; !ok {
			numbering[cgn] = len(numbering)
			for callee := range callgraph[cgn] {
				number(callee)
			}
		}
	}
	number(root)

	return &callgraphResult{
		root:      root,
		callgraph: callgraph,
		numbering: numbering,
	}, nil
}

type callgraphResult struct {
	root      pointer.CallGraphNode
	callgraph pointer.CallGraph
	numbering map[pointer.CallGraphNode]int
}

func (r *callgraphResult) display(printf printfFunc) {
	printf(nil, `
Below is a call graph of the entire program.
The numbered nodes form a spanning tree.
Non-numbered nodes indicate back- or cross-edges to the node whose
 number follows in parentheses.
Some nodes may appear multiple times due to context-sensitive
 treatment of some calls.
`)

	// TODO(adonovan): compute the numbers as we print; right now
	// it depends on map iteration so it's arbitrary,which is ugly.
	seen := make(map[pointer.CallGraphNode]bool)
	var print func(cgn pointer.CallGraphNode, indent int)
	print = func(cgn pointer.CallGraphNode, indent int) {
		n := r.numbering[cgn]
		if !seen[cgn] {
			seen[cgn] = true
			printf(cgn.Func(), "%d\t%s%s", n, strings.Repeat("    ", indent), cgn.Func())
			for callee := range r.callgraph[cgn] {
				print(callee, indent+1)
			}
		} else {
			printf(cgn.Func(), "\t%s%s (%d)", strings.Repeat("    ", indent), cgn.Func(), n)
		}
	}
	print(r.root, 0)
}

func (r *callgraphResult) toJSON(res *json.Result, fset *token.FileSet) {
	cg := make([]json.CallGraph, len(r.numbering))
	for n, i := range r.numbering {
		j := &cg[i]
		fn := n.Func()
		j.Name = fn.String()
		j.Pos = fset.Position(fn.Pos()).String()
		for callee := range r.callgraph[n] {
			j.Children = append(j.Children, r.numbering[callee])
		}
	}
	res.Callgraph = cg
}
