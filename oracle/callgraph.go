// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"
	"strings"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/oracle/serial"
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
// TODO(adonovan): add an option to partition edges by call site.
//
// TODO(adonovan): elide nodes for synthetic functions?
//
func callgraph(o *Oracle, _ *QueryPos) (queryResult, error) {
	buildSSA(o)

	// Run the pointer analysis and build the complete callgraph.
	o.config.BuildCallGraph = true
	ptares := ptrAnalysis(o)

	return &callgraphResult{
		callgraph: ptares.CallGraph,
	}, nil
}

type callgraphResult struct {
	callgraph call.Graph
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

	seen := make(map[call.GraphNode]int)
	var print func(cgn call.GraphNode, indent int)
	print = func(cgn call.GraphNode, indent int) {
		fn := cgn.Func()
		if num, ok := seen[cgn]; !ok {
			num = len(seen)
			seen[cgn] = num
			printf(fn, "%d\t%s%s", num, strings.Repeat("    ", indent), fn)
			// Don't use Edges(), which distinguishes callees by call site.
			for callee := range call.CalleesOf(cgn) {
				print(callee, indent+1)
			}
		} else {
			printf(fn, "\t%s%s (%d)", strings.Repeat("    ", indent), fn, num)
		}
	}
	print(r.callgraph.Root(), 0)
}

func (r *callgraphResult) toSerial(res *serial.Result, fset *token.FileSet) {
	nodes := r.callgraph.Nodes()

	numbering := make(map[call.GraphNode]int)
	for i, n := range nodes {
		numbering[n] = i
	}

	cg := make([]serial.CallGraph, len(nodes))
	for i, n := range nodes {
		j := &cg[i]
		fn := n.Func()
		j.Name = fn.String()
		j.Pos = fset.Position(fn.Pos()).String()
		for callee := range call.CalleesOf(n) {
			j.Children = append(j.Children, numbering[callee])
		}
	}
	res.Callgraph = cg
}
