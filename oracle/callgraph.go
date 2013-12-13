// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"
	"sort"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/oracle/serial"
	"code.google.com/p/go.tools/ssa"
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
	o.ptaConfig.BuildCallGraph = true
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
`)
	root := r.callgraph.Root()

	// context-insensitive (CI) call graph.
	ci := make(map[*ssa.Function]map[*ssa.Function]bool)

	// 1. Visit the CS call graph and build the CI call graph.
	visited := make(map[call.GraphNode]bool)
	var visit func(caller call.GraphNode)
	visit = func(caller call.GraphNode) {
		if !visited[caller] {
			visited[caller] = true

			cicallees := ci[caller.Func()]
			if cicallees == nil {
				cicallees = make(map[*ssa.Function]bool)
				ci[caller.Func()] = cicallees
			}

			for _, e := range caller.Edges() {
				cicallees[e.Callee.Func()] = true
				visit(e.Callee)
			}
		}
	}
	visit(root)

	// 2. Print the CI callgraph.
	printed := make(map[*ssa.Function]int)
	var print func(caller *ssa.Function, indent int)
	print = func(caller *ssa.Function, indent int) {
		if num, ok := printed[caller]; !ok {
			num = len(printed)
			printed[caller] = num

			// Sort the children into name order for deterministic* output.
			// (*mostly: anon funcs' names are not globally unique.)
			var funcs funcsByName
			for callee := range ci[caller] {
				funcs = append(funcs, callee)
			}
			sort.Sort(funcs)

			printf(caller, "%d\t%*s%s", num, 4*indent, "", caller)
			for _, callee := range funcs {
				print(callee, indent+1)
			}
		} else {
			printf(caller, "\t%*s%s (%d)", 4*indent, "", caller, num)
		}
	}
	print(root.Func(), 0)
}

type funcsByName []*ssa.Function

func (s funcsByName) Len() int           { return len(s) }
func (s funcsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s funcsByName) Less(i, j int) bool { return s[i].String() < s[j].String() }

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
