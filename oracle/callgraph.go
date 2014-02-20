// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/callgraph"
	"code.google.com/p/go.tools/go/ssa"
	"code.google.com/p/go.tools/oracle/serial"
)

// doCallgraph displays the entire callgraph of the current program.
//
// TODO(adonovan): add options for restricting the display to a region
// of interest: function, package, subgraph, dirtree, goroutine, etc.
//
// TODO(adonovan): add an option to partition edges by call site.
//
// TODO(adonovan): elide nodes for synthetic functions?
//
func doCallgraph(o *Oracle, _ *QueryPos) (queryResult, error) {
	buildSSA(o)

	// Run the pointer analysis and build the complete callgraph.
	o.ptaConfig.BuildCallGraph = true
	ptares := ptrAnalysis(o)

	return &callgraphResult{
		callgraph: ptares.CallGraph,
	}, nil
}

type callgraphResult struct {
	callgraph *callgraph.Graph
}

func (r *callgraphResult) display(printf printfFunc) {
	printf(nil, `
Below is a call graph of the entire program.
The numbered nodes form a spanning tree.
Non-numbered nodes indicate back- or cross-edges to the node whose
 number follows in parentheses.
`)

	printed := make(map[*callgraph.Node]int)
	var print func(caller *callgraph.Node, indent int)
	print = func(caller *callgraph.Node, indent int) {
		if num, ok := printed[caller]; !ok {
			num = len(printed)
			printed[caller] = num

			// Sort the children into name order for deterministic* output.
			// (*mostly: anon funcs' names are not globally unique.)
			var funcs funcsByName
			for callee := range callgraph.CalleesOf(caller) {
				funcs = append(funcs, callee.Func)
			}
			sort.Sort(funcs)

			printf(caller.Func, "%d\t%*s%s", num, 4*indent, "", caller.Func)
			for _, callee := range funcs {
				print(r.callgraph.Nodes[callee], indent+1)
			}
		} else {
			printf(caller, "\t%*s%s (%d)", 4*indent, "", caller.Func, num)
		}
	}
	print(r.callgraph.Root, 0)
}

type funcsByName []*ssa.Function

func (s funcsByName) Len() int           { return len(s) }
func (s funcsByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s funcsByName) Less(i, j int) bool { return s[i].String() < s[j].String() }

func (r *callgraphResult) toSerial(res *serial.Result, fset *token.FileSet) {
	cg := make([]serial.CallGraph, len(r.callgraph.Nodes))
	for _, n := range r.callgraph.Nodes {
		j := &cg[n.ID]
		fn := n.Func
		j.Name = fn.String()
		j.Pos = fset.Position(fn.Pos()).String()
		for callee := range callgraph.CalleesOf(n) {
			j.Children = append(j.Children, callee.ID)
		}
		sort.Ints(j.Children)
	}
	res.Callgraph = cg
}
