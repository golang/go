package oracle

import (
	"strings"

	"code.google.com/p/go.tools/pointer"
)

// callgraph displays the entire callgraph of the current program.
//
// Nodes may be seem to appear multiple times due to (limited)
// context sensitivity.
//
// TODO(adonovan): add options for restricting the display to a region
// of interest: function, package, subgraph, dirtree, etc.
//
func callgraph(o *oracle) (queryResult, error) {
	buildSSA(o)

	// Run the pointer analysis and build the complete callgraph.
	callgraph := make(pointer.CallGraph)
	o.config.Call = callgraph.AddEdge
	root := ptrAnalysis(o)

	return &callgraphResult{
		root:      root,
		callgraph: callgraph,
	}, nil
}

type callgraphResult struct {
	root      pointer.CallGraphNode
	callgraph pointer.CallGraph

	numbering map[pointer.CallGraphNode]int // used by display
}

func (r *callgraphResult) print(o *oracle, cgn pointer.CallGraphNode, indent int) {
	if n := r.numbering[cgn]; n == 0 {
		n = 1 + len(r.numbering)
		r.numbering[cgn] = n
		o.printf(cgn.Func(), "%d\t%s%s", n, strings.Repeat("    ", indent), cgn.Func())
		for callee := range r.callgraph[cgn] {
			r.print(o, callee, indent+1)
		}
	} else {
		o.printf(cgn.Func(), "\t%s%s (%d)", strings.Repeat("    ", indent), cgn.Func(), n)
	}
}

func (r *callgraphResult) display(o *oracle) {
	o.printf(nil, `
Below is a call graph of the entire program.
The numbered nodes form a spanning tree.
Non-numbered nodes indicate back- or cross-edges to the node whose
 number follows in parentheses.
Some nodes may appear multiple times due to context-sensitive
 treatment of some calls.
`)

	r.numbering = make(map[pointer.CallGraphNode]int)
	r.print(o, r.root, 0)
}
