// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WORK IN PROGRESS

package pgo

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"fmt"
	"internal/profile"
	"log"
	"os"
	"strconv"
	"strings"
)

// IRGraph is the key datastrcture that is built from profile. It is essentially a call graph with nodes pointing to IRs of functions and edges carrying weights and callsite information. The graph is bidirectional that helps in removing nodes efficiently.
type IRGraph struct {
	// Nodes of the graph
	IRNodes  map[string]*IRNode
	OutEdges IREdgeMap
	InEdges  IREdgeMap
}

// IRNode represents a node in the IRGraph.
type IRNode struct {
	// Pointer to the IR of the Function represented by this node.
	AST *ir.Func
	// Flat weight of the IRNode, obtained from profile.
	Flat int64
	// Cumulative weight of the IRNode.
	Cum int64
}

// IREdgeMap maps an IRNode to its successors.
type IREdgeMap map[*IRNode][]*IREdge

// IREdge represents a call edge in the IRGraph with source, destination, weight, callsite, and line number information.
type IREdge struct {
	// Source and destination of the edge in IRNode.
	Src, Dst *IRNode
	Weight   int64
	CallSite int
}

// NodeMapKey represents a hash key to identify unique call-edges in profile and in IR. Used for deduplication of call edges found in profile.
type NodeMapKey struct {
	CallerName string
	CalleeName string
	CallSite   int
}

// Weights capture both node weight and edge weight.
type Weights struct {
	NFlat   int64
	NCum    int64
	EWeight int64
}

// CallSiteInfo captures call-site information and its caller/callee.
type CallSiteInfo struct {
	Line   int
	Caller *ir.Func
	Callee *ir.Func
}

var (
	// Aggregated NodeWeights and EdgeWeights across profiles. This helps us determine the percentage threshold for hot/cold partitioning.
	GlobalTotalNodeWeight = int64(0)
	GlobalTotalEdgeWeight = int64(0)

	// Global node and their aggregated weight information.
	GlobalNodeMap = make(map[NodeMapKey]*Weights)

	// WeightedCG represents the IRGraph built from profile, which we will update as part of inlining.
	WeightedCG *IRGraph

	// Original profile-graph.
	ProfileGraph *Graph

	// Per-caller data structure to track the list of hot call sites. This gets rewritten every caller leaving it to GC for cleanup.
	ListOfHotCallSites = make(map[CallSiteInfo]struct{})
)

// BuildProfileGraph generates a profile-graph from the profile.
func BuildProfileGraph(profileFile string) {

	// if possible, we should cache the profile-graph.
	if ProfileGraph != nil {
		return
	}

	// open the profile file.
	f, err := os.Open(profileFile)
	if err != nil {
		log.Fatal("failed to open file " + profileFile)
		return
	}
	defer f.Close()
	p, err := profile.Parse(f)
	if err != nil {
		log.Fatal("failed to Parse profile file.")
		return
	}
	// Build the options.
	opt := &Options{
		CallTree:    false,
		SampleValue: func(v []int64) int64 { return v[1] },
	}
	// Build the graph using profile package.
	ProfileGraph = New(p, opt)

	// Build various global maps from profile.
	preprocessProfileGraph()

}

// BuildWeightedCallGraph generates a weighted callgraph from the profile for the current package.
func BuildWeightedCallGraph() {

	// Bail if there is no profile-graph available.
	if ProfileGraph == nil {
		return
	}

	// Create package-level call graph with weights from profile and IR.
	WeightedCG = createIRGraph()
}

// ConvertLine2Int converts ir.Line string to integer.
func ConvertLine2Int(line string) int {
	splits := strings.Split(line, ":")
	cs, _ := strconv.ParseInt(splits[len(splits)-2], 0, 64)
	return int(cs)
}

// preprocessProfileGraph builds various maps from the profile-graph. It builds GlobalNodeMap and other information based on the name and callsite to compute node and edge weights which will be used later on to create edges for WeightedCG.
func preprocessProfileGraph() {
	nFlat := make(map[string]int64)
	nCum := make(map[string]int64)

	// Accummulate weights for the same node.
	for _, n := range ProfileGraph.Nodes {
		canonicalName := n.Info.Name
		nFlat[canonicalName] += n.FlatValue()
		nCum[canonicalName] += n.CumValue()
	}

	// Process ProfileGraph and build various node and edge maps which will be consumed by AST walk.
	for _, n := range ProfileGraph.Nodes {
		GlobalTotalNodeWeight += n.FlatValue()
		canonicalName := n.Info.Name
		// Create the key to the NodeMapKey.
		nodeinfo := NodeMapKey{
			CallerName: canonicalName,
			CallSite:   n.Info.Lineno,
		}

		for _, e := range n.Out {
			GlobalTotalEdgeWeight += e.WeightValue()
			nodeinfo.CalleeName = e.Dest.Info.Name
			if w, ok := GlobalNodeMap[nodeinfo]; ok {
				w.EWeight += e.WeightValue()
			} else {
				weights := new(Weights)
				weights.NFlat = nFlat[canonicalName]
				weights.NCum = nCum[canonicalName]
				weights.EWeight = e.WeightValue()
				GlobalNodeMap[nodeinfo] = weights
			}
		}
	}
}

// createIRGraph builds the IRGraph by visting all the ir.Func in decl list of a package.
func createIRGraph() *IRGraph {
	var g IRGraph
	// Bottomup walk over the function to create IRGraph.
	ir.VisitFuncsBottomUp(typecheck.Target.Decls, func(list []*ir.Func, recursive bool) {
		for _, n := range list {
			g.Visit(n, recursive)
		}
	})
	return &g
}

// Visit traverses the body of each ir.Func and use GlobalNodeMap to determine if we need to add an edge from ir.Func and any node in the ir.Func body.
func (g *IRGraph) Visit(fn *ir.Func, recursive bool) {
	if g.IRNodes == nil {
		g.IRNodes = make(map[string]*IRNode)
	}
	if g.OutEdges == nil {
		g.OutEdges = make(map[*IRNode][]*IREdge)
	}
	if g.InEdges == nil {
		g.InEdges = make(map[*IRNode][]*IREdge)
	}
	name := ir.PkgFuncName(fn)
	node := new(IRNode)
	node.AST = fn
	if g.IRNodes[name] == nil {
		g.IRNodes[name] = node
	}
	// Create the key for the NodeMapKey.
	nodeinfo := NodeMapKey{
		CallerName: name,
		CalleeName: "",
		CallSite:   -1,
	}
	// If the node exists, then update its node weight.
	if weights, ok := GlobalNodeMap[nodeinfo]; ok {
		g.IRNodes[name].Flat = weights.NFlat
		g.IRNodes[name].Cum = weights.NCum
	}

	// Recursively walk over the body of the function to create IRGraph edges.
	g.createIRGraphEdge(fn, g.IRNodes[name], name)
}

// addEdge adds an edge between caller and new node that points to `callee` based on the profile-graph and GlobalNodeMap.
func (g *IRGraph) addEdge(caller *IRNode, callee *ir.Func, n *ir.Node, callername string, line int) {

	// Create an IRNode for the callee.
	calleenode := new(IRNode)
	calleenode.AST = callee
	calleename := ir.PkgFuncName(callee)

	// Create key for NodeMapKey.
	nodeinfo := NodeMapKey{
		CallerName: callername,
		CalleeName: calleename,
		CallSite:   line,
	}

	// Create the callee node with node weight.
	if g.IRNodes[calleename] == nil {
		g.IRNodes[calleename] = calleenode
		nodeinfo2 := NodeMapKey{
			CallerName: calleename,
			CalleeName: "",
			CallSite:   -1,
		}
		if weights, ok := GlobalNodeMap[nodeinfo2]; ok {
			g.IRNodes[calleename].Flat = weights.NFlat
			g.IRNodes[calleename].Cum = weights.NCum
		}
	}

	if weights, ok := GlobalNodeMap[nodeinfo]; ok {
		caller.Flat = weights.NFlat
		caller.Cum = weights.NCum

		// Add edge in the IRGraph from caller to callee.
		info := &IREdge{Src: caller, Dst: g.IRNodes[calleename], Weight: weights.EWeight, CallSite: line}
		g.OutEdges[caller] = append(g.OutEdges[caller], info)
		g.InEdges[g.IRNodes[calleename]] = append(g.InEdges[g.IRNodes[calleename]], info)
	} else {
		nodeinfo.CalleeName = ""
		nodeinfo.CallSite = -1
		if weights, ok := GlobalNodeMap[nodeinfo]; ok {
			caller.Flat = weights.NFlat
			caller.Cum = weights.NCum
			info := &IREdge{Src: caller, Dst: g.IRNodes[calleename], Weight: 0, CallSite: line}
			g.OutEdges[caller] = append(g.OutEdges[caller], info)
			g.InEdges[g.IRNodes[calleename]] = append(g.InEdges[g.IRNodes[calleename]], info)
		} else {
			info := &IREdge{Src: caller, Dst: g.IRNodes[calleename], Weight: 0, CallSite: line}
			g.OutEdges[caller] = append(g.OutEdges[caller], info)
			g.InEdges[g.IRNodes[calleename]] = append(g.InEdges[g.IRNodes[calleename]], info)
		}
	}
}

// createIRGraphEdge traverses the nodes in the body of ir.Func and add edges between callernode which points to the ir.Func and the nodes in the body.
func (g *IRGraph) createIRGraphEdge(fn *ir.Func, callernode *IRNode, name string) {
	var doNode func(ir.Node) bool
	doNode = func(n ir.Node) bool {
		switch n.Op() {
		default:
			ir.DoChildren(n, doNode)
		case ir.OCALLFUNC:
			call := n.(*ir.CallExpr)
			line := ConvertLine2Int(ir.Line(n))
			// Find the callee function from the call site and add the edge.
			f := inlCallee(call.X)
			if f != nil {
				g.addEdge(callernode, f, &n, name, line)
			}
		case ir.OCALLMETH:
			call := n.(*ir.CallExpr)
			// Find the callee method from the call site and add the edge.
			fn2 := ir.MethodExprName(call.X).Func
			line := ConvertLine2Int(ir.Line(n))
			g.addEdge(callernode, fn2, &n, name, line)
		}
		return false
	}
	doNode(fn)
}

// WeightInPercentage converts profile weights to a percentage.
func WeightInPercentage(value int64, total int64) float64 {
	var ratio float64
	if total != 0 {
		ratio = (float64(value) / float64(total)) * 100
	}
	return ratio
}

// PrintWeightedCallGraphDOT prints IRGraph in DOT format.
func PrintWeightedCallGraphDOT(nodeThreshold float64, edgeThreshold float64) {
	fmt.Printf("\ndigraph G {\n")
	fmt.Printf("forcelabels=true;\n")

	// List of functions in this package.
	funcs := make(map[string]struct{})
	ir.VisitFuncsBottomUp(typecheck.Target.Decls, func(list []*ir.Func, recursive bool) {
		for _, f := range list {
			name := ir.PkgFuncName(f)
			funcs[name] = struct{}{}
		}
	})

	// Determine nodes of DOT.
	nodes := make(map[string]*ir.Func)
	for name, _ := range funcs {
		if n, ok := WeightedCG.IRNodes[name]; ok {
			for _, e := range WeightedCG.OutEdges[n] {
				if _, ok := nodes[ir.PkgFuncName(e.Src.AST)]; !ok {
					nodes[ir.PkgFuncName(e.Src.AST)] = e.Src.AST
				}
				if _, ok := nodes[ir.PkgFuncName(e.Dst.AST)]; !ok {
					nodes[ir.PkgFuncName(e.Dst.AST)] = e.Dst.AST
				}
			}
			if _, ok := nodes[ir.PkgFuncName(n.AST)]; !ok {
				nodes[ir.PkgFuncName(n.AST)] = n.AST
			}
		}
	}

	// Print nodes.
	for name, ast := range nodes {
		if n, ok := WeightedCG.IRNodes[name]; ok {
			nodeweight := WeightInPercentage(n.Flat, GlobalTotalNodeWeight)
			color := "black"
			if nodeweight > nodeThreshold {
				color = "red"
			}
			if ast.Inl != nil {
				fmt.Printf("\"%v\" [color=%v,label=\"%v,freq=%.2f,inl_cost=%d\"];\n", ir.PkgFuncName(ast), color, ir.PkgFuncName(ast), nodeweight, ast.Inl.Cost)
			} else {
				fmt.Printf("\"%v\" [color=%v, label=\"%v,freq=%.2f\"];\n", ir.PkgFuncName(ast), color, ir.PkgFuncName(ast), nodeweight)
			}
		}
	}
	// Print edges.
	ir.VisitFuncsBottomUp(typecheck.Target.Decls, func(list []*ir.Func, recursive bool) {
		for _, f := range list {
			name := ir.PkgFuncName(f)
			if n, ok := WeightedCG.IRNodes[name]; ok {
				for _, e := range WeightedCG.OutEdges[n] {
					edgepercent := WeightInPercentage(e.Weight, GlobalTotalEdgeWeight)
					if edgepercent > edgeThreshold {
						fmt.Printf("edge [color=red, style=solid];\n")
					} else {
						fmt.Printf("edge [color=black, style=solid];\n")
					}

					fmt.Printf("\"%v\" -> \"%v\" [label=\"%.2f\"];\n", ir.PkgFuncName(n.AST), ir.PkgFuncName(e.Dst.AST), edgepercent)
				}
			}
		}
	})
	fmt.Printf("}\n")
}

// redirectEdges deletes the cur node out-edges and redirect them so now these edges are the parent node out-edges.
func redirectEdges(g *IRGraph, parent *IRNode, cur *IRNode) {
	for _, outEdge := range g.OutEdges[cur] {
		outEdge.Src = parent
		g.OutEdges[parent] = append(g.OutEdges[parent], outEdge)
	}
	delete(g.OutEdges, cur)
}

// RedirectEdges deletes and redirects out-edges from node cur based on inlining information via inlinedCallSites.
func RedirectEdges(cur *IRNode, inlinedCallSites map[CallSiteInfo]struct{}) {
	g := WeightedCG
	for i, outEdge := range g.OutEdges[cur] {
		if _, found := inlinedCallSites[CallSiteInfo{Line: outEdge.CallSite, Caller: cur.AST}]; !found {
			for _, InEdge := range g.InEdges[cur] {
				if _, ok := inlinedCallSites[CallSiteInfo{Line: InEdge.CallSite, Caller: InEdge.Src.AST}]; ok {
					weight := calculateweight(g, InEdge.Src, cur)
					redirectEdge(g, InEdge.Src, cur, outEdge, weight, i)
				}
			}
		} else {
			remove(g, cur, i, outEdge.Dst.AST.Nname)
		}
	}
	removeall(g, cur)
}

// calculateweight calculates the weight of the new redirected edge.
func calculateweight(g *IRGraph, parent *IRNode, cur *IRNode) int64 {
	sum := int64(0)
	pw := int64(0)
	for _, InEdge := range g.InEdges[cur] {
		sum = sum + InEdge.Weight
		if InEdge.Src == parent {
			pw = InEdge.Weight
		}
	}
	weight := int64(0)
	if sum != 0 {
		weight = pw / sum
	} else {
		weight = pw
	}
	return weight
}

// redirectEdge deletes the cur-node's out-edges and redirect them so now these edges are the parent node out-edges.
func redirectEdge(g *IRGraph, parent *IRNode, cur *IRNode, outEdge *IREdge, weight int64, idx int) {
	outEdge.Src = parent
	outEdge.Weight = weight * outEdge.Weight
	g.OutEdges[parent] = append(g.OutEdges[parent], outEdge)
	remove(g, cur, idx, outEdge.Dst.AST.Nname)
}

// remove deletes the cur-node's out-edges at index idx.
func remove(g *IRGraph, cur *IRNode, idx int, name *ir.Name) {
	if len(g.OutEdges[cur]) >= 2 {
		g.OutEdges[cur][idx] = &IREdge{CallSite: -1}
	} else {
		delete(g.OutEdges, cur)
	}
}

// removeall deletes all cur-node's out-edges that marked to be removed .
func removeall(g *IRGraph, cur *IRNode) {
	for i := len(g.OutEdges[cur]) - 1; i >= 0; i-- {
		if g.OutEdges[cur][i].CallSite == -1 {
			g.OutEdges[cur][i] = g.OutEdges[cur][len(g.OutEdges[cur])-1]
			g.OutEdges[cur] = g.OutEdges[cur][:len(g.OutEdges[cur])-1]
		}
	}
}

// inlCallee is same as the implementation for inl.go with one change. The change is that we do not invoke CanInline on a closure.
func inlCallee(fn ir.Node) *ir.Func {
	fn = ir.StaticValue(fn)
	switch fn.Op() {
	case ir.OMETHEXPR:
		fn := fn.(*ir.SelectorExpr)
		n := ir.MethodExprName(fn)
		// Check that receiver type matches fn.X.
		// TODO(mdempsky): Handle implicit dereference
		// of pointer receiver argument?
		if n == nil || !types.Identical(n.Type().Recv().Type, fn.X.Type()) {
			return nil
		}
		return n.Func
	case ir.ONAME:
		fn := fn.(*ir.Name)
		if fn.Class == ir.PFUNC {
			return fn.Func
		}
	case ir.OCLOSURE:
		fn := fn.(*ir.ClosureExpr)
		c := fn.Func
		return c
	}
	return nil
}
