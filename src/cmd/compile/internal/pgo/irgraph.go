package pgo

import (
	"cmd/compile/internal/base"
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

// IRGraph is the key datastrcture that is built from pprof profile. It is essentially a call graph with nodes pointing to IRs of functions and edges carrying weights and callsite information. The graph is bidirectional that helps in removing nodes efficiently. The IrGraph is updated as we pass through optimization phases, for example, after a function is inlined, we update the graph with new nodes and edges.
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
	// Flat weight of the IRNode, obtained from pprof.
	Flat int64
	// Cumulative weight of the IRNode.
	Cum int64
	// Is this function a recursive function.
	Recursive bool
	// Is this function a hot node?
	HotNode bool
}

// IREdgeMap maps an IRNode to its successors.
type IREdgeMap map[*IRNode][]*IREdge

// IREdge represents a call edge in the IRGraph with source, destination, weight, callsite, and line number information.
type IREdge struct {
	Src, Dst *IRNode
	DstNode  *ir.Node
	Weight   int64
	CallSite string
}

// NodeMapInfo represents a hash key to identify unique call-edges in pprof and in IR. Used for deduplication of call edges found in pprof.
type NodeMapInfo struct {
	FuncName string
	DstName  string
	CallSite int
}

// Weights capture both node weight and edge weight.
type Weights struct {
	NWeight      int64
	NTotalWeight int64
	EWeight      int64
}

// CallSiteInfo captures call site information and its static callee.
type CallSiteInfo struct {
	Line   string
	Caller *ir.Func
}

var (
	// Aggregated NodeWeights and EdgeWeights across profiles. This helps us determine the percentage threshold for hot/cold partitioning.
	GlobalTotalNodeWeight = int64(0)
	GlobalTotalEdgeWeight = int64(0)

	// Global node and their aggregated weight information.
	GlobalNodeMap = make(map[NodeMapInfo]*Weights)

	// WeightedCG represents the IRGraph built from pprof profile, which we will update as part of inlining.
	WeightedCG *IRGraph = nil

	// Original cross-package PProf Graph.
	PProfGraph *profile.Graph = nil
)

// BuildPProfGraph generates a pprof-graph from cpu-profile.
func BuildPProfGraph(profileFile string, opt *profile.Options) *profile.Graph {

	if PProfGraph != nil {
		return PProfGraph
	}
	// open the pprof profile file.
	f, err := os.Open(profileFile)
	if err != nil {
		log.Fatal("failed to open file " + profileFile)
		return nil
	}
	defer f.Close()
	p, err := profile.Parse(f)
	if err != nil {
		log.Fatal("failed to Parse profile file.")
		return nil
	}
	// Build the graph using google's pprof package.
	pProfGraph := profile.New(p, opt)

	// Build various global maps from pprof profile.
	preprocessPProfGraph(pProfGraph)

	return pProfGraph
}

// BuildWeightedCallGraph generates a weighted callgraph from the pprof profile for the current package.
func BuildWeightedCallGraph() *IRGraph {

	// Bail if there is no pprof-graph available.
	if PProfGraph == nil {
		return nil
	}

	// Create package-level call graph with weights from pprof profile and IR.
	weightedCG := createIRGraph()

	if weightedCG != nil && base.Flag.LowerM > 1 {
		log.Println("weighted call graph created successfully!")
	}

	return weightedCG
}

// preprocessPProfGraph builds various maps from profiles. It builds GlobalNodeMap and other information based on the name and callsite to compute node and edge weights which will be used later on to create edges of WeightedCG.
func preprocessPProfGraph(pProfGraph *profile.Graph) {
	nweight := make(map[string]int64)
	nTotalWeight := make(map[string]int64)

	// Accummulate weights for the same nodes.
	for _, n := range pProfGraph.Nodes {
		canonicalName := n.Info.Name
		if _, ok := nweight[canonicalName]; ok {
			nweight[canonicalName] += n.FlatValue()
			nTotalWeight[canonicalName] += n.CumValue()
		} else {
			nweight[canonicalName] = n.FlatValue()
			nTotalWeight[canonicalName] = n.CumValue()
		}
	}

	// Process PProfGraph and build various node and edge maps which will be consumed by AST walk.
	for _, n := range pProfGraph.Nodes {
		GlobalTotalNodeWeight += n.FlatValue()
		canonicalName := n.Info.Name
		// Create the key to the NodeMap.
		nodeinfo := NodeMapInfo{
			FuncName: canonicalName,
			CallSite: n.Info.Lineno,
		}
		// If there are no outgoing edges, we still need to create the node [for cold sites] with no callee information.
		if len(n.Out) == 0 {
			nodeinfo.DstName = ""
			nodeinfo.CallSite = -1

			weights := new(Weights)
			weights.NWeight = nweight[canonicalName]
			weights.NTotalWeight = nTotalWeight[canonicalName]
			weights.EWeight = 0

			GlobalNodeMap[nodeinfo] = weights
		}

		for _, e := range n.Out {
			GlobalTotalEdgeWeight += e.WeightValue()
			nodeinfo.DstName = e.Dest.Info.Name
			if w, ok := GlobalNodeMap[nodeinfo]; ok {
				w.EWeight += e.WeightValue()
			} else {
				weights := new(Weights)
				weights.NWeight = nweight[canonicalName]
				weights.NTotalWeight = nTotalWeight[canonicalName]
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
	node.Recursive = recursive
	if g.IRNodes[name] == nil {
		g.IRNodes[name] = node
	}
	// Create the hash key of the GlobalNodeMap.
	nodeinfo := NodeMapInfo{
		FuncName: name,
		DstName:  "",
		CallSite: -1,
	}
	// If the node exists, then update its node weight.
	if weights, ok := GlobalNodeMap[nodeinfo]; ok {
		g.IRNodes[name].Flat = weights.NWeight
		g.IRNodes[name].Cum = weights.NTotalWeight
	}

	// Recursively walk over the body of the function to create IRGraph edges.
	g.createIRGraphEdge(fn, g.IRNodes[name], name)
}

// addEdge adds an edge between node1 and new node that points to f based on pprof graph using GlobalNodeMap. node1 represents the caller. Callee as f. CallSite n.
func (g *IRGraph) addEdge(node1 *IRNode, f *ir.Func, n *ir.Node, name string, line string) {

	splits := strings.Split(line, ":")
	line2, _ := strconv.ParseInt(splits[len(splits)-2], 0, 64)

	// Create an IRNode for Callee.
	node2 := new(IRNode)
	node2.AST = f
	name2 := ir.PkgFuncName(f)

	// Create a hash key using NodeMapInfo with Caller FuncName and Callee fname.
	nodeinfo := NodeMapInfo{
		FuncName: name,
		DstName:  name2,
		CallSite: int(line2),
	}

	// A new callee node? If so create the node with node weight.
	// Remove this TODO.
	if g.IRNodes[name2] == nil {
		g.IRNodes[name2] = node2
		nodeinfo2 := NodeMapInfo{
			FuncName: name2,
			DstName:  "",
			CallSite: -1,
		}
		if weights, ok := GlobalNodeMap[nodeinfo2]; ok {
			g.IRNodes[name2].Flat = weights.NWeight
			g.IRNodes[name2].Cum = weights.NTotalWeight
		}
	}

	if weights, ok := GlobalNodeMap[nodeinfo]; ok {
		node1.Flat = weights.NWeight
		node1.Cum = weights.NTotalWeight

		// Add edge in the IRGraph from caller to callee [callee is an interface type here which can have multiple targets].
		info := &IREdge{Src: node1, Dst: g.IRNodes[name2], DstNode: n, Weight: weights.EWeight, CallSite: line}
		g.OutEdges[node1] = append(g.OutEdges[node1], info)
		g.InEdges[g.IRNodes[name2]] = append(g.InEdges[g.IRNodes[name2]], info)
	} else {
		nodeinfo.DstName = ""
		nodeinfo.CallSite = -1
		if weights, ok := GlobalNodeMap[nodeinfo]; ok {
			node1.Flat = weights.NWeight
			node1.Cum = weights.NTotalWeight
			info := &IREdge{Src: node1, Dst: g.IRNodes[name2], DstNode: n, Weight: 0, CallSite: line}
			g.OutEdges[node1] = append(g.OutEdges[node1], info)
			g.InEdges[g.IRNodes[name2]] = append(g.InEdges[g.IRNodes[name2]], info)
		} else {
			info := &IREdge{Src: node1, Dst: g.IRNodes[name2], DstNode: n, Weight: 0, CallSite: line}
			g.OutEdges[node1] = append(g.OutEdges[node1], info)
			g.InEdges[g.IRNodes[name2]] = append(g.InEdges[g.IRNodes[name2]], info)
		}
	}
}

// createIRGraphEdge traverses the nodes in the body of ir.Func and add edges between node1 which points to the ir.Func and the nodes in the body.
func (g *IRGraph) createIRGraphEdge(fn *ir.Func, node1 *IRNode, name string) {
	var doNode func(ir.Node) bool
	doNode = func(n ir.Node) bool {
		switch n.Op() {
		default:
			ir.DoChildren(n, doNode)
		case ir.OCALLFUNC:
			call := n.(*ir.CallExpr)
			line := ir.Line(n)
			// Find the callee function from the call site and add the edge.
			f := inlCallee(call.X)
			if f != nil {
				g.addEdge(node1, f, &n, name, line)
			}
		case ir.OCALLMETH:
			call := n.(*ir.CallExpr)
			// Find the callee method from the call site and add the edge.
			fn2 := ir.MethodExprName(call.X).Func
			line := ir.Line(n)
			g.addEdge(node1, fn2, &n, name, line)
		}
		return false
	}
	doNode(fn)
}

// WeightInPercentage converts profile weights to a percentage.
func WeightInPercentage(value int64, total int64) float64 {
	var ratio float64
	// percentage is computed at the (weight/totalweights) * 100
	// e.g. if edge weight is 30 and the sum of all the edges weight is 126
	// the percentage will be 23.8%
	if total != 0 {
		ratio = (float64(value) / float64(total)) * 100
	}
	return ratio
}

// PrintWeightedCallGraphDOT prints IRGraph in DOT format..
func PrintWeightedCallGraphDOT(threshold float64) {
	fmt.Printf("digraph G {\n")
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
			nodeweight := WeightInPercentage(n.Flat, GlobalTotalNodeWeight)
			for _, e := range WeightedCG.OutEdges[n] {
				if e.Weight != 0 {
					p := WeightInPercentage(e.Weight, GlobalTotalEdgeWeight)
					if p > threshold {
						nodes[ir.PkgFuncName(e.Src.AST)] = e.Src.AST
						nodes[ir.PkgFuncName(e.Dst.AST)] = e.Dst.AST
					}
				}
			}
			if nodeweight > threshold {
				nodes[ir.PkgFuncName(n.AST)] = n.AST
			}
		}
	}

	// Print nodes.
	for name, ast := range nodes {
		if n, ok := WeightedCG.IRNodes[name]; ok {
			nodeweight := WeightInPercentage(n.Flat, GlobalTotalNodeWeight)
			if ast.Inl != nil {
				fmt.Printf("\"%v\" [label=\"%v,freq=%.2f,inl_cost=%d\"];\n", ir.PkgFuncName(ast), ir.PkgFuncName(ast), nodeweight, ast.Inl.Cost)
			} else {
				fmt.Printf("\"%v\" [label=\"%v,freq=%.2f\"];\n", ir.PkgFuncName(ast), ir.PkgFuncName(ast), nodeweight)
			}
		}
	}
	// Print edges.
	ir.VisitFuncsBottomUp(typecheck.Target.Decls, func(list []*ir.Func, recursive bool) {
		for _, f := range list {
			name := ir.PkgFuncName(f)
			if n, ok := WeightedCG.IRNodes[name]; ok {
				for _, e := range WeightedCG.OutEdges[n] {
					if e.Weight != 0 {
						p := WeightInPercentage(e.Weight, GlobalTotalEdgeWeight)
						if p > threshold {
							fmt.Printf("edge [color=red, style=solid];\n")
						} else {
							fmt.Printf("edge [color=black, style=solid];\n")
						}

						fmt.Printf("\"%v\" -> \"%v\" [label=\"%.2f\"];\n", ir.PkgFuncName(n.AST), ir.PkgFuncName(e.Dst.AST), p)
					}
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
		if _, found := inlinedCallSites[CallSiteInfo{outEdge.CallSite, cur.AST}]; !found {
			for _, InEdge := range g.InEdges[cur] {
				if _, ok := inlinedCallSites[CallSiteInfo{InEdge.CallSite, InEdge.Src.AST}]; ok {
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
		g.OutEdges[cur][idx] = &IREdge{CallSite: "removed"}
	} else {
		delete(g.OutEdges, cur)
	}
}

// removeall deletes all cur-node's out-edges that marked to be removed .
func removeall(g *IRGraph, cur *IRNode) {
	for i := len(g.OutEdges[cur]) - 1; i >= 0; i-- {
		if g.OutEdges[cur][i].CallSite == "removed" {
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
