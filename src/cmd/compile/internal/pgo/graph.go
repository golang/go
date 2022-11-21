// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package graph collects a set of samples into a directed graph.

// Original file location: https://github.com/google/pprof/tree/main/internal/graph/graph.go
package pgo

import (
	"fmt"
	"internal/profile"
	"math"
	"sort"
	"strings"
)

// Options encodes the options for constructing a graph
type Options struct {
	SampleValue       func(s []int64) int64 // Function to compute the value of a sample
	SampleMeanDivisor func(s []int64) int64 // Function to compute the divisor for mean graphs, or nil

	CallTree     bool // Build a tree instead of a graph
	DropNegative bool // Drop nodes with overall negative values

	KeptNodes NodeSet // If non-nil, only use nodes in this set
}

// Nodes is an ordered collection of graph nodes.
type Nodes []*Node

// Node is an entry on a profiling report. It represents a unique
// program location.
type Node struct {
	// Info describes the source location associated to this node.
	Info NodeInfo

	// Function represents the function that this node belongs to. On
	// graphs with sub-function resolution (eg line number or
	// addresses), two nodes in a NodeMap that are part of the same
	// function have the same value of Node.Function. If the Node
	// represents the whole function, it points back to itself.
	Function *Node

	// Values associated to this node. Flat is exclusive to this node,
	// Cum includes all descendents.
	Flat, FlatDiv, Cum, CumDiv int64

	// In and out Contains the nodes immediately reaching or reached by
	// this node.
	In, Out EdgeMap
}

// Graph summarizes a performance profile into a format that is
// suitable for visualization.
type Graph struct {
	Nodes Nodes
}

// FlatValue returns the exclusive value for this node, computing the
// mean if a divisor is available.
func (n *Node) FlatValue() int64 {
	if n.FlatDiv == 0 {
		return n.Flat
	}
	return n.Flat / n.FlatDiv
}

// CumValue returns the inclusive value for this node, computing the
// mean if a divisor is available.
func (n *Node) CumValue() int64 {
	if n.CumDiv == 0 {
		return n.Cum
	}
	return n.Cum / n.CumDiv
}

// AddToEdge increases the weight of an edge between two nodes. If
// there isn't such an edge one is created.
func (n *Node) AddToEdge(to *Node, v int64, residual, inline bool) {
	n.AddToEdgeDiv(to, 0, v, residual, inline)
}

// AddToEdgeDiv increases the weight of an edge between two nodes. If
// there isn't such an edge one is created.
func (n *Node) AddToEdgeDiv(to *Node, dv, v int64, residual, inline bool) {
	if e := n.Out.FindTo(to); e != nil {
		e.WeightDiv += dv
		e.Weight += v
		if residual {
			e.Residual = true
		}
		if !inline {
			e.Inline = false
		}
		return
	}

	info := &Edge{Src: n, Dest: to, WeightDiv: dv, Weight: v, Residual: residual, Inline: inline}
	n.Out.Add(info)
	to.In.Add(info)
}

// NodeInfo contains the attributes for a node.
type NodeInfo struct {
	Name              string
	Address           uint64
	StartLine, Lineno int
	//File            string
	//OrigName        string
	//Objfile         string
}

// PrintableName calls the Node's Formatter function with a single space separator.
func (i *NodeInfo) PrintableName() string {
	return strings.Join(i.NameComponents(), " ")
}

// NameComponents returns the components of the printable name to be used for a node.
func (i *NodeInfo) NameComponents() []string {
	var name []string
	if i.Address != 0 {
		name = append(name, fmt.Sprintf("%016x", i.Address))
	}
	if fun := i.Name; fun != "" {
		name = append(name, fun)
	}

	switch {
	case i.Lineno != 0:
		// User requested line numbers, provide what we have.
		name = append(name, fmt.Sprintf(":%d", i.Lineno))
	case i.Name != "":
		// User requested function name. It was already included.
	default:
		// Do not leave it empty if there is no information at all.
		name = append(name, "<unknown>")
	}
	return name
}

// NodeMap maps from a node info struct to a node. It is used to merge
// report entries with the same info.
type NodeMap map[NodeInfo]*Node

// NodeSet is a collection of node info structs.
type NodeSet map[NodeInfo]bool

// NodePtrSet is a collection of nodes. Trimming a graph or tree requires a set
// of objects which uniquely identify the nodes to keep. In a graph, NodeInfo
// works as a unique identifier; however, in a tree multiple nodes may share
// identical NodeInfos. A *Node does uniquely identify a node so we can use that
// instead. Though a *Node also uniquely identifies a node in a graph,
// currently, during trimming, graphs are rebuilt from scratch using only the
// NodeSet, so there would not be the required context of the initial graph to
// allow for the use of *Node.
type NodePtrSet map[*Node]bool

// FindOrInsertNode takes the info for a node and either returns a matching node
// from the node map if one exists, or adds one to the map if one does not.
// If kept is non-nil, nodes are only added if they can be located on it.
func (nm NodeMap) FindOrInsertNode(info NodeInfo, kept NodeSet) *Node {
	if kept != nil {
		if _, ok := kept[info]; !ok {
			return nil
		}
	}

	if n, ok := nm[info]; ok {
		return n
	}

	n := &Node{
		Info: info,
	}
	nm[info] = n
	if info.Address == 0 && info.Lineno == 0 {
		// This node represents the whole function, so point Function
		// back to itself.
		n.Function = n
		return n
	}
	// Find a node that represents the whole function.
	info.Address = 0
	info.Lineno = 0
	n.Function = nm.FindOrInsertNode(info, nil)
	return n
}

// EdgeMap is used to represent the incoming/outgoing edges from a node.
type EdgeMap []*Edge

func (em EdgeMap) FindTo(n *Node) *Edge {
	for _, e := range em {
		if e.Dest == n {
			return e
		}
	}
	return nil
}

func (em *EdgeMap) Add(e *Edge) {
	*em = append(*em, e)
}

func (em *EdgeMap) Delete(e *Edge) {
	for i, edge := range *em {
		if edge == e {
			(*em)[i] = (*em)[len(*em)-1]
			*em = (*em)[:len(*em)-1]
			return
		}
	}
}

// Edge contains any attributes to be represented about edges in a graph.
type Edge struct {
	Src, Dest *Node
	// The summary weight of the edge
	Weight, WeightDiv int64

	// residual edges connect nodes that were connected through a
	// separate node, which has been removed from the report.
	Residual bool
	// An inline edge represents a call that was inlined into the caller.
	Inline bool
}

// WeightValue returns the weight value for this edge, normalizing if a
// divisor is available.
func (e *Edge) WeightValue() int64 {
	if e.WeightDiv == 0 {
		return e.Weight
	}
	return e.Weight / e.WeightDiv
}

// newGraph computes a graph from a profile.
func newGraph(prof *profile.Profile, o *Options) *Graph {
	nodes, locationMap := CreateNodes(prof, o)
	seenNode := make(map[*Node]bool)
	seenEdge := make(map[nodePair]bool)
	for _, sample := range prof.Sample {
		var w, dw int64
		w = o.SampleValue(sample.Value)
		if o.SampleMeanDivisor != nil {
			dw = o.SampleMeanDivisor(sample.Value)
		}
		if dw == 0 && w == 0 {
			continue
		}
		for k := range seenNode {
			delete(seenNode, k)
		}
		for k := range seenEdge {
			delete(seenEdge, k)
		}
		var parent *Node
		// A residual edge goes over one or more nodes that were not kept.
		residual := false

		// Group the sample frames, based on a global map.
		// Count only the last two frames as a call edge. Frames higher up
		// the stack are unlikely to be repeated calls (e.g. runtime.main
		// calling main.main). So adding weights to call edges higher up
		// the stack may be not reflecting the actual call edge weights
		// in the program. Without a branch profile this is just an
		// approximation.
		i := 1
		if last := len(sample.Location) - 1; last < i {
			i = last
		}
		for ; i >= 0; i-- {
			l := sample.Location[i]
			locNodes := locationMap.get(l.ID)
			for ni := len(locNodes) - 1; ni >= 0; ni-- {
				n := locNodes[ni]
				if n == nil {
					residual = true
					continue
				}
				// Add cum weight to all nodes in stack, avoiding double counting.
				_, sawNode := seenNode[n]
				if !sawNode {
					seenNode[n] = true
					n.addSample(dw, w, false)
				}
				// Update edge weights for all edges in stack, avoiding double counting.
				if (!sawNode || !seenEdge[nodePair{n, parent}]) && parent != nil && n != parent {
					seenEdge[nodePair{n, parent}] = true
					parent.AddToEdgeDiv(n, dw, w, residual, ni != len(locNodes)-1)
				}

				parent = n
				residual = false
			}
		}
		if parent != nil && !residual {
			// Add flat weight to leaf node.
			parent.addSample(dw, w, true)
		}
	}

	return selectNodesForGraph(nodes, o.DropNegative)
}

func selectNodesForGraph(nodes Nodes, dropNegative bool) *Graph {
	// Collect nodes into a graph.
	gNodes := make(Nodes, 0, len(nodes))
	for _, n := range nodes {
		if n == nil {
			continue
		}
		if n.Cum == 0 && n.Flat == 0 {
			continue
		}
		if dropNegative && isNegative(n) {
			continue
		}
		gNodes = append(gNodes, n)
	}
	return &Graph{gNodes}
}

type nodePair struct {
	src, dest *Node
}

func newTree(prof *profile.Profile, o *Options) (g *Graph) {
	parentNodeMap := make(map[*Node]NodeMap, len(prof.Sample))
	for _, sample := range prof.Sample {
		var w, dw int64
		w = o.SampleValue(sample.Value)
		if o.SampleMeanDivisor != nil {
			dw = o.SampleMeanDivisor(sample.Value)
		}
		if dw == 0 && w == 0 {
			continue
		}
		var parent *Node
		// Group the sample frames, based on a per-node map.
		for i := len(sample.Location) - 1; i >= 0; i-- {
			l := sample.Location[i]
			lines := l.Line
			if len(lines) == 0 {
				lines = []profile.Line{{}} // Create empty line to include location info.
			}
			for lidx := len(lines) - 1; lidx >= 0; lidx-- {
				nodeMap := parentNodeMap[parent]
				if nodeMap == nil {
					nodeMap = make(NodeMap)
					parentNodeMap[parent] = nodeMap
				}
				n := nodeMap.findOrInsertLine(l, lines[lidx], o)
				if n == nil {
					continue
				}
				n.addSample(dw, w, false)
				if parent != nil {
					parent.AddToEdgeDiv(n, dw, w, false, lidx != len(lines)-1)
				}
				parent = n
			}
		}
		if parent != nil {
			parent.addSample(dw, w, true)
		}
	}

	nodes := make(Nodes, len(prof.Location))
	for _, nm := range parentNodeMap {
		nodes = append(nodes, nm.nodes()...)
	}
	return selectNodesForGraph(nodes, o.DropNegative)
}

// isNegative returns true if the node is considered as "negative" for the
// purposes of drop_negative.
func isNegative(n *Node) bool {
	switch {
	case n.Flat < 0:
		return true
	case n.Flat == 0 && n.Cum < 0:
		return true
	default:
		return false
	}
}

type locationMap struct {
	s []Nodes          // a slice for small sequential IDs
	m map[uint64]Nodes // fallback for large IDs (unlikely)
}

func (l *locationMap) add(id uint64, n Nodes) {
	if id < uint64(len(l.s)) {
		l.s[id] = n
	} else {
		l.m[id] = n
	}
}

func (l locationMap) get(id uint64) Nodes {
	if id < uint64(len(l.s)) {
		return l.s[id]
	} else {
		return l.m[id]
	}
}

// CreateNodes creates graph nodes for all locations in a profile. It
// returns set of all nodes, plus a mapping of each location to the
// set of corresponding nodes (one per location.Line).
func CreateNodes(prof *profile.Profile, o *Options) (Nodes, locationMap) {
	locations := locationMap{make([]Nodes, len(prof.Location)+1), make(map[uint64]Nodes)}
	nm := make(NodeMap, len(prof.Location))
	for _, l := range prof.Location {
		lines := l.Line
		if len(lines) == 0 {
			lines = []profile.Line{{}} // Create empty line to include location info.
		}
		nodes := make(Nodes, len(lines))
		for ln := range lines {
			nodes[ln] = nm.findOrInsertLine(l, lines[ln], o)
		}
		locations.add(l.ID, nodes)
	}
	return nm.nodes(), locations
}

func (nm NodeMap) nodes() Nodes {
	nodes := make(Nodes, 0, len(nm))
	for _, n := range nm {
		nodes = append(nodes, n)
	}
	return nodes
}

func (nm NodeMap) findOrInsertLine(l *profile.Location, li profile.Line, o *Options) *Node {
	var objfile string
	if m := l.Mapping; m != nil && m.File != "" {
		objfile = m.File
	}

	if ni := nodeInfo(l, li, objfile, o); ni != nil {
		return nm.FindOrInsertNode(*ni, o.KeptNodes)
	}
	return nil
}

func nodeInfo(l *profile.Location, line profile.Line, objfile string, o *Options) *NodeInfo {
	if line.Function == nil {
		return &NodeInfo{Address: l.Address}
	}
	ni := &NodeInfo{
		Address: l.Address,
		Lineno:  int(line.Line),
		Name:    line.Function.Name,
	}
	ni.StartLine = int(line.Function.StartLine)
	return ni
}

// Sum adds the flat and cum values of a set of nodes.
func (ns Nodes) Sum() (flat int64, cum int64) {
	for _, n := range ns {
		flat += n.Flat
		cum += n.Cum
	}
	return
}

func (n *Node) addSample(dw, w int64, flat bool) {
	// Update sample value
	if flat {
		n.FlatDiv += dw
		n.Flat += w
	} else {
		n.CumDiv += dw
		n.Cum += w
	}
}

// String returns a text representation of a graph, for debugging purposes.
func (g *Graph) String() string {
	var s []string

	nodeIndex := make(map[*Node]int, len(g.Nodes))

	for i, n := range g.Nodes {
		nodeIndex[n] = i + 1
	}

	for i, n := range g.Nodes {
		name := n.Info.PrintableName()
		var in, out []int

		for _, from := range n.In {
			in = append(in, nodeIndex[from.Src])
		}
		for _, to := range n.Out {
			out = append(out, nodeIndex[to.Dest])
		}
		s = append(s, fmt.Sprintf("%d: %s[flat=%d cum=%d] %x -> %v ", i+1, name, n.Flat, n.Cum, in, out))
	}
	return strings.Join(s, "\n")
}

// DiscardLowFrequencyNodes returns a set of the nodes at or over a
// specific cum value cutoff.
func (g *Graph) DiscardLowFrequencyNodes(nodeCutoff int64) NodeSet {
	return makeNodeSet(g.Nodes, nodeCutoff)
}

// DiscardLowFrequencyNodePtrs returns a NodePtrSet of nodes at or over a
// specific cum value cutoff.
func (g *Graph) DiscardLowFrequencyNodePtrs(nodeCutoff int64) NodePtrSet {
	cutNodes := getNodesAboveCumCutoff(g.Nodes, nodeCutoff)
	kept := make(NodePtrSet, len(cutNodes))
	for _, n := range cutNodes {
		kept[n] = true
	}
	return kept
}

func makeNodeSet(nodes Nodes, nodeCutoff int64) NodeSet {
	cutNodes := getNodesAboveCumCutoff(nodes, nodeCutoff)
	kept := make(NodeSet, len(cutNodes))
	for _, n := range cutNodes {
		kept[n.Info] = true
	}
	return kept
}

// getNodesAboveCumCutoff returns all the nodes which have a Cum value greater
// than or equal to cutoff.
func getNodesAboveCumCutoff(nodes Nodes, nodeCutoff int64) Nodes {
	cutoffNodes := make(Nodes, 0, len(nodes))
	for _, n := range nodes {
		if abs64(n.Cum) < nodeCutoff {
			continue
		}
		cutoffNodes = append(cutoffNodes, n)
	}
	return cutoffNodes
}

// TrimLowFrequencyEdges removes edges that have less than
// the specified weight. Returns the number of edges removed
func (g *Graph) TrimLowFrequencyEdges(edgeCutoff int64) int {
	var droppedEdges int
	for _, n := range g.Nodes {
		for _, e := range n.In {
			if abs64(e.Weight) < edgeCutoff {
				n.In.Delete(e)
				e.Src.Out.Delete(e)
				droppedEdges++
			}
		}
	}
	return droppedEdges
}

// SortNodes sorts the nodes in a graph based on a specific heuristic.
func (g *Graph) SortNodes(cum bool, visualMode bool) {
	// Sort nodes based on requested mode
	switch {
	case visualMode:
		// Specialized sort to produce a more visually-interesting graph
		g.Nodes.Sort(EntropyOrder)
	case cum:
		g.Nodes.Sort(CumNameOrder)
	default:
		g.Nodes.Sort(FlatNameOrder)
	}
}

// SelectTopNodePtrs returns a set of the top maxNodes *Node in a graph.
func (g *Graph) SelectTopNodePtrs(maxNodes int, visualMode bool) NodePtrSet {
	set := make(NodePtrSet)
	for _, node := range g.selectTopNodes(maxNodes, visualMode) {
		set[node] = true
	}
	return set
}

// SelectTopNodes returns a set of the top maxNodes nodes in a graph.
func (g *Graph) SelectTopNodes(maxNodes int, visualMode bool) NodeSet {
	return makeNodeSet(g.selectTopNodes(maxNodes, visualMode), 0)
}

// selectTopNodes returns a slice of the top maxNodes nodes in a graph.
func (g *Graph) selectTopNodes(maxNodes int, visualMode bool) Nodes {
	if maxNodes > len(g.Nodes) {
		maxNodes = len(g.Nodes)
	}
	return g.Nodes[:maxNodes]
}

// nodeSorter is a mechanism used to allow a report to be sorted
// in different ways.
type nodeSorter struct {
	rs   Nodes
	less func(l, r *Node) bool
}

func (s nodeSorter) Len() int           { return len(s.rs) }
func (s nodeSorter) Swap(i, j int)      { s.rs[i], s.rs[j] = s.rs[j], s.rs[i] }
func (s nodeSorter) Less(i, j int) bool { return s.less(s.rs[i], s.rs[j]) }

// Sort reorders a slice of nodes based on the specified ordering
// criteria. The result is sorted in decreasing order for (absolute)
// numeric quantities, alphabetically for text, and increasing for
// addresses.
func (ns Nodes) Sort(o NodeOrder) error {
	var s nodeSorter

	switch o {
	case FlatNameOrder:
		s = nodeSorter{ns,
			func(l, r *Node) bool {
				if iv, jv := abs64(l.Flat), abs64(r.Flat); iv != jv {
					return iv > jv
				}
				if iv, jv := l.Info.PrintableName(), r.Info.PrintableName(); iv != jv {
					return iv < jv
				}
				if iv, jv := abs64(l.Cum), abs64(r.Cum); iv != jv {
					return iv > jv
				}
				return compareNodes(l, r)
			},
		}
	case FlatCumNameOrder:
		s = nodeSorter{ns,
			func(l, r *Node) bool {
				if iv, jv := abs64(l.Flat), abs64(r.Flat); iv != jv {
					return iv > jv
				}
				if iv, jv := abs64(l.Cum), abs64(r.Cum); iv != jv {
					return iv > jv
				}
				if iv, jv := l.Info.PrintableName(), r.Info.PrintableName(); iv != jv {
					return iv < jv
				}
				return compareNodes(l, r)
			},
		}
	case NameOrder:
		s = nodeSorter{ns,
			func(l, r *Node) bool {
				if iv, jv := l.Info.Name, r.Info.Name; iv != jv {
					return iv < jv
				}
				return compareNodes(l, r)
			},
		}
	case FileOrder:
		s = nodeSorter{ns,
			func(l, r *Node) bool {
				if iv, jv := l.Info.StartLine, r.Info.StartLine; iv != jv {
					return iv < jv
				}
				return compareNodes(l, r)
			},
		}
	case AddressOrder:
		s = nodeSorter{ns,
			func(l, r *Node) bool {
				if iv, jv := l.Info.Address, r.Info.Address; iv != jv {
					return iv < jv
				}
				return compareNodes(l, r)
			},
		}
	case CumNameOrder, EntropyOrder:
		// Hold scoring for score-based ordering
		var score map[*Node]int64
		scoreOrder := func(l, r *Node) bool {
			if iv, jv := abs64(score[l]), abs64(score[r]); iv != jv {
				return iv > jv
			}
			if iv, jv := l.Info.PrintableName(), r.Info.PrintableName(); iv != jv {
				return iv < jv
			}
			if iv, jv := abs64(l.Flat), abs64(r.Flat); iv != jv {
				return iv > jv
			}
			return compareNodes(l, r)
		}

		switch o {
		case CumNameOrder:
			score = make(map[*Node]int64, len(ns))
			for _, n := range ns {
				score[n] = n.Cum
			}
			s = nodeSorter{ns, scoreOrder}
		case EntropyOrder:
			score = make(map[*Node]int64, len(ns))
			for _, n := range ns {
				score[n] = entropyScore(n)
			}
			s = nodeSorter{ns, scoreOrder}
		}
	default:
		return fmt.Errorf("report: unrecognized sort ordering: %d", o)
	}
	sort.Sort(s)
	return nil
}

// compareNodes compares two nodes to provide a deterministic ordering
// between them. Two nodes cannot have the same Node.Info value.
func compareNodes(l, r *Node) bool {
	return fmt.Sprint(l.Info) < fmt.Sprint(r.Info)
}

// entropyScore computes a score for a node representing how important
// it is to include this node on a graph visualization. It is used to
// sort the nodes and select which ones to display if we have more
// nodes than desired in the graph. This number is computed by looking
// at the flat and cum weights of the node and the incoming/outgoing
// edges. The fundamental idea is to penalize nodes that have a simple
// fallthrough from their incoming to the outgoing edge.
func entropyScore(n *Node) int64 {
	score := float64(0)

	if len(n.In) == 0 {
		score++ // Favor entry nodes
	} else {
		score += edgeEntropyScore(n, n.In, 0)
	}

	if len(n.Out) == 0 {
		score++ // Favor leaf nodes
	} else {
		score += edgeEntropyScore(n, n.Out, n.Flat)
	}

	return int64(score*float64(n.Cum)) + n.Flat
}

// edgeEntropyScore computes the entropy value for a set of edges
// coming in or out of a node. Entropy (as defined in information
// theory) refers to the amount of information encoded by the set of
// edges. A set of edges that have a more interesting distribution of
// samples gets a higher score.
func edgeEntropyScore(n *Node, edges EdgeMap, self int64) float64 {
	score := float64(0)
	total := self
	for _, e := range edges {
		if e.Weight > 0 {
			total += abs64(e.Weight)
		}
	}
	if total != 0 {
		for _, e := range edges {
			frac := float64(abs64(e.Weight)) / float64(total)
			score += -frac * math.Log2(frac)
		}
		if self > 0 {
			frac := float64(abs64(self)) / float64(total)
			score += -frac * math.Log2(frac)
		}
	}
	return score
}

// NodeOrder sets the ordering for a Sort operation
type NodeOrder int

// Sorting options for node sort.
const (
	FlatNameOrder NodeOrder = iota
	FlatCumNameOrder
	CumNameOrder
	NameOrder
	FileOrder
	AddressOrder
	EntropyOrder
)

// Sort returns a slice of the edges in the map, in a consistent
// order. The sort order is first based on the edge weight
// (higher-to-lower) and then by the node names to avoid flakiness.
func (e EdgeMap) Sort() []*Edge {
	el := make(edgeList, 0, len(e))
	for _, w := range e {
		el = append(el, w)
	}

	sort.Sort(el)
	return el
}

// Sum returns the total weight for a set of nodes.
func (e EdgeMap) Sum() int64 {
	var ret int64
	for _, edge := range e {
		ret += edge.Weight
	}
	return ret
}

type edgeList []*Edge

func (el edgeList) Len() int {
	return len(el)
}

func (el edgeList) Less(i, j int) bool {
	if el[i].Weight != el[j].Weight {
		return abs64(el[i].Weight) > abs64(el[j].Weight)
	}

	from1 := el[i].Src.Info.PrintableName()
	from2 := el[j].Src.Info.PrintableName()
	if from1 != from2 {
		return from1 < from2
	}

	to1 := el[i].Dest.Info.PrintableName()
	to2 := el[j].Dest.Info.PrintableName()

	return to1 < to2
}

func (el edgeList) Swap(i, j int) {
	el[i], el[j] = el[j], el[i]
}

func abs64(i int64) int64 {
	if i < 0 {
		return -i
	}
	return i
}
