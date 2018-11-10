package graph

import (
	"fmt"
	"testing"
)

func edgeDebugString(edge *Edge) string {
	debug := ""
	debug += fmt.Sprintf("\t\tSrc: %p\n", edge.Src)
	debug += fmt.Sprintf("\t\tDest: %p\n", edge.Dest)
	debug += fmt.Sprintf("\t\tWeight: %d\n", edge.Weight)
	debug += fmt.Sprintf("\t\tResidual: %t\n", edge.Residual)
	debug += fmt.Sprintf("\t\tInline: %t\n", edge.Inline)
	return debug
}

func edgeMapsDebugString(in, out EdgeMap) string {
	debug := ""
	debug += "In Edges:\n"
	for parent, edge := range in {
		debug += fmt.Sprintf("\tParent: %p\n", parent)
		debug += edgeDebugString(edge)
	}
	debug += "Out Edges:\n"
	for child, edge := range out {
		debug += fmt.Sprintf("\tChild: %p\n", child)
		debug += edgeDebugString(edge)
	}
	return debug
}

func graphDebugString(graph *Graph) string {
	debug := ""
	for i, node := range graph.Nodes {
		debug += fmt.Sprintf("Node %d: %p\n", i, node)
	}

	for i, node := range graph.Nodes {
		debug += "\n"
		debug += fmt.Sprintf("===  Node %d: %p  ===\n", i, node)
		debug += edgeMapsDebugString(node.In, node.Out)
	}
	return debug
}

func expectedNodesDebugString(expected []expectedNode) string {
	debug := ""
	for i, node := range expected {
		debug += fmt.Sprintf("Node %d: %p\n", i, node.node)
	}

	for i, node := range expected {
		debug += "\n"
		debug += fmt.Sprintf("===  Node %d: %p  ===\n", i, node.node)
		debug += edgeMapsDebugString(node.in, node.out)
	}
	return debug
}

// edgeMapsEqual checks if all the edges in this equal all the edges in that.
func edgeMapsEqual(this, that EdgeMap) bool {
	if len(this) != len(that) {
		return false
	}
	for node, thisEdge := range this {
		if *thisEdge != *that[node] {
			return false
		}
	}
	return true
}

// nodesEqual checks if node is equal to expected.
func nodesEqual(node *Node, expected expectedNode) bool {
	return node == expected.node && edgeMapsEqual(node.In, expected.in) &&
		edgeMapsEqual(node.Out, expected.out)
}

// graphsEqual checks if graph is equivalent to the graph templated by expected.
func graphsEqual(graph *Graph, expected []expectedNode) bool {
	if len(graph.Nodes) != len(expected) {
		return false
	}
	expectedSet := make(map[*Node]expectedNode)
	for i := range expected {
		expectedSet[expected[i].node] = expected[i]
	}

	for _, node := range graph.Nodes {
		expectedNode, found := expectedSet[node]
		if !found || !nodesEqual(node, expectedNode) {
			return false
		}
	}
	return true
}

type expectedNode struct {
	node    *Node
	in, out EdgeMap
}

type trimTreeTestcase struct {
	initial  *Graph
	expected []expectedNode
	keep     NodePtrSet
}

// makeExpectedEdgeResidual makes the edge from parent to child residual.
func makeExpectedEdgeResidual(parent, child expectedNode) {
	parent.out[child.node].Residual = true
	child.in[parent.node].Residual = true
}

func makeEdgeInline(edgeMap EdgeMap, node *Node) {
	edgeMap[node].Inline = true
}

func setEdgeWeight(edgeMap EdgeMap, node *Node, weight int64) {
	edgeMap[node].Weight = weight
}

// createEdges creates directed edges from the parent to each of the children.
func createEdges(parent *Node, children ...*Node) {
	for _, child := range children {
		edge := &Edge{
			Src:  parent,
			Dest: child,
		}
		parent.Out[child] = edge
		child.In[parent] = edge
	}
}

// createEmptyNode creates a node without any edges.
func createEmptyNode() *Node {
	return &Node{
		In:  make(EdgeMap),
		Out: make(EdgeMap),
	}
}

// createExpectedNodes creates a slice of expectedNodes from nodes.
func createExpectedNodes(nodes ...*Node) ([]expectedNode, NodePtrSet) {
	expected := make([]expectedNode, len(nodes))
	keep := make(NodePtrSet, len(nodes))

	for i, node := range nodes {
		expected[i] = expectedNode{
			node: node,
			in:   make(EdgeMap),
			out:  make(EdgeMap),
		}
		keep[node] = true
	}

	return expected, keep
}

// createExpectedEdges creates directed edges from the parent to each of the
// children.
func createExpectedEdges(parent expectedNode, children ...expectedNode) {
	for _, child := range children {
		edge := &Edge{
			Src:  parent.node,
			Dest: child.node,
		}
		parent.out[child.node] = edge
		child.in[parent.node] = edge
	}
}

// createTestCase1 creates a test case that initally looks like:
//     0
//     |(5)
//     1
// (3)/ \(4)
//   2   3.
//
// After keeping 0, 2, and 3, it expects the graph:
//     0
// (3)/ \(4)
//   2   3.
func createTestCase1() trimTreeTestcase {
	// Create initial graph
	graph := &Graph{make(Nodes, 4)}
	nodes := graph.Nodes
	for i := range nodes {
		nodes[i] = createEmptyNode()
	}
	createEdges(nodes[0], nodes[1])
	createEdges(nodes[1], nodes[2], nodes[3])
	makeEdgeInline(nodes[0].Out, nodes[1])
	makeEdgeInline(nodes[1].Out, nodes[2])
	setEdgeWeight(nodes[0].Out, nodes[1], 5)
	setEdgeWeight(nodes[1].Out, nodes[2], 3)
	setEdgeWeight(nodes[1].Out, nodes[3], 4)

	// Create expected graph
	expected, keep := createExpectedNodes(nodes[0], nodes[2], nodes[3])
	createExpectedEdges(expected[0], expected[1], expected[2])
	makeEdgeInline(expected[0].out, expected[1].node)
	makeExpectedEdgeResidual(expected[0], expected[1])
	makeExpectedEdgeResidual(expected[0], expected[2])
	setEdgeWeight(expected[0].out, expected[1].node, 3)
	setEdgeWeight(expected[0].out, expected[2].node, 4)
	return trimTreeTestcase{
		initial:  graph,
		expected: expected,
		keep:     keep,
	}
}

// createTestCase2 creates a test case that initially looks like:
//   3
//   | (12)
//   1
//   | (8)
//   2
//   | (15)
//   0
//   | (10)
//   4.
//
// After keeping 3 and 4, it expects the graph:
//   3
//   | (10)
//   4.
func createTestCase2() trimTreeTestcase {
	// Create initial graph
	graph := &Graph{make(Nodes, 5)}
	nodes := graph.Nodes
	for i := range nodes {
		nodes[i] = createEmptyNode()
	}
	createEdges(nodes[3], nodes[1])
	createEdges(nodes[1], nodes[2])
	createEdges(nodes[2], nodes[0])
	createEdges(nodes[0], nodes[4])
	setEdgeWeight(nodes[3].Out, nodes[1], 12)
	setEdgeWeight(nodes[1].Out, nodes[2], 8)
	setEdgeWeight(nodes[2].Out, nodes[0], 15)
	setEdgeWeight(nodes[0].Out, nodes[4], 10)

	// Create expected graph
	expected, keep := createExpectedNodes(nodes[3], nodes[4])
	createExpectedEdges(expected[0], expected[1])
	makeExpectedEdgeResidual(expected[0], expected[1])
	setEdgeWeight(expected[0].out, expected[1].node, 10)
	return trimTreeTestcase{
		initial:  graph,
		expected: expected,
		keep:     keep,
	}
}

// createTestCase3 creates an initally empty graph and expects an empty graph
// after trimming.
func createTestCase3() trimTreeTestcase {
	graph := &Graph{make(Nodes, 0)}
	expected, keep := createExpectedNodes()
	return trimTreeTestcase{
		initial:  graph,
		expected: expected,
		keep:     keep,
	}
}

// createTestCase4 creates a test case that initially looks like:
//   0.
//
// After keeping 0, it expects the graph:
//   0.
func createTestCase4() trimTreeTestcase {
	graph := &Graph{make(Nodes, 1)}
	nodes := graph.Nodes
	for i := range nodes {
		nodes[i] = createEmptyNode()
	}
	expected, keep := createExpectedNodes(nodes[0])
	return trimTreeTestcase{
		initial:  graph,
		expected: expected,
		keep:     keep,
	}
}

func createTrimTreeTestCases() []trimTreeTestcase {
	caseGenerators := []func() trimTreeTestcase{
		createTestCase1,
		createTestCase2,
		createTestCase3,
		createTestCase4,
	}
	cases := make([]trimTreeTestcase, len(caseGenerators))
	for i, gen := range caseGenerators {
		cases[i] = gen()
	}
	return cases
}

func TestTrimTree(t *testing.T) {
	tests := createTrimTreeTestCases()
	for _, test := range tests {
		graph := test.initial
		graph.TrimTree(test.keep)
		if !graphsEqual(graph, test.expected) {
			t.Fatalf("Graphs do not match.\nExpected: %s\nFound: %s\n",
				expectedNodesDebugString(test.expected),
				graphDebugString(graph))
		}
	}
}
