// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"container/heap"
	"fmt"
)

// initOrder computes the Info.InitOrder for package variables.
func (check *Checker) initOrder() {
	// An InitOrder may already have been computed if a package is
	// built from several calls to (*Checker).Files.  Clear it.
	check.Info.InitOrder = check.Info.InitOrder[:0]

	// compute the object dependency graph and
	// initialize a priority queue with the list
	// of graph nodes
	pq := nodeQueue(dependencyGraph(check.objMap))
	heap.Init(&pq)

	const debug = false
	if debug {
		fmt.Printf("package %s: object dependency graph\n", check.pkg.Name())
		for _, n := range pq {
			for _, o := range n.out {
				fmt.Printf("\t%s -> %s\n", n.obj.Name(), o.obj.Name())
			}
		}
		fmt.Println()
		fmt.Printf("package %s: initialization order\n", check.pkg.Name())
	}

	// determine initialization order by removing the highest priority node
	// (the one with the fewest dependencies) and its edges from the graph,
	// repeatedly, until there are no nodes left.
	// In a valid Go program, those nodes always have zero dependencies (after
	// removing all incoming dependencies), otherwise there are initialization
	// cycles.
	mark := 0
	emitted := make(map[*declInfo]bool)
	for len(pq) > 0 {
		// get the next node
		n := heap.Pop(&pq).(*objNode)

		// if n still depends on other nodes, we have a cycle
		if n.in > 0 {
			mark++ // mark nodes using a different value each time
			cycle := findPath(n, n, mark)
			if i := valIndex(cycle); i >= 0 {
				check.reportCycle(cycle, i)
			}
			// ok to continue, but the variable initialization order
			// will be incorrect at this point since it assumes no
			// cycle errors
		}

		// reduce dependency count of all dependent nodes
		// and update priority queue
		for _, out := range n.out {
			out.in--
			heap.Fix(&pq, out.index)
		}

		// record the init order for variables with initializers only
		v, _ := n.obj.(*Var)
		info := check.objMap[v]
		if v == nil || !info.hasInitializer() {
			continue
		}

		// n:1 variable declarations such as: a, b = f()
		// introduce a node for each lhs variable (here: a, b);
		// but they all have the same initializer - emit only
		// one, for the first variable seen
		if emitted[info] {
			continue // initializer already emitted, if any
		}
		emitted[info] = true

		infoLhs := info.lhs // possibly nil (see declInfo.lhs field comment)
		if infoLhs == nil {
			infoLhs = []*Var{v}
		}
		init := &Initializer{infoLhs, info.init}
		check.Info.InitOrder = append(check.Info.InitOrder, init)

		if debug {
			fmt.Printf("\t%s\n", init)
		}
	}

	if debug {
		fmt.Println()
	}
}

// findPath returns the (reversed) list of nodes z, ... c, b, a,
// such that there is a path (list of edges) from a to z.
// If there is no such path, the result is nil.
// Nodes marked with the value mark are considered "visited";
// unvisited nodes are marked during the graph search.
func findPath(a, z *objNode, mark int) []*objNode {
	if a.mark == mark {
		return nil // node already seen
	}
	a.mark = mark

	for _, n := range a.out {
		if n == z {
			return []*objNode{z}
		}
		if P := findPath(n, z, mark); P != nil {
			return append(P, n)
		}
	}

	return nil
}

// valIndex returns the index of the first constant or variable in a,
// if any; or a value < 0.
func valIndex(a []*objNode) int {
	for i, n := range a {
		switch n.obj.(type) {
		case *Const, *Var:
			return i
		}
	}
	return -1
}

// reportCycle reports an error for the cycle starting at i.
func (check *Checker) reportCycle(cycle []*objNode, i int) {
	obj := cycle[i].obj
	check.errorf(obj.Pos(), "initialization cycle for %s", obj.Name())
	// print cycle
	for _ = range cycle {
		check.errorf(obj.Pos(), "\t%s refers to", obj.Name()) // secondary error, \t indented
		i++
		if i >= len(cycle) {
			i = 0
		}
		obj = cycle[i].obj
	}
	check.errorf(obj.Pos(), "\t%s", obj.Name())
}

// An objNode represents a node in the object dependency graph.
// Each node b in a.out represents an edge a->b indicating that
// b depends on a.
// Nodes may be marked for cycle detection. A node n is marked
// if n.mark corresponds to the current mark value.
type objNode struct {
	obj   Object     // object represented by this node
	in    int        // number of nodes this node depends on
	out   []*objNode // list of nodes that depend on this node
	index int        // node index in list of nodes
	mark  int        // for cycle detection
}

// dependencyGraph computes the transposed object dependency graph
// from the given objMap. The transposed graph is returned as a list
// of nodes; an edge d->n indicates that node n depends on node d.
func dependencyGraph(objMap map[Object]*declInfo) []*objNode {
	// M maps each object to its corresponding node
	M := make(map[Object]*objNode, len(objMap))
	for obj := range objMap {
		M[obj] = &objNode{obj: obj}
	}

	// G is the graph of nodes n
	G := make([]*objNode, len(M))
	i := 0
	for obj, n := range M {
		deps := objMap[obj].deps
		n.in = len(deps)
		for d := range deps {
			d := M[d]                // node n depends on node d
			d.out = append(d.out, n) // add edge d->n
		}

		G[i] = n
		n.index = i
		i++
	}

	return G
}

// nodeQueue implements the container/heap interface;
// a nodeQueue may be used as a priority queue.
type nodeQueue []*objNode

func (a nodeQueue) Len() int { return len(a) }

func (a nodeQueue) Swap(i, j int) {
	x, y := a[i], a[j]
	a[i], a[j] = y, x
	x.index, y.index = j, i
}

func (a nodeQueue) Less(i, j int) bool {
	x, y := a[i], a[j]
	// nodes are prioritized by number of incoming dependencies (1st key)
	// and source order (2nd key)
	return x.in < y.in || x.in == y.in && x.obj.order() < y.obj.order()
}

func (a *nodeQueue) Push(x interface{}) {
	panic("unreachable")
}

func (a *nodeQueue) Pop() interface{} {
	n := len(*a)
	x := (*a)[n-1]
	x.index = -1 // for safety
	*a = (*a)[:n-1]
	return x
}
