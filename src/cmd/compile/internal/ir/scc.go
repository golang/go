// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

// Strongly connected components.
//
// Run analysis on minimal sets of mutually recursive functions
// or single non-recursive functions, bottom up.
//
// Finding these sets is finding strongly connected components
// by reverse topological order in the static call graph.
// The algorithm (known as Tarjan's algorithm) for doing that is taken from
// Sedgewick, Algorithms, Second Edition, p. 482, with two adaptations.
//
// First, a closure function (fn.IsClosure()) cannot be
// the root of a connected component. Refusing to use it as a root forces
// it into the component of the function in which it appears.  This is
// more convenient for escape analysis.
//
// Second, each function becomes two virtual nodes in the graph,
// with numbers n and n+1. We record the function's node number as n
// but search from node n+1. If the search tells us that the component
// number (minVisitGen) is n+1, we know that this is a trivial component: one function
// plus its closures. If the search tells us that the component number is
// n, then there was a path from node n+1 back to node n, meaning that
// the function set is mutually recursive. The escape analysis can be
// more precise when analyzing a single non-recursive function than
// when analyzing a set of mutually recursive functions.

type bottomUpVisitor struct {
	analyze  func([]*Func, bool)
	visitgen uint32
	nodeID   map[*Func]uint32
	stack    []*Func
}

// VisitFuncsBottomUp invokes analyze on the ODCLFUNC nodes listed in list.
// It calls analyze with successive groups of functions, working from
// the bottom of the call graph upward. Each time analyze is called with
// a list of functions, every function on that list only calls other functions
// on the list or functions that have been passed in previous invocations of
// analyze. Closures appear in the same list as their outer functions.
// The lists are as short as possible while preserving those requirements.
// (In a typical program, many invocations of analyze will be passed just
// a single function.) The boolean argument 'recursive' passed to analyze
// specifies whether the functions on the list are mutually recursive.
// If recursive is false, the list consists of only a single function and its closures.
// If recursive is true, the list may still contain only a single function,
// if that function is itself recursive.
func VisitFuncsBottomUp(list []*Func, analyze func(list []*Func, recursive bool)) {
	var v bottomUpVisitor
	v.analyze = analyze
	v.nodeID = make(map[*Func]uint32)
	for _, n := range list {
		if !n.IsClosure() {
			v.visit(n)
		}
	}
}

func (v *bottomUpVisitor) visit(n *Func) uint32 {
	if id := v.nodeID[n]; id > 0 {
		// already visited
		return id
	}

	v.visitgen++
	id := v.visitgen
	v.nodeID[n] = id
	v.visitgen++
	minVisitGen := v.visitgen
	v.stack = append(v.stack, n)

	do := func(defn Node) {
		if defn != nil {
			if m := v.visit(defn.(*Func)); m < minVisitGen {
				minVisitGen = m
			}
		}
	}

	Visit(n, func(n Node) {
		switch n.Op() {
		case ONAME:
			if n := n.(*Name); n.Class == PFUNC {
				do(n.Defn)
			}
		case ODOTMETH, OMETHVALUE, OMETHEXPR:
			if fn := MethodExprName(n); fn != nil {
				do(fn.Defn)
			}
		case OCLOSURE:
			n := n.(*ClosureExpr)
			do(n.Func)
		}
	})

	if (minVisitGen == id || minVisitGen == id+1) && !n.IsClosure() {
		// This node is the root of a strongly connected component.

		// The original minVisitGen was id+1. If the bottomUpVisitor found its way
		// back to id, then this block is a set of mutually recursive functions.
		// Otherwise, it's just a lone function that does not recurse.
		recursive := minVisitGen == id

		// Remove connected component from stack and mark v.nodeID so that future
		// visits return a large number, which will not affect the caller's min.
		var i int
		for i = len(v.stack) - 1; i >= 0; i-- {
			x := v.stack[i]
			v.nodeID[x] = ^uint32(0)
			if x == n {
				break
			}
		}
		block := v.stack[i:]
		// Call analyze on this set of functions.
		v.stack = v.stack[:i]
		v.analyze(block, recursive)
	}

	return minVisitGen
}
