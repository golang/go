// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file implements renumbering, a pre-solver optimization to
// improve the efficiency of the solver's points-to set representation.
//
// TODO(adonovan): rename file "renumber.go"

import "fmt"

// renumber permutes a.nodes so that all nodes within an addressable
// object appear before all non-addressable nodes, maintaining the
// order of nodes within the same object (as required by offsetAddr).
//
// renumber must update every nodeid in the analysis (constraints,
// Pointers, callgraph, etc) to reflect the new ordering.
//
// This is an optimisation to increase the locality and efficiency of
// sparse representations of points-to sets.  (Typically only about
// 20% of nodes are within an object.)
//
// NB: nodes added during solving (e.g. for reflection, SetFinalizer)
// will be appended to the end.
//
// Renumbering makes the PTA log inscrutable.  To aid debugging, later
// phases (e.g. HVN) must not rely on it having occurred.
//
func (a *analysis) renumber() {
	if a.log != nil {
		fmt.Fprintf(a.log, "\n\n==== Renumbering\n\n")
	}

	N := nodeid(len(a.nodes))
	newNodes := make([]*node, N)
	renumbering := make([]nodeid, N) // maps old to new

	var i, j nodeid

	// The zero node is special.
	newNodes[j] = a.nodes[i]
	renumbering[i] = j
	i++
	j++

	// Pass 1: object nodes.
	for i < N {
		obj := a.nodes[i].obj
		if obj == nil {
			i++
			continue
		}

		end := i + nodeid(obj.size)
		for i < end {
			newNodes[j] = a.nodes[i]
			renumbering[i] = j
			i++
			j++
		}
	}
	nobj := j

	// Pass 2: non-object nodes.
	for i = 1; i < N; {
		obj := a.nodes[i].obj
		if obj != nil {
			i += nodeid(obj.size)
			continue
		}

		newNodes[j] = a.nodes[i]
		renumbering[i] = j
		i++
		j++
	}

	if j != N {
		panic(fmt.Sprintf("internal error: j=%d, N=%d", j, N))
	}

	// Log the remapping table.
	if a.log != nil {
		fmt.Fprintf(a.log, "Renumbering nodes to improve density:\n")
		fmt.Fprintf(a.log, "(%d object nodes of %d total)\n", nobj, N)
		for old, new := range renumbering {
			fmt.Fprintf(a.log, "\tn%d -> n%d\n", old, new)
		}
	}

	// Now renumber all existing nodeids to use the new node permutation.
	// It is critical that all reachable nodeids are accounted for!

	// Renumber nodeids in queried Pointers.
	for v, ptr := range a.result.Queries {
		ptr.n = renumbering[ptr.n]
		a.result.Queries[v] = ptr
	}
	for v, ptr := range a.result.IndirectQueries {
		ptr.n = renumbering[ptr.n]
		a.result.IndirectQueries[v] = ptr
	}
	for _, queries := range a.config.extendedQueries {
		for _, query := range queries {
			if query.ptr != nil {
				query.ptr.n = renumbering[query.ptr.n]
			}
		}
	}

	// Renumber nodeids in global objects.
	for v, id := range a.globalobj {
		a.globalobj[v] = renumbering[id]
	}

	// Renumber nodeids in constraints.
	for _, c := range a.constraints {
		c.renumber(renumbering)
	}

	// Renumber nodeids in the call graph.
	for _, cgn := range a.cgnodes {
		cgn.obj = renumbering[cgn.obj]
		for _, site := range cgn.sites {
			site.targets = renumbering[site.targets]
		}
	}

	a.nodes = newNodes
}
