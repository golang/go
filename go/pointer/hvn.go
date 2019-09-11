// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file implements Hash-Value Numbering (HVN), a pre-solver
// constraint optimization described in Hardekopf & Lin, SAS'07 (see
// doc.go) that analyses the graph topology to determine which sets of
// variables are "pointer equivalent" (PE), i.e. must have identical
// points-to sets in the solution.
//
// A separate ("offline") graph is constructed.  Its nodes are those of
// the main-graph, plus an additional node *X for each pointer node X.
// With this graph we can reason about the unknown points-to set of
// dereferenced pointers.  (We do not generalize this to represent
// unknown fields x->f, perhaps because such fields would be numerous,
// though it might be worth an experiment.)
//
// Nodes whose points-to relations are not entirely captured by the
// graph are marked as "indirect": the *X nodes, the parameters of
// address-taken functions (which includes all functions in method
// sets), or nodes updated by the solver rules for reflection, etc.
//
// All addr (y=&x) nodes are initially assigned a pointer-equivalence
// (PE) label equal to x's nodeid in the main graph.  (These are the
// only PE labels that are less than len(a.nodes).)
//
// All offsetAddr (y=&x.f) constraints are initially assigned a PE
// label; such labels are memoized, keyed by (x, f), so that equivalent
// nodes y as assigned the same label.
//
// Then we process each strongly connected component (SCC) of the graph
// in topological order, assigning it a PE label based on the set P of
// PE labels that flow to it from its immediate dependencies.
//
// If any node in P is "indirect", the entire SCC is assigned a fresh PE
// label.  Otherwise:
//
// |P|=0  if P is empty, all nodes in the SCC are non-pointers (e.g.
//        uninitialized variables, or formal params of dead functions)
//        and the SCC is assigned the PE label of zero.
//
// |P|=1  if P is a singleton, the SCC is assigned the same label as the
//        sole element of P.
//
// |P|>1  if P contains multiple labels, a unique label representing P is
//        invented and recorded in an hash table, so that other
//        equivalent SCCs may also be assigned this label, akin to
//        conventional hash-value numbering in a compiler.
//
// Finally, a renumbering is computed such that each node is replaced by
// the lowest-numbered node with the same PE label.  All constraints are
// renumbered, and any resulting duplicates are eliminated.
//
// The only nodes that are not renumbered are the objects x in addr
// (y=&x) constraints, since the ids of these nodes (and fields derived
// from them via offsetAddr rules) are the elements of all points-to
// sets, so they must remain as they are if we want the same solution.
//
// The solverStates (node.solve) for nodes in the same equivalence class
// are linked together so that all nodes in the class have the same
// solution.  This avoids the need to renumber nodeids buried in
// Queries, cgnodes, etc (like (*analysis).renumber() does) since only
// the solution is needed.
//
// The result of HVN is that the number of distinct nodes and
// constraints is reduced, but the solution is identical (almost---see
// CROSS-CHECK below).  In particular, both linear and cyclic chains of
// copies are each replaced by a single node.
//
// Nodes and constraints created "online" (e.g. while solving reflection
// constraints) are not subject to this optimization.
//
// PERFORMANCE
//
// In two benchmarks (guru and godoc), HVN eliminates about two thirds
// of nodes, the majority accounted for by non-pointers: nodes of
// non-pointer type, pointers that remain nil, formal parameters of dead
// functions, nodes of untracked types, etc.  It also reduces the number
// of constraints, also by about two thirds, and the solving time by
// 30--42%, although we must pay about 15% for the running time of HVN
// itself.  The benefit is greater for larger applications.
//
// There are many possible optimizations to improve the performance:
// * Use fewer than 1:1 onodes to main graph nodes: many of the onodes
//   we create are not needed.
// * HU (HVN with Union---see paper): coalesce "union" peLabels when
//   their expanded-out sets are equal.
// * HR (HVN with deReference---see paper): this will require that we
//   apply HVN until fixed point, which may need more bookkeeping of the
//   correspondence of main nodes to onodes.
// * Location Equivalence (see paper): have points-to sets contain not
//   locations but location-equivalence class labels, each representing
//   a set of locations.
// * HVN with field-sensitive ref: model each of the fields of a
//   pointer-to-struct.
//
// CROSS-CHECK
//
// To verify the soundness of the optimization, when the
// debugHVNCrossCheck option is enabled, we run the solver twice, once
// before and once after running HVN, dumping the solution to disk, and
// then we compare the results.  If they are not identical, the analysis
// panics.
//
// The solution dumped to disk includes only the N*N submatrix of the
// complete solution where N is the number of nodes after generation.
// In other words, we ignore pointer variables and objects created by
// the solver itself, since their numbering depends on the solver order,
// which is affected by the optimization.  In any case, that's the only
// part the client cares about.
//
// The cross-check is too strict and may fail spuriously.  Although the
// H&L paper describing HVN states that the solutions obtained should be
// identical, this is not the case in practice because HVN can collapse
// cycles involving *p even when pts(p)={}.  Consider this example
// distilled from testdata/hello.go:
//
//	var x T
//	func f(p **T) {
//		t0 = *p
//		...
//		t1 = φ(t0, &x)
//		*p = t1
//	}
//
// If f is dead code, we get:
// 	unoptimized:  pts(p)={} pts(t0)={} pts(t1)={&x}
// 	optimized:    pts(p)={} pts(t0)=pts(t1)=pts(*p)={&x}
//
// It's hard to argue that this is a bug: the result is sound and the
// loss of precision is inconsequential---f is dead code, after all.
// But unfortunately it limits the usefulness of the cross-check since
// failures must be carefully analyzed.  Ben Hardekopf suggests (in
// personal correspondence) some approaches to mitigating it:
//
// 	If there is a node with an HVN points-to set that is a superset
// 	of the NORM points-to set, then either it's a bug or it's a
// 	result of this issue. If it's a result of this issue, then in
// 	the offline constraint graph there should be a REF node inside
// 	some cycle that reaches this node, and in the NORM solution the
// 	pointer being dereferenced by that REF node should be the empty
// 	set. If that isn't true then this is a bug. If it is true, then
// 	you can further check that in the NORM solution the "extra"
// 	points-to info in the HVN solution does in fact come from that
// 	purported cycle (if it doesn't, then this is still a bug). If
// 	you're doing the further check then you'll need to do it for
// 	each "extra" points-to element in the HVN points-to set.
//
// 	There are probably ways to optimize these checks by taking
// 	advantage of graph properties. For example, extraneous points-to
// 	info will flow through the graph and end up in many
// 	nodes. Rather than checking every node with extra info, you
// 	could probably work out the "origin point" of the extra info and
// 	just check there. Note that the check in the first bullet is
// 	looking for soundness bugs, while the check in the second bullet
// 	is looking for precision bugs; depending on your needs, you may
// 	care more about one than the other.
//
// which we should evaluate.  The cross-check is nonetheless invaluable
// for all but one of the programs in the pointer_test suite.

import (
	"fmt"
	"go/types"
	"io"
	"reflect"

	"golang.org/x/tools/container/intsets"
)

// A peLabel is a pointer-equivalence label: two nodes with the same
// peLabel have identical points-to solutions.
//
// The numbers are allocated consecutively like so:
// 	0	not a pointer
//	1..N-1	addrConstraints (equals the constraint's .src field, hence sparse)
//	...	offsetAddr constraints
//	...	SCCs (with indirect nodes or multiple inputs)
//
// Each PE label denotes a set of pointers containing a single addr, a
// single offsetAddr, or some set of other PE labels.
//
type peLabel int

type hvn struct {
	a        *analysis
	N        int                // len(a.nodes) immediately after constraint generation
	log      io.Writer          // (optional) log of HVN lemmas
	onodes   []*onode           // nodes of the offline graph
	label    peLabel            // the next available PE label
	hvnLabel map[string]peLabel // hash-value numbering (PE label) for each set of onodeids
	stack    []onodeid          // DFS stack
	index    int32              // next onode.index, from Tarjan's SCC algorithm

	// For each distinct offsetAddrConstraint (src, offset) pair,
	// offsetAddrLabels records a unique PE label >= N.
	offsetAddrLabels map[offsetAddr]peLabel
}

// The index of an node in the offline graph.
// (Currently the first N align with the main nodes,
// but this may change with HRU.)
type onodeid uint32

// An onode is a node in the offline constraint graph.
// (Where ambiguous, members of analysis.nodes are referred to as
// "main graph" nodes.)
//
// Edges in the offline constraint graph (edges and implicit) point to
// the source, i.e. against the flow of values: they are dependencies.
// Implicit edges are used for SCC computation, but not for gathering
// incoming labels.
//
type onode struct {
	rep onodeid // index of representative of SCC in offline constraint graph

	edges    intsets.Sparse // constraint edges X-->Y (this onode is X)
	implicit intsets.Sparse // implicit edges *X-->*Y (this onode is X)
	peLabels intsets.Sparse // set of peLabels are pointer-equivalent to this one
	indirect bool           // node has points-to relations not represented in graph

	// Tarjan's SCC algorithm
	index, lowlink int32 // Tarjan numbering
	scc            int32 // -ve => on stack; 0 => unvisited; +ve => node is root of a found SCC
}

type offsetAddr struct {
	ptr    nodeid
	offset uint32
}

// nextLabel issues the next unused pointer-equivalence label.
func (h *hvn) nextLabel() peLabel {
	h.label++
	return h.label
}

// ref(X) returns the index of the onode for *X.
func (h *hvn) ref(id onodeid) onodeid {
	return id + onodeid(len(h.a.nodes))
}

// hvn computes pointer-equivalence labels (peLabels) using the Hash-based
// Value Numbering (HVN) algorithm described in Hardekopf & Lin, SAS'07.
//
func (a *analysis) hvn() {
	start("HVN")

	if a.log != nil {
		fmt.Fprintf(a.log, "\n\n==== Pointer equivalence optimization\n\n")
	}

	h := hvn{
		a:                a,
		N:                len(a.nodes),
		log:              a.log,
		hvnLabel:         make(map[string]peLabel),
		offsetAddrLabels: make(map[offsetAddr]peLabel),
	}

	if h.log != nil {
		fmt.Fprintf(h.log, "\nCreating offline graph nodes...\n")
	}

	// Create offline nodes.  The first N nodes correspond to main
	// graph nodes; the next N are their corresponding ref() nodes.
	h.onodes = make([]*onode, 2*h.N)
	for id := range a.nodes {
		id := onodeid(id)
		h.onodes[id] = &onode{}
		h.onodes[h.ref(id)] = &onode{indirect: true}
	}

	// Each node initially represents just itself.
	for id, o := range h.onodes {
		o.rep = onodeid(id)
	}

	h.markIndirectNodes()

	// Reserve the first N PE labels for addrConstraints.
	h.label = peLabel(h.N)

	// Add offline constraint edges.
	if h.log != nil {
		fmt.Fprintf(h.log, "\nAdding offline graph edges...\n")
	}
	for _, c := range a.constraints {
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "; %s\n", c)
		}
		c.presolve(&h)
	}

	// Find and collapse SCCs.
	if h.log != nil {
		fmt.Fprintf(h.log, "\nFinding SCCs...\n")
	}
	h.index = 1
	for id, o := range h.onodes {
		if id > 0 && o.index == 0 {
			// Start depth-first search at each unvisited node.
			h.visit(onodeid(id))
		}
	}

	// Dump the solution
	// (NB: somewhat redundant with logging from simplify().)
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\nPointer equivalences:\n")
		for id, o := range h.onodes {
			if id == 0 {
				continue
			}
			if id == int(h.N) {
				fmt.Fprintf(h.log, "---\n")
			}
			fmt.Fprintf(h.log, "o%d\t", id)
			if o.rep != onodeid(id) {
				fmt.Fprintf(h.log, "rep=o%d", o.rep)
			} else {
				fmt.Fprintf(h.log, "p%d", o.peLabels.Min())
				if o.indirect {
					fmt.Fprint(h.log, " indirect")
				}
			}
			fmt.Fprintln(h.log)
		}
	}

	// Simplify the main constraint graph
	h.simplify()

	a.showCounts()

	stop("HVN")
}

// ---- constraint-specific rules ----

// dst := &src
func (c *addrConstraint) presolve(h *hvn) {
	// Each object (src) is an initial PE label.
	label := peLabel(c.src) // label < N
	if debugHVNVerbose && h.log != nil {
		// duplicate log messages are possible
		fmt.Fprintf(h.log, "\tcreate p%d: {&n%d}\n", label, c.src)
	}
	odst := onodeid(c.dst)
	osrc := onodeid(c.src)

	// Assign dst this label.
	h.onodes[odst].peLabels.Insert(int(label))
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d has p%d\n", odst, label)
	}

	h.addImplicitEdge(h.ref(odst), osrc) // *dst ~~> src.
}

// dst = src
func (c *copyConstraint) presolve(h *hvn) {
	odst := onodeid(c.dst)
	osrc := onodeid(c.src)
	h.addEdge(odst, osrc)                       //  dst -->  src
	h.addImplicitEdge(h.ref(odst), h.ref(osrc)) // *dst ~~> *src
}

// dst = *src + offset
func (c *loadConstraint) presolve(h *hvn) {
	odst := onodeid(c.dst)
	osrc := onodeid(c.src)
	if c.offset == 0 {
		h.addEdge(odst, h.ref(osrc)) // dst --> *src
	} else {
		// We don't interpret load-with-offset, e.g. results
		// of map value lookup, R-block of dynamic call, slice
		// copy/append, reflection.
		h.markIndirect(odst, "load with offset")
	}
}

// *dst + offset = src
func (c *storeConstraint) presolve(h *hvn) {
	odst := onodeid(c.dst)
	osrc := onodeid(c.src)
	if c.offset == 0 {
		h.onodes[h.ref(odst)].edges.Insert(int(osrc)) // *dst --> src
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "\to%d --> o%d\n", h.ref(odst), osrc)
		}
	} else {
		// We don't interpret store-with-offset.
		// See discussion of soundness at markIndirectNodes.
	}
}

// dst = &src.offset
func (c *offsetAddrConstraint) presolve(h *hvn) {
	// Give each distinct (addr, offset) pair a fresh PE label.
	// The cache performs CSE, effectively.
	key := offsetAddr{c.src, c.offset}
	label, ok := h.offsetAddrLabels[key]
	if !ok {
		label = h.nextLabel()
		h.offsetAddrLabels[key] = label
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "\tcreate p%d: {&n%d.#%d}\n",
				label, c.src, c.offset)
		}
	}

	// Assign dst this label.
	h.onodes[c.dst].peLabels.Insert(int(label))
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d has p%d\n", c.dst, label)
	}
}

// dst = src.(typ)  where typ is an interface
func (c *typeFilterConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.dst), "typeFilter result")
}

// dst = src.(typ)  where typ is concrete
func (c *untagConstraint) presolve(h *hvn) {
	odst := onodeid(c.dst)
	for end := odst + onodeid(h.a.sizeof(c.typ)); odst < end; odst++ {
		h.markIndirect(odst, "untag result")
	}
}

// dst = src.method(c.params...)
func (c *invokeConstraint) presolve(h *hvn) {
	// All methods are address-taken functions, so
	// their formal P-blocks were already marked indirect.

	// Mark the caller's targets node as indirect.
	sig := c.method.Type().(*types.Signature)
	id := c.params
	h.markIndirect(onodeid(c.params), "invoke targets node")
	id++

	id += nodeid(h.a.sizeof(sig.Params()))

	// Mark the caller's R-block as indirect.
	end := id + nodeid(h.a.sizeof(sig.Results()))
	for id < end {
		h.markIndirect(onodeid(id), "invoke R-block")
		id++
	}
}

// markIndirectNodes marks as indirect nodes whose points-to relations
// are not entirely captured by the offline graph, including:
//
//    (a) All address-taken nodes (including the following nodes within
//        the same object).  This is described in the paper.
//
// The most subtle cause of indirect nodes is the generation of
// store-with-offset constraints since the offline graph doesn't
// represent them.  A global audit of constraint generation reveals the
// following uses of store-with-offset:
//
//    (b) genDynamicCall, for P-blocks of dynamically called functions,
//        to which dynamic copy edges will be added to them during
//        solving: from storeConstraint for standalone functions,
//        and from invokeConstraint for methods.
//        All such P-blocks must be marked indirect.
//    (c) MakeUpdate, to update the value part of a map object.
//        All MakeMap objects's value parts must be marked indirect.
//    (d) copyElems, to update the destination array.
//        All array elements must be marked indirect.
//
// Not all indirect marking happens here.  ref() nodes are marked
// indirect at construction, and each constraint's presolve() method may
// mark additional nodes.
//
func (h *hvn) markIndirectNodes() {
	// (a) all address-taken nodes, plus all nodes following them
	//     within the same object, since these may be indirectly
	//     stored or address-taken.
	for _, c := range h.a.constraints {
		if c, ok := c.(*addrConstraint); ok {
			start := h.a.enclosingObj(c.src)
			end := start + nodeid(h.a.nodes[start].obj.size)
			for id := c.src; id < end; id++ {
				h.markIndirect(onodeid(id), "A-T object")
			}
		}
	}

	// (b) P-blocks of all address-taken functions.
	for id := 0; id < h.N; id++ {
		obj := h.a.nodes[id].obj

		// TODO(adonovan): opt: if obj.cgn.fn is a method and
		// obj.cgn is not its shared contour, this is an
		// "inlined" static method call.  We needn't consider it
		// address-taken since no invokeConstraint will affect it.

		if obj != nil && obj.flags&otFunction != 0 && h.a.atFuncs[obj.cgn.fn] {
			// address-taken function
			if debugHVNVerbose && h.log != nil {
				fmt.Fprintf(h.log, "n%d is address-taken: %s\n", id, obj.cgn.fn)
			}
			h.markIndirect(onodeid(id), "A-T func identity")
			id++
			sig := obj.cgn.fn.Signature
			psize := h.a.sizeof(sig.Params())
			if sig.Recv() != nil {
				psize += h.a.sizeof(sig.Recv().Type())
			}
			for end := id + int(psize); id < end; id++ {
				h.markIndirect(onodeid(id), "A-T func P-block")
			}
			id--
			continue
		}
	}

	// (c) all map objects' value fields.
	for _, id := range h.a.mapValues {
		h.markIndirect(onodeid(id), "makemap.value")
	}

	// (d) all array element objects.
	// TODO(adonovan): opt: can we do better?
	for id := 0; id < h.N; id++ {
		// Identity node for an object of array type?
		if tArray, ok := h.a.nodes[id].typ.(*types.Array); ok {
			// Mark the array element nodes indirect.
			// (Skip past the identity field.)
			for range h.a.flatten(tArray.Elem()) {
				id++
				h.markIndirect(onodeid(id), "array elem")
			}
		}
	}
}

func (h *hvn) markIndirect(oid onodeid, comment string) {
	h.onodes[oid].indirect = true
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d is indirect: %s\n", oid, comment)
	}
}

// Adds an edge dst-->src.
// Note the unusual convention: edges are dependency (contraflow) edges.
func (h *hvn) addEdge(odst, osrc onodeid) {
	h.onodes[odst].edges.Insert(int(osrc))
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d --> o%d\n", odst, osrc)
	}
}

func (h *hvn) addImplicitEdge(odst, osrc onodeid) {
	h.onodes[odst].implicit.Insert(int(osrc))
	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d ~~> o%d\n", odst, osrc)
	}
}

// visit implements the depth-first search of Tarjan's SCC algorithm.
// Precondition: x is canonical.
func (h *hvn) visit(x onodeid) {
	h.checkCanonical(x)
	xo := h.onodes[x]
	xo.index = h.index
	xo.lowlink = h.index
	h.index++

	h.stack = append(h.stack, x) // push
	assert(xo.scc == 0, "node revisited")
	xo.scc = -1

	var deps []int
	deps = xo.edges.AppendTo(deps)
	deps = xo.implicit.AppendTo(deps)

	for _, y := range deps {
		// Loop invariant: x is canonical.

		y := h.find(onodeid(y))

		if x == y {
			continue // nodes already coalesced
		}

		xo := h.onodes[x]
		yo := h.onodes[y]

		switch {
		case yo.scc > 0:
			// y is already a collapsed SCC

		case yo.scc < 0:
			// y is on the stack, and thus in the current SCC.
			if yo.index < xo.lowlink {
				xo.lowlink = yo.index
			}

		default:
			// y is unvisited; visit it now.
			h.visit(y)
			// Note: x and y are now non-canonical.

			x = h.find(onodeid(x))

			if yo.lowlink < xo.lowlink {
				xo.lowlink = yo.lowlink
			}
		}
	}
	h.checkCanonical(x)

	// Is x the root of an SCC?
	if xo.lowlink == xo.index {
		// Coalesce all nodes in the SCC.
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "scc o%d\n", x)
		}
		for {
			// Pop y from stack.
			i := len(h.stack) - 1
			y := h.stack[i]
			h.stack = h.stack[:i]

			h.checkCanonical(x)
			xo := h.onodes[x]
			h.checkCanonical(y)
			yo := h.onodes[y]

			if xo == yo {
				// SCC is complete.
				xo.scc = 1
				h.labelSCC(x)
				break
			}
			h.coalesce(x, y)
		}
	}
}

// Precondition: x is canonical.
func (h *hvn) labelSCC(x onodeid) {
	h.checkCanonical(x)
	xo := h.onodes[x]
	xpe := &xo.peLabels

	// All indirect nodes get new labels.
	if xo.indirect {
		label := h.nextLabel()
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "\tcreate p%d: indirect SCC\n", label)
			fmt.Fprintf(h.log, "\to%d has p%d\n", x, label)
		}

		// Remove pre-labeling, in case a direct pre-labeled node was
		// merged with an indirect one.
		xpe.Clear()
		xpe.Insert(int(label))

		return
	}

	// Invariant: all peLabels sets are non-empty.
	// Those that are logically empty contain zero as their sole element.
	// No other sets contains zero.

	// Find all labels coming in to the coalesced SCC node.
	for _, y := range xo.edges.AppendTo(nil) {
		y := h.find(onodeid(y))
		if y == x {
			continue // already coalesced
		}
		ype := &h.onodes[y].peLabels
		if debugHVNVerbose && h.log != nil {
			fmt.Fprintf(h.log, "\tedge from o%d = %s\n", y, ype)
		}

		if ype.IsEmpty() {
			if debugHVNVerbose && h.log != nil {
				fmt.Fprintf(h.log, "\tnode has no PE label\n")
			}
		}
		assert(!ype.IsEmpty(), "incoming node has no PE label")

		if ype.Has(0) {
			// {0} represents a non-pointer.
			assert(ype.Len() == 1, "PE set contains {0, ...}")
		} else {
			xpe.UnionWith(ype)
		}
	}

	switch xpe.Len() {
	case 0:
		// SCC has no incoming non-zero PE labels: it is a non-pointer.
		xpe.Insert(0)

	case 1:
		// already a singleton

	default:
		// SCC has multiple incoming non-zero PE labels.
		// Find the canonical label representing this set.
		// We use String() as a fingerprint consistent with Equals().
		key := xpe.String()
		label, ok := h.hvnLabel[key]
		if !ok {
			label = h.nextLabel()
			if debugHVNVerbose && h.log != nil {
				fmt.Fprintf(h.log, "\tcreate p%d: union %s\n", label, xpe.String())
			}
			h.hvnLabel[key] = label
		}
		xpe.Clear()
		xpe.Insert(int(label))
	}

	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\to%d has p%d\n", x, xpe.Min())
	}
}

// coalesce combines two nodes in the offline constraint graph.
// Precondition: x and y are canonical.
func (h *hvn) coalesce(x, y onodeid) {
	xo := h.onodes[x]
	yo := h.onodes[y]

	// x becomes y's canonical representative.
	yo.rep = x

	if debugHVNVerbose && h.log != nil {
		fmt.Fprintf(h.log, "\tcoalesce o%d into o%d\n", y, x)
	}

	// x accumulates y's edges.
	xo.edges.UnionWith(&yo.edges)
	yo.edges.Clear()

	// x accumulates y's implicit edges.
	xo.implicit.UnionWith(&yo.implicit)
	yo.implicit.Clear()

	// x accumulates y's pointer-equivalence labels.
	xo.peLabels.UnionWith(&yo.peLabels)
	yo.peLabels.Clear()

	// x accumulates y's indirect flag.
	if yo.indirect {
		xo.indirect = true
	}
}

// simplify computes a degenerate renumbering of nodeids from the PE
// labels assigned by the hvn, and uses it to simplify the main
// constraint graph, eliminating non-pointer nodes and duplicate
// constraints.
//
func (h *hvn) simplify() {
	// canon maps each peLabel to its canonical main node.
	canon := make([]nodeid, h.label)
	for i := range canon {
		canon[i] = nodeid(h.N) // indicates "unset"
	}

	// mapping maps each main node index to the index of the canonical node.
	mapping := make([]nodeid, len(h.a.nodes))

	for id := range h.a.nodes {
		id := nodeid(id)
		if id == 0 {
			canon[0] = 0
			mapping[0] = 0
			continue
		}
		oid := h.find(onodeid(id))
		peLabels := &h.onodes[oid].peLabels
		assert(peLabels.Len() == 1, "PE class is not a singleton")
		label := peLabel(peLabels.Min())

		canonId := canon[label]
		if canonId == nodeid(h.N) {
			// id becomes the representative of the PE label.
			canonId = id
			canon[label] = canonId

			if h.a.log != nil {
				fmt.Fprintf(h.a.log, "\tpts(n%d) is canonical : \t(%s)\n",
					id, h.a.nodes[id].typ)
			}

		} else {
			// Link the solver states for the two nodes.
			assert(h.a.nodes[canonId].solve != nil, "missing solver state")
			h.a.nodes[id].solve = h.a.nodes[canonId].solve

			if h.a.log != nil {
				// TODO(adonovan): debug: reorganize the log so it prints
				// one line:
				// 	pe y = x1, ..., xn
				// for each canonical y.  Requires allocation.
				fmt.Fprintf(h.a.log, "\tpts(n%d) = pts(n%d) : %s\n",
					id, canonId, h.a.nodes[id].typ)
			}
		}

		mapping[id] = canonId
	}

	// Renumber the constraints, eliminate duplicates, and eliminate
	// any containing non-pointers (n0).
	addrs := make(map[addrConstraint]bool)
	copys := make(map[copyConstraint]bool)
	loads := make(map[loadConstraint]bool)
	stores := make(map[storeConstraint]bool)
	offsetAddrs := make(map[offsetAddrConstraint]bool)
	untags := make(map[untagConstraint]bool)
	typeFilters := make(map[typeFilterConstraint]bool)
	invokes := make(map[invokeConstraint]bool)

	nbefore := len(h.a.constraints)
	cc := h.a.constraints[:0] // in-situ compaction
	for _, c := range h.a.constraints {
		// Renumber.
		switch c := c.(type) {
		case *addrConstraint:
			// Don't renumber c.src since it is the label of
			// an addressable object and will appear in PT sets.
			c.dst = mapping[c.dst]
		default:
			c.renumber(mapping)
		}

		if c.ptr() == 0 {
			continue // skip: constraint attached to non-pointer
		}

		var dup bool
		switch c := c.(type) {
		case *addrConstraint:
			_, dup = addrs[*c]
			addrs[*c] = true

		case *copyConstraint:
			if c.src == c.dst {
				continue // skip degenerate copies
			}
			if c.src == 0 {
				continue // skip copy from non-pointer
			}
			_, dup = copys[*c]
			copys[*c] = true

		case *loadConstraint:
			if c.src == 0 {
				continue // skip load from non-pointer
			}
			_, dup = loads[*c]
			loads[*c] = true

		case *storeConstraint:
			if c.src == 0 {
				continue // skip store from non-pointer
			}
			_, dup = stores[*c]
			stores[*c] = true

		case *offsetAddrConstraint:
			if c.src == 0 {
				continue // skip offset from non-pointer
			}
			_, dup = offsetAddrs[*c]
			offsetAddrs[*c] = true

		case *untagConstraint:
			if c.src == 0 {
				continue // skip untag of non-pointer
			}
			_, dup = untags[*c]
			untags[*c] = true

		case *typeFilterConstraint:
			if c.src == 0 {
				continue // skip filter of non-pointer
			}
			_, dup = typeFilters[*c]
			typeFilters[*c] = true

		case *invokeConstraint:
			if c.params == 0 {
				panic("non-pointer invoke.params")
			}
			if c.iface == 0 {
				continue // skip invoke on non-pointer
			}
			_, dup = invokes[*c]
			invokes[*c] = true

		default:
			// We don't bother de-duping advanced constraints
			// (e.g. reflection) since they are uncommon.

			// Eliminate constraints containing non-pointer nodeids.
			//
			// We use reflection to find the fields to avoid
			// adding yet another method to constraint.
			//
			// TODO(adonovan): experiment with a constraint
			// method that returns a slice of pointers to
			// nodeids fields to enable uniform iteration;
			// the renumber() method could be removed and
			// implemented using the new one.
			//
			// TODO(adonovan): opt: this is unsound since
			// some constraints still have an effect if one
			// of the operands is zero: rVCall, rVMapIndex,
			// rvSetMapIndex.  Handle them specially.
			rtNodeid := reflect.TypeOf(nodeid(0))
			x := reflect.ValueOf(c).Elem()
			for i, nf := 0, x.NumField(); i < nf; i++ {
				f := x.Field(i)
				if f.Type() == rtNodeid {
					if f.Uint() == 0 {
						dup = true // skip it
						break
					}
				}
			}
		}
		if dup {
			continue // skip duplicates
		}

		cc = append(cc, c)
	}
	h.a.constraints = cc

	if h.log != nil {
		fmt.Fprintf(h.log, "#constraints: was %d, now %d\n", nbefore, len(h.a.constraints))
	}
}

// find returns the canonical onodeid for x.
// (The onodes form a disjoint set forest.)
func (h *hvn) find(x onodeid) onodeid {
	// TODO(adonovan): opt: this is a CPU hotspot.  Try "union by rank".
	xo := h.onodes[x]
	rep := xo.rep
	if rep != x {
		rep = h.find(rep) // simple path compression
		xo.rep = rep
	}
	return rep
}

func (h *hvn) checkCanonical(x onodeid) {
	if debugHVN {
		assert(x == h.find(x), "not canonical")
	}
}

func assert(p bool, msg string) {
	if debugHVN && !p {
		panic("assertion failed: " + msg)
	}
}
