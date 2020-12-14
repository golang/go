// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"container/heap"
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// Package initialization
//
// Here we implement the algorithm for ordering package-level variable
// initialization. The spec is written in terms of variable
// initialization, but multiple variables initialized by a single
// assignment are handled together, so here we instead focus on
// ordering initialization assignments. Conveniently, this maps well
// to how we represent package-level initializations using the Node
// AST.
//
// Assignments are in one of three phases: NotStarted, Pending, or
// Done. For assignments in the Pending phase, we use Xoffset to
// record the number of unique variable dependencies whose
// initialization assignment is not yet Done. We also maintain a
// "blocking" map that maps assignments back to all of the assignments
// that depend on it.
//
// For example, for an initialization like:
//
//     var x = f(a, b, b)
//     var a, b = g()
//
// the "x = f(a, b, b)" assignment depends on two variables (a and b),
// so its Xoffset will be 2. Correspondingly, the "a, b = g()"
// assignment's "blocking" entry will have two entries back to x's
// assignment.
//
// Logically, initialization works by (1) taking all NotStarted
// assignments, calculating their dependencies, and marking them
// Pending; (2) adding all Pending assignments with Xoffset==0 to a
// "ready" priority queue (ordered by variable declaration position);
// and (3) iteratively processing the next Pending assignment from the
// queue, decreasing the Xoffset of assignments it's blocking, and
// adding them to the queue if decremented to 0.
//
// As an optimization, we actually apply each of these three steps for
// each assignment. This yields the same order, but keeps queue size
// down and thus also heap operation costs.

// Static initialization phase.
// These values are stored in two bits in Node.flags.
const (
	InitNotStarted = iota
	InitDone
	InitPending
)

type InitOrder struct {
	// blocking maps initialization assignments to the assignments
	// that depend on it.
	blocking map[ir.Node][]ir.Node

	// ready is the queue of Pending initialization assignments
	// that are ready for initialization.
	ready declOrder
}

// initOrder computes initialization order for a list l of
// package-level declarations (in declaration order) and outputs the
// corresponding list of statements to include in the init() function
// body.
func initOrder(l []ir.Node) []ir.Node {
	s := InitSchedule{
		initplans: make(map[ir.Node]*InitPlan),
		inittemps: make(map[ir.Node]ir.Node),
	}
	o := InitOrder{
		blocking: make(map[ir.Node][]ir.Node),
	}

	// Process all package-level assignment in declaration order.
	for _, n := range l {
		switch n.Op() {
		case ir.OAS, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
			o.processAssign(n)
			o.flushReady(s.staticInit)
		case ir.ODCLCONST, ir.ODCLFUNC, ir.ODCLTYPE:
			// nop
		default:
			base.Fatalf("unexpected package-level statement: %v", n)
		}
	}

	// Check that all assignments are now Done; if not, there must
	// have been a dependency cycle.
	for _, n := range l {
		switch n.Op() {
		case ir.OAS, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
			if n.Initorder() != InitDone {
				// If there have already been errors
				// printed, those errors may have
				// confused us and there might not be
				// a loop. Let the user fix those
				// first.
				base.ExitIfErrors()

				findInitLoopAndExit(firstLHS(n), new([]*ir.Name))
				base.Fatalf("initialization unfinished, but failed to identify loop")
			}
		}
	}

	// Invariant consistency check. If this is non-zero, then we
	// should have found a cycle above.
	if len(o.blocking) != 0 {
		base.Fatalf("expected empty map: %v", o.blocking)
	}

	return s.out
}

func (o *InitOrder) processAssign(n ir.Node) {
	if n.Initorder() != InitNotStarted || n.Offset() != types.BADWIDTH {
		base.Fatalf("unexpected state: %v, %v, %v", n, n.Initorder(), n.Offset())
	}

	n.SetInitorder(InitPending)
	n.SetOffset(0)

	// Compute number of variable dependencies and build the
	// inverse dependency ("blocking") graph.
	for dep := range collectDeps(n, true) {
		defn := dep.Defn
		// Skip dependencies on functions (PFUNC) and
		// variables already initialized (InitDone).
		if dep.Class() != ir.PEXTERN || defn.Initorder() == InitDone {
			continue
		}
		n.SetOffset(n.Offset() + 1)
		o.blocking[defn] = append(o.blocking[defn], n)
	}

	if n.Offset() == 0 {
		heap.Push(&o.ready, n)
	}
}

// flushReady repeatedly applies initialize to the earliest (in
// declaration order) assignment ready for initialization and updates
// the inverse dependency ("blocking") graph.
func (o *InitOrder) flushReady(initialize func(ir.Node)) {
	for o.ready.Len() != 0 {
		n := heap.Pop(&o.ready).(ir.Node)
		if n.Initorder() != InitPending || n.Offset() != 0 {
			base.Fatalf("unexpected state: %v, %v, %v", n, n.Initorder(), n.Offset())
		}

		initialize(n)
		n.SetInitorder(InitDone)
		n.SetOffset(types.BADWIDTH)

		blocked := o.blocking[n]
		delete(o.blocking, n)

		for _, m := range blocked {
			m.SetOffset(m.Offset() - 1)
			if m.Offset() == 0 {
				heap.Push(&o.ready, m)
			}
		}
	}
}

// findInitLoopAndExit searches for an initialization loop involving variable
// or function n. If one is found, it reports the loop as an error and exits.
//
// path points to a slice used for tracking the sequence of
// variables/functions visited. Using a pointer to a slice allows the
// slice capacity to grow and limit reallocations.
func findInitLoopAndExit(n *ir.Name, path *[]*ir.Name) {
	// We implement a simple DFS loop-finding algorithm. This
	// could be faster, but initialization cycles are rare.

	for i, x := range *path {
		if x == n {
			reportInitLoopAndExit((*path)[i:])
			return
		}
	}

	// There might be multiple loops involving n; by sorting
	// references, we deterministically pick the one reported.
	refers := collectDeps(n.Name().Defn, false).Sorted(func(ni, nj *ir.Name) bool {
		return ni.Pos().Before(nj.Pos())
	})

	*path = append(*path, n)
	for _, ref := range refers {
		// Short-circuit variables that were initialized.
		if ref.Class() == ir.PEXTERN && ref.Defn.Initorder() == InitDone {
			continue
		}

		findInitLoopAndExit(ref, path)
	}
	*path = (*path)[:len(*path)-1]
}

// reportInitLoopAndExit reports and initialization loop as an error
// and exits. However, if l is not actually an initialization loop, it
// simply returns instead.
func reportInitLoopAndExit(l []*ir.Name) {
	// Rotate loop so that the earliest variable declaration is at
	// the start.
	i := -1
	for j, n := range l {
		if n.Class() == ir.PEXTERN && (i == -1 || n.Pos().Before(l[i].Pos())) {
			i = j
		}
	}
	if i == -1 {
		// False positive: loop only involves recursive
		// functions. Return so that findInitLoop can continue
		// searching.
		return
	}
	l = append(l[i:], l[:i]...)

	// TODO(mdempsky): Method values are printed as "T.m-fm"
	// rather than "T.m". Figure out how to avoid that.

	var msg bytes.Buffer
	fmt.Fprintf(&msg, "initialization loop:\n")
	for _, n := range l {
		fmt.Fprintf(&msg, "\t%v: %v refers to\n", ir.Line(n), n)
	}
	fmt.Fprintf(&msg, "\t%v: %v", ir.Line(l[0]), l[0])

	base.ErrorfAt(l[0].Pos(), msg.String())
	base.ErrorExit()
}

// collectDeps returns all of the package-level functions and
// variables that declaration n depends on. If transitive is true,
// then it also includes the transitive dependencies of any depended
// upon functions (but not variables).
func collectDeps(n ir.Node, transitive bool) ir.NameSet {
	d := initDeps{transitive: transitive}
	switch n.Op() {
	case ir.OAS:
		d.inspect(n.Right())
	case ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
		d.inspect(n.Rlist().First())
	case ir.ODCLFUNC:
		d.inspectList(n.Body())
	default:
		base.Fatalf("unexpected Op: %v", n.Op())
	}
	return d.seen
}

type initDeps struct {
	transitive bool
	seen       ir.NameSet
}

func (d *initDeps) inspect(n ir.Node)      { ir.Inspect(n, d.visit) }
func (d *initDeps) inspectList(l ir.Nodes) { ir.InspectList(l, d.visit) }

// visit calls foundDep on any package-level functions or variables
// referenced by n, if any.
func (d *initDeps) visit(n ir.Node) bool {
	switch n.Op() {
	case ir.OMETHEXPR:
		d.foundDep(methodExprName(n))
		return false

	case ir.ONAME:
		switch n.Class() {
		case ir.PEXTERN, ir.PFUNC:
			d.foundDep(n.(*ir.Name))
		}

	case ir.OCLOSURE:
		d.inspectList(n.Func().Body())

	case ir.ODOTMETH, ir.OCALLPART:
		d.foundDep(methodExprName(n))
	}

	return true
}

// foundDep records that we've found a dependency on n by adding it to
// seen.
func (d *initDeps) foundDep(n *ir.Name) {
	// Can happen with method expressions involving interface
	// types; e.g., fixedbugs/issue4495.go.
	if n == nil {
		return
	}

	// Names without definitions aren't interesting as far as
	// initialization ordering goes.
	if n.Defn == nil {
		return
	}

	if d.seen.Has(n) {
		return
	}
	d.seen.Add(n)
	if d.transitive && n.Class() == ir.PFUNC {
		d.inspectList(n.Defn.Body())
	}
}

// declOrder implements heap.Interface, ordering assignment statements
// by the position of their first LHS expression.
//
// N.B., the Pos of the first LHS expression is used because because
// an OAS node's Pos may not be unique. For example, given the
// declaration "var a, b = f(), g()", "a" must be ordered before "b",
// but both OAS nodes use the "=" token's position as their Pos.
type declOrder []ir.Node

func (s declOrder) Len() int { return len(s) }
func (s declOrder) Less(i, j int) bool {
	return firstLHS(s[i]).Pos().Before(firstLHS(s[j]).Pos())
}
func (s declOrder) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s *declOrder) Push(x interface{}) { *s = append(*s, x.(ir.Node)) }
func (s *declOrder) Pop() interface{} {
	n := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return n
}

// firstLHS returns the first expression on the left-hand side of
// assignment n.
func firstLHS(n ir.Node) *ir.Name {
	switch n.Op() {
	case ir.OAS:
		return n.Left().Name()
	case ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2RECV, ir.OAS2MAPR:
		return n.List().First().Name()
	}

	base.Fatalf("unexpected Op: %v", n.Op())
	return nil
}
