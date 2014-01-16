// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file defines a naive Andersen-style solver for the inclusion
// constraint system.

import (
	"fmt"

	"code.google.com/p/go.tools/go/types"
)

func (a *analysis) solve() {
	// Solver main loop.
	for round := 1; ; round++ {
		if a.log != nil {
			fmt.Fprintf(a.log, "Solving, round %d\n", round)
		}

		// Add new constraints to the graph:
		// static constraints from SSA on round 1,
		// dynamic constraints from reflection thereafter.
		a.processNewConstraints()

		id := a.work.take()
		if id == empty {
			break
		}
		if a.log != nil {
			fmt.Fprintf(a.log, "\tnode n%d\n", id)
		}

		n := a.nodes[id]

		// Difference propagation.
		delta := n.pts.diff(n.prevPts)
		if delta == nil {
			continue
		}
		n.prevPts = n.pts.clone()

		// Apply all resolution rules attached to n.
		a.solveConstraints(n, delta)

		if a.log != nil {
			fmt.Fprintf(a.log, "\t\tpts(n%d) = %s\n", id, n.pts)
		}
	}

	if len(a.nodes[0].pts) > 0 {
		panic(fmt.Sprintf("pts(0) is nonempty: %s", a.nodes[0].pts))
	}

	if a.log != nil {
		fmt.Fprintf(a.log, "Solver done\n")
	}
}

// processNewConstraints takes the new constraints from a.constraints
// and adds them to the graph, ensuring
// that new constraints are applied to pre-existing labels and
// that pre-existing constraints are applied to new labels.
//
func (a *analysis) processNewConstraints() {
	// Take the slice of new constraints.
	// (May grow during call to solveConstraints.)
	constraints := a.constraints
	a.constraints = nil

	// Initialize points-to sets from addr-of (base) constraints.
	for _, c := range constraints {
		if c, ok := c.(*addrConstraint); ok {
			dst := a.nodes[c.dst]
			dst.pts.add(c.src)

			// Populate the worklist with nodes that point to
			// something initially (due to addrConstraints) and
			// have other constraints attached.
			// (A no-op in round 1.)
			if dst.copyTo != nil || dst.complex != nil {
				a.addWork(c.dst)
			}
		}
	}

	// Attach simple (copy) and complex constraints to nodes.
	var stale nodeset
	for _, c := range constraints {
		var id nodeid
		switch c := c.(type) {
		case *addrConstraint:
			// base constraints handled in previous loop
			continue
		case *copyConstraint:
			// simple (copy) constraint
			id = c.src
			a.nodes[id].copyTo.add(c.dst)
		default:
			// complex constraint
			id = c.ptr()
			a.nodes[id].complex.add(c)
		}

		if n := a.nodes[id]; len(n.pts) > 0 {
			if len(n.prevPts) > 0 {
				stale.add(id)
			}
			a.addWork(id)
		}
	}
	// Apply new constraints to pre-existing PTS labels.
	for id := range stale {
		n := a.nodes[id]
		a.solveConstraints(n, n.prevPts)
	}
}

// solveConstraints applies each resolution rule attached to node n to
// the set of labels delta.  It may generate new constraints in
// a.constraints.
//
func (a *analysis) solveConstraints(n *node, delta nodeset) {
	if delta == nil {
		return
	}

	// Process complex constraints dependent on n.
	for c := range n.complex {
		if a.log != nil {
			fmt.Fprintf(a.log, "\t\tconstraint %s\n", c)
		}
		// TODO(adonovan): parameter n is never needed, since
		// it's equal to c.ptr().  Remove.
		c.solve(a, n, delta)
	}

	// Process copy constraints.
	var copySeen nodeset
	for mid := range n.copyTo {
		if copySeen.add(mid) {
			if a.nodes[mid].pts.addAll(delta) {
				a.addWork(mid)
			}
		}
	}
}

// addLabel adds label to the points-to set of ptr and reports whether the set grew.
func (a *analysis) addLabel(ptr, label nodeid) bool {
	return a.nodes[ptr].pts.add(label)
}

func (a *analysis) addWork(id nodeid) {
	a.work.add(id)
	if a.log != nil {
		fmt.Fprintf(a.log, "\t\twork: n%d\n", id)
	}
}

func (c *addrConstraint) ptr() nodeid {
	panic("addrConstraint: not a complex constraint")
}
func (c *copyConstraint) ptr() nodeid {
	panic("addrConstraint: not a complex constraint")
}

// Complex constraints attach themselves to the relevant pointer node.

func (c *storeConstraint) ptr() nodeid {
	return c.dst
}
func (c *loadConstraint) ptr() nodeid {
	return c.src
}
func (c *offsetAddrConstraint) ptr() nodeid {
	return c.src
}
func (c *typeFilterConstraint) ptr() nodeid {
	return c.src
}
func (c *untagConstraint) ptr() nodeid {
	return c.src
}
func (c *invokeConstraint) ptr() nodeid {
	return c.iface
}

// onlineCopy adds a copy edge.  It is called online, i.e. during
// solving, so it adds edges and pts members directly rather than by
// instantiating a 'constraint'.
//
// The size of the copy is implicitly 1.
// It returns true if pts(dst) changed.
//
func (a *analysis) onlineCopy(dst, src nodeid) bool {
	if dst != src {
		if nsrc := a.nodes[src]; nsrc.copyTo.add(dst) {
			if a.log != nil {
				fmt.Fprintf(a.log, "\t\t\tdynamic copy n%d <- n%d\n", dst, src)
			}
			// TODO(adonovan): most calls to onlineCopy
			// are followed by addWork, possibly batched
			// via a 'changed' flag; see if there's a
			// noticeable penalty to calling addWork here.
			return a.nodes[dst].pts.addAll(nsrc.pts)
		}
	}
	return false
}

// Returns sizeof.
// Implicitly adds nodes to worklist.
//
// TODO(adonovan): now that we support a.copy() during solving, we
// could eliminate onlineCopyN, but it's much slower.  Investigate.
//
func (a *analysis) onlineCopyN(dst, src nodeid, sizeof uint32) uint32 {
	for i := uint32(0); i < sizeof; i++ {
		if a.onlineCopy(dst, src) {
			a.addWork(dst)
		}
		src++
		dst++
	}
	return sizeof
}

func (c *loadConstraint) solve(a *analysis, n *node, delta nodeset) {
	var changed bool
	for k := range delta {
		koff := k + nodeid(c.offset)
		if a.onlineCopy(c.dst, koff) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.dst)
	}
}

func (c *storeConstraint) solve(a *analysis, n *node, delta nodeset) {
	for k := range delta {
		koff := k + nodeid(c.offset)
		if a.onlineCopy(koff, c.src) {
			a.addWork(koff)
		}
	}
}

func (c *offsetAddrConstraint) solve(a *analysis, n *node, delta nodeset) {
	dst := a.nodes[c.dst]
	for k := range delta {
		if dst.pts.add(k + nodeid(c.offset)) {
			a.addWork(c.dst)
		}
	}
}

func (c *typeFilterConstraint) solve(a *analysis, n *node, delta nodeset) {
	for ifaceObj := range delta {
		tDyn, _, indirect := a.taggedValue(ifaceObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		if types.IsAssignableTo(tDyn, c.typ) {
			if a.addLabel(c.dst, ifaceObj) {
				a.addWork(c.dst)
			}
		}
	}
}

func (c *untagConstraint) solve(a *analysis, n *node, delta nodeset) {
	predicate := types.IsAssignableTo
	if c.exact {
		predicate = types.IsIdentical
	}
	for ifaceObj := range delta {
		tDyn, v, indirect := a.taggedValue(ifaceObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		if predicate(tDyn, c.typ) {
			// Copy payload sans tag to dst.
			//
			// TODO(adonovan): opt: if tConc is
			// nonpointerlike we can skip this entire
			// constraint, perhaps.  We only care about
			// pointers among the fields.
			a.onlineCopyN(c.dst, v, a.sizeof(tDyn))
		}
	}
}

func (c *invokeConstraint) solve(a *analysis, n *node, delta nodeset) {
	for ifaceObj := range delta {
		tDyn, v, indirect := a.taggedValue(ifaceObj)
		if indirect {
			// TODO(adonovan): we may need to implement this if
			// we ever apply invokeConstraints to reflect.Value PTSs,
			// e.g. for (reflect.Value).Call.
			panic("indirect tagged object")
		}

		// Look up the concrete method.
		meth := tDyn.MethodSet().Lookup(c.method.Pkg(), c.method.Name())
		if meth == nil {
			panic(fmt.Sprintf("n%d: type %s has no method %s (iface=n%d)",
				c.iface, tDyn, c.method, ifaceObj))
		}
		fn := a.prog.Method(meth)
		if fn == nil {
			panic(fmt.Sprintf("n%d: no ssa.Function for %s", c.iface, meth))
		}
		sig := fn.Signature

		fnObj := a.globalobj[fn] // dynamic calls use shared contour
		if fnObj == 0 {
			// a.objectNode(fn) was not called during gen phase.
			panic(fmt.Sprintf("a.globalobj[%s]==nil", fn))
		}

		// Make callsite's fn variable point to identity of
		// concrete method.  (There's no need to add it to
		// worklist since it never has attached constraints.)
		a.addLabel(c.params, fnObj)

		// Extract value and connect to method's receiver.
		// Copy payload to method's receiver param (arg0).
		arg0 := a.funcParams(fnObj)
		recvSize := a.sizeof(sig.Recv().Type())
		a.onlineCopyN(arg0, v, recvSize)

		src := c.params + 1 // skip past identity
		dst := arg0 + nodeid(recvSize)

		// Copy caller's argument block to method formal parameters.
		paramsSize := a.sizeof(sig.Params())
		a.onlineCopyN(dst, src, paramsSize)
		src += nodeid(paramsSize)
		dst += nodeid(paramsSize)

		// Copy method results to caller's result block.
		resultsSize := a.sizeof(sig.Results())
		a.onlineCopyN(src, dst, resultsSize)
	}
}

func (c *addrConstraint) solve(a *analysis, n *node, delta nodeset) {
	panic("addr is not a complex constraint")
}

func (c *copyConstraint) solve(a *analysis, n *node, delta nodeset) {
	panic("copy is not a complex constraint")
}
