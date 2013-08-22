package pointer

// This file defines a naive Andersen-style solver for the inclusion
// constraint system.

import (
	"fmt"

	"code.google.com/p/go.tools/go/types"
)

func (a *analysis) solve() {
	// Initialize points-to sets and complex constraint sets.
	for _, c := range a.constraints {
		c.init(a)
	}
	a.constraints = nil // aid GC

	work := a.work

	// Now we've initialized all constraints, we populate the
	// worklist with nodes that point to something initially (due
	// to addrConstraints) and have other constraints attached.
	for id, n := range a.nodes {
		if len(n.pts) > 0 && (n.copyTo != nil || n.complex != nil) {
			if a.log != nil {
				fmt.Fprintf(a.log, "Adding to worklist n%d\n", id)
			}
			a.addWork(nodeid(id))
		}
	}
	work.swap()

	// Solver main loop.
	for round := 1; ; round++ {
		if work.swap() {
			if a.log != nil {
				fmt.Fprintf(a.log, "Solving, round %d\n", round)
			}

			// Next iteration.
			if work.empty() {
				break // done
			}
		}

		id := work.take()
		n := a.nodes[id]

		if a.log != nil {
			fmt.Fprintf(a.log, "\tnode n%d\n", id)
		}

		// Difference propagation.
		delta := n.pts.diff(n.prevPts)
		if delta == nil {
			continue
		}
		n.prevPts = n.pts.clone()

		// Process complex constraints dependent on n.
		for c := range n.complex {
			if a.log != nil {
				fmt.Fprintf(a.log, "\t\tconstraint %s\n", c)
			}
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

		if a.log != nil {
			fmt.Fprintf(a.log, "\t\tpts(n%d) = %s\n", id, n.pts)
		}
	}

	if a.log != nil {
		fmt.Fprintf(a.log, "Solver done\n")
	}
}

func (a *analysis) addWork(id nodeid) {
	a.work.add(id)
	if a.log != nil {
		fmt.Fprintf(a.log, "\t\twork: n%d\n", id)
	}
}

func (c *addrConstraint) init(a *analysis) {
	a.nodes[c.dst].pts.add(c.src)
}
func (c *copyConstraint) init(a *analysis) {
	a.nodes[c.src].copyTo.add(c.dst)
}

// Complex constraints attach themselves to the relevant pointer node.

func (c *storeConstraint) init(a *analysis) {
	a.nodes[c.dst].complex.add(c)
}
func (c *loadConstraint) init(a *analysis) {
	a.nodes[c.src].complex.add(c)
}
func (c *offsetAddrConstraint) init(a *analysis) {
	a.nodes[c.src].complex.add(c)
}
func (c *typeAssertConstraint) init(a *analysis) {
	a.nodes[c.src].complex.add(c)
}
func (c *invokeConstraint) init(a *analysis) {
	a.nodes[c.iface].complex.add(c)
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
			return a.nodes[dst].pts.addAll(nsrc.pts)
		}
	}
	return false
}

// Returns sizeof.
// Implicitly adds nodes to worklist.
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

func (c *typeAssertConstraint) solve(a *analysis, n *node, delta nodeset) {
	tIface, _ := c.typ.Underlying().(*types.Interface)

	for ifaceObj := range delta {
		ifaceValue, tConc := a.interfaceValue(ifaceObj)

		if tIface != nil {
			if types.IsAssignableTo(tConc, tIface) {
				if a.nodes[c.dst].pts.add(ifaceObj) {
					a.addWork(c.dst)
				}
			}
		} else {
			if types.IsIdentical(tConc, c.typ) {
				// Copy entire payload to dst.
				//
				// TODO(adonovan): opt: if tConc is
				// nonpointerlike we can skip this
				// entire constraint, perhaps.  We
				// only care about pointers among the
				// fields.
				a.onlineCopyN(c.dst, ifaceValue, a.sizeof(tConc))
			}
		}
	}
}

func (c *invokeConstraint) solve(a *analysis, n *node, delta nodeset) {
	for ifaceObj := range delta {
		ifaceValue, tConc := a.interfaceValue(ifaceObj)

		// Look up the concrete method.
		meth := tConc.MethodSet().Lookup(c.method.Pkg(), c.method.Name())
		if meth == nil {
			panic(fmt.Sprintf("n%d: type %s has no method %s (iface=n%d)",
				c.iface, tConc, c.method, ifaceObj))
		}
		fn := a.prog.Method(meth)
		if fn == nil {
			panic(fmt.Sprintf("n%d: no ssa.Function for %s", c.iface, meth))
		}
		sig := fn.Signature

		fnObj := a.funcObj[fn]

		// Make callsite's fn variable point to identity of
		// concrete method.  (There's no need to add it to
		// worklist since it never has attached constraints.)
		a.nodes[c.params].pts.add(fnObj)

		// Extract value and connect to method's receiver.
		// Copy payload to method's receiver param (arg0).
		arg0 := a.funcParams(fnObj)
		recvSize := a.sizeof(sig.Recv().Type())
		a.onlineCopyN(arg0, ifaceValue, recvSize)

		// Copy iface object payload to method receiver.
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
