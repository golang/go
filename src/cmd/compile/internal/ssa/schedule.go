// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Schedule the Values in each Block.  After this phase returns, the
// order of b.Values matters and is the order in which those values
// will appear in the assembly output.  For now it generates an
// arbitrary valid schedule using a topological sort.  TODO(khr):
// schedule smarter.
func schedule(f *Func) {
	const (
		unmarked = 0
		found    = 1
		expanded = 2
		done     = 3
	)
	state := make([]byte, f.NumValues())
	var queue []*Value //stack-like worklist.  Contains found and expanded nodes.
	var order []*Value

	nextMem := make([]*Value, f.NumValues()) // maps mem values to the next live value
	additionalEdges := make([][]*Value, f.NumValues())
	for _, b := range f.Blocks {
		// Set the nextMem values for this block.  If the previous
		// write is from a different block, then its nextMem entry
		// might have already been set during processing of an earlier
		// block.  This loop resets the nextMem entries to be correct
		// for this block.
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				for _, w := range v.Args {
					if w.Type.IsMemory() {
						nextMem[w.ID] = v
					}
				}
			}
		}
		// Add a anti-dependency between each load v and the memory value n
		// following the memory value that v loads from.
		// This will enforce the single-live-mem restriction.
		for _, v := range b.Values {
			if v.Type.IsMemory() {
				continue
			}
			for _, w := range v.Args {
				if w.Type.IsMemory() && nextMem[w.ID] != nil {
					// Filter for intra-block edges.
					if n := nextMem[w.ID]; n.Block == b {
						additionalEdges[n.ID] = append(additionalEdges[n.ID], v)
					}
				}
			}
		}

		order = order[:0]

		// Schedule phis first
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// TODO: what if a phi is also a control op?  It happens for
				// mem ops all the time, which shouldn't matter.  But for
				// regular ops we might be violating invariants about where
				// control ops live.
				if v == b.Control && !v.Type.IsMemory() {
					f.Unimplementedf("phi is a control op %s %s", v, b)
				}
				order = append(order, v)
			}
		}

		// Topologically sort the non-phi values in b.
		for _, v := range b.Values {
			if v.Op == OpPhi {
				continue
			}
			if v == b.Control {
				continue
			}
			if state[v.ID] != unmarked {
				if state[v.ID] != done {
					panic("bad state")
				}
				continue
			}
			state[v.ID] = found
			queue = append(queue, v)
			for len(queue) > 0 {
				v = queue[len(queue)-1]
				switch state[v.ID] {
				case found:
					state[v.ID] = expanded
					// Note that v is not popped.  We leave it in place
					// until all its children have been explored.
					for _, w := range v.Args {
						if w.Block == b && w.Op != OpPhi && w != b.Control && state[w.ID] == unmarked {
							state[w.ID] = found
							queue = append(queue, w)
						}
					}
					for _, w := range additionalEdges[v.ID] {
						if w.Block == b && w.Op != OpPhi && w != b.Control && state[w.ID] == unmarked {
							state[w.ID] = found
							queue = append(queue, w)
						}
					}
				case expanded:
					queue = queue[:len(queue)-1]
					state[v.ID] = done
					order = append(order, v)
				default:
					panic("bad state")
				}
			}
		}
		if b.Control != nil {
			order = append(order, b.Control)
		}
		copy(b.Values, order)
	}
	// TODO: only allow one live flags type (x86)
	// This restriction will force and any flag uses to appear before
	// the next flag update.  This "anti-dependence" is not recorded
	// explicitly in ssa form.
}
