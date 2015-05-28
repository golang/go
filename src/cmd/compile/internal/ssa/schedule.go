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

	for _, b := range f.Blocks {
		// Topologically sort the values in b.
		order = order[:0]
		for _, v := range b.Values {
			if v.Op == OpPhi {
				// Phis all go first.  We handle phis specially
				// because they may have self edges "a = phi(a, b, c)"
				order = append(order, v)
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
						if w.Block == b && w.Op != OpPhi && state[w.ID] == unmarked {
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
		copy(b.Values, order)
	}
	// TODO: only allow one live mem type and one live flags type (x86)
	// This restriction will force any loads (and any flag uses) to appear
	// before the next store (flag update).  This "anti-dependence" is not
	// recorded explicitly in ssa form.
}
