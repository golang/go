// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// copyelim removes all copies from f.
func copyelim(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			copyelimValue(v)
		}
		v := b.Control
		if v != nil {
			for v.Op == OpCopy {
				v = v.Args[0]
			}
			b.Control = v
		}
	}

	// Update named values.
	for _, name := range f.Names {
		values := f.NamedValues[name]
		for i, v := range values {
			x := v
			for x.Op == OpCopy {
				x = x.Args[0]
			}
			if x != v {
				values[i] = x
			}
		}
	}
}

func copyelimValue(v *Value) {
	// elide any copies generated during rewriting
	for i, a := range v.Args {
		if a.Op != OpCopy {
			continue
		}
		// Rewriting can generate OpCopy loops.
		// They are harmless (see removePredecessor),
		// but take care to stop if we find a cycle.
		slow := a // advances every other iteration
		var advance bool
		for a.Op == OpCopy {
			a = a.Args[0]
			if slow == a {
				break
			}
			if advance {
				slow = slow.Args[0]
			}
			advance = !advance
		}
		v.Args[i] = a
	}
}
