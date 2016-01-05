// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// copyelim removes all copies from f.
func copyelim(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, w := range v.Args {
				x := w
				for x.Op == OpCopy {
					x = x.Args[0]
				}
				if x != w {
					v.Args[i] = x
				}
			}
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
				values[i] = v
			}
		}
	}
}
