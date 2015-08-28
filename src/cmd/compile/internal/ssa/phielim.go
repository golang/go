// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// phielim eliminates redundant phi values from f.
// A phi is redundant if its arguments are all equal.  For
// purposes of counting, ignore the phi itself.  Both of
// these phis are redundant:
//   v = phi(x,x,x)
//   v = phi(x,v,x,v)
func phielim(f *Func) {
	argSet := newSparseSet(f.NumValues())
	var args []*Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			argSet.clear()
			args = args[:0]
			for _, x := range v.Args {
				for x.Op == OpCopy {
					x = x.Args[0]
				}
				if x != v && !argSet.contains(x.ID) {
					argSet.add(x.ID)
					args = append(args, x)
				}
			}
			if len(args) == 1 {
				v.Op = OpCopy
				v.SetArgs1(args[0])
			}
		}
	}
}
