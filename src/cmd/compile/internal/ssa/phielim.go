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
	args := newSparseSet(f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			args.clear()
			for _, x := range v.Args {
				for x.Op == OpCopy {
					x = x.Args[0]
				}
				args.add(x.ID)
			}
			switch {
			case args.size() == 1:
				v.Op = OpCopy
				v.SetArgs1(v.Args[0])
			case args.size() == 2 && args.contains(v.ID):
				var w *Value
				for _, x := range v.Args {
					if x.ID != v.ID {
						w = x
						break
					}
				}
				v.Op = OpCopy
				v.SetArgs1(w)
			}
		}
	}
}
