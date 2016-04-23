// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// trim removes blocks with no code in them.
// These blocks were inserted to remove critical edges.
func trim(f *Func) {
	i := 0
	for _, b := range f.Blocks {
		if b.Kind != BlockPlain || len(b.Values) != 0 || len(b.Preds) != 1 {
			f.Blocks[i] = b
			i++
			continue
		}
		// TODO: handle len(b.Preds)>1 case.

		// Splice b out of the graph.
		pred := b.Preds[0]
		succ := b.Succs[0]
		for j, s := range pred.Succs {
			if s == b {
				pred.Succs[j] = succ
			}
		}
		for j, p := range succ.Preds {
			if p == b {
				succ.Preds[j] = pred
			}
		}
	}
	for j := i; j < len(f.Blocks); j++ {
		f.Blocks[j] = nil
	}
	f.Blocks = f.Blocks[:i]
}
