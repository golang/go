// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// trim removes blocks with no code in them.
// These blocks were inserted to remove critical edges.
func trim(f *Func) {
	n := 0
	for _, b := range f.Blocks {
		if b.Kind != BlockPlain || len(b.Values) != 0 || len(b.Preds) != 1 {
			f.Blocks[n] = b
			n++
			continue
		}
		// TODO: handle len(b.Preds)>1 case.

		// Splice b out of the graph.
		p := b.Preds[0].b
		i := b.Preds[0].i
		s := b.Succs[0].b
		j := b.Succs[0].i
		p.Succs[i] = Edge{s, j}
		s.Preds[j] = Edge{p, i}
	}
	tail := f.Blocks[n:]
	for i := range tail {
		tail[i] = nil
	}
	f.Blocks = f.Blocks[:n]
}
