// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// critical splits critical edges (those that go from a block with
// more than one outedge to a block with more than one inedge).
// Regalloc wants a critical-edge-free CFG so it can implement phi values.
func critical(f *Func) {
	for _, b := range f.Blocks {
		if len(b.Preds) <= 1 {
			continue
		}

		// split input edges coming from multi-output blocks.
		for i, c := range b.Preds {
			if c.Kind == BlockPlain {
				continue // only single output block
			}

			// allocate a new block to place on the edge
			d := f.NewBlock(BlockPlain)
			d.Line = c.Line

			// splice it in
			d.Preds = append(d.Preds, c)
			d.Succs = append(d.Succs, b)
			b.Preds[i] = d
			// replace b with d in c's successor list.
			for j, b2 := range c.Succs {
				if b2 == b {
					c.Succs[j] = d
					break
				}
			}
		}
	}
}
