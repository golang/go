// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// fuse simplifies control flow by joining basic blocks.
func fuse(f *Func) {
	for _, b := range f.Blocks {
		if b.Kind != BlockPlain {
			continue
		}
		c := b.Succs[0]
		if len(c.Preds) != 1 {
			continue
		}

		// move all of b's values to c.
		for _, v := range b.Values {
			v.Block = c
			c.Values = append(c.Values, v)
		}

		// replace b->c edge with preds(b) -> c
		c.predstorage[0] = nil
		if len(b.Preds) > len(b.predstorage) {
			c.Preds = b.Preds
		} else {
			c.Preds = append(c.predstorage[:0], b.Preds...)
		}
		for _, p := range c.Preds {
			for i, q := range p.Succs {
				if q == b {
					p.Succs[i] = c
				}
			}
		}
		if f.Entry == b {
			f.Entry = c
		}

		// trash b, just in case
		b.Kind = BlockInvalid
		b.Values = nil
		b.Preds = nil
		b.Succs = nil
	}
}
