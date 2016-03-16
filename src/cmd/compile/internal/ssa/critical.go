// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// critical splits critical edges (those that go from a block with
// more than one outedge to a block with more than one inedge).
// Regalloc wants a critical-edge-free CFG so it can implement phi values.
func critical(f *Func) {
	// maps from phi arg ID to the new block created for that argument
	blocks := make([]*Block, f.NumValues())
	// need to iterate over f.Blocks without range, as we might
	// need to split critical edges on newly constructed blocks
	for j := 0; j < len(f.Blocks); j++ {
		b := f.Blocks[j]
		if len(b.Preds) <= 1 {
			continue
		}

		var phi *Value
		// determine if we've only got a single phi in this
		// block, this is easier to handle than the general
		// case of a block with multiple phi values.
		for _, v := range b.Values {
			if v.Op == OpPhi {
				if phi != nil {
					phi = nil
					break
				}
				phi = v
			}
		}

		// reset our block map
		if phi != nil {
			for _, v := range phi.Args {
				blocks[v.ID] = nil
			}
		}

		// split input edges coming from multi-output blocks.
		for i := 0; i < len(b.Preds); i++ {
			c := b.Preds[i]
			if c.Kind == BlockPlain {
				continue // only single output block
			}

			var d *Block         // new block used to remove critical edge
			reusedBlock := false // if true, then this is not the first use of this block
			if phi != nil {
				argID := phi.Args[i].ID
				// find or record the block that we used to split
				// critical edges for this argument
				if d = blocks[argID]; d == nil {
					d = f.NewBlock(BlockPlain)
					d.Line = c.Line
					blocks[argID] = d
					if f.pass.debug > 0 {
						f.Config.Warnl(c.Line, "split critical edge")
					}
				} else {
					reusedBlock = true
				}
			} else {
				// no existing block, so allocate a new block
				// to place on the edge
				d = f.NewBlock(BlockPlain)
				d.Line = c.Line
				if f.pass.debug > 0 {
					f.Config.Warnl(c.Line, "split critical edge")
				}
			}

			// if this not the first argument for the
			// block, then we need to remove the
			// corresponding elements from the block
			// predecessors and phi args
			if reusedBlock {
				d.Preds = append(d.Preds, c)
				b.Preds[i] = nil
				phi.Args[i] = nil
			} else {
				// splice it in
				d.Preds = append(d.Preds, c)
				d.Succs = append(d.Succs, b)
				b.Preds[i] = d
			}

			// replace b with d in c's successor list.
			for j, b2 := range c.Succs {
				if b2 == b {
					c.Succs[j] = d
					break
				}
			}
		}

		// clean up phi's args and b's predecessor list
		if phi != nil {
			phi.Args = filterNilValues(phi.Args)
			b.Preds = filterNilBlocks(b.Preds)
		}
	}
}

// filterNilValues preserves the order of v, while filtering out nils.
func filterNilValues(v []*Value) []*Value {
	nv := v[:0]
	for i := range v {
		if v[i] != nil {
			nv = append(nv, v[i])
		}
	}
	return nv
}

// filterNilBlocks preserves the order of b, while filtering out nils.
func filterNilBlocks(b []*Block) []*Block {
	nb := b[:0]
	for i := range b {
		if b[i] != nil {
			nb = append(nb, b[i])
		}
	}
	return nb
}
