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
		for i := 0; i < len(b.Preds); {
			e := b.Preds[i]
			p := e.b
			pi := e.i
			if p.Kind == BlockPlain {
				i++
				continue // only single output block
			}

			var d *Block         // new block used to remove critical edge
			reusedBlock := false // if true, then this is not the first use of this block
			if phi != nil {
				argID := phi.Args[i].ID
				// find or record the block that we used to split
				// critical edges for this argument
				if d = blocks[argID]; d == nil {
					// splitting doesn't necessarily remove the critical edge,
					// since we're iterating over len(f.Blocks) above, this forces
					// the new blocks to be re-examined.
					d = f.NewBlock(BlockPlain)
					d.Pos = p.Pos
					blocks[argID] = d
					if f.pass.debug > 0 {
						f.Warnl(p.Pos, "split critical edge")
					}
				} else {
					reusedBlock = true
				}
			} else {
				// no existing block, so allocate a new block
				// to place on the edge
				d = f.NewBlock(BlockPlain)
				d.Pos = p.Pos
				if f.pass.debug > 0 {
					f.Warnl(p.Pos, "split critical edge")
				}
			}

			// if this not the first argument for the
			// block, then we need to remove the
			// corresponding elements from the block
			// predecessors and phi args
			if reusedBlock {
				// Add p->d edge
				p.Succs[pi] = Edge{d, len(d.Preds)}
				d.Preds = append(d.Preds, Edge{p, pi})

				// Remove p as a predecessor from b.
				b.removePred(i)

				// Update corresponding phi args
				n := len(b.Preds)
				phi.Args[i].Uses--
				phi.Args[i] = phi.Args[n]
				phi.Args[n] = nil
				phi.Args = phi.Args[:n]
				// splitting occasionally leads to a phi having
				// a single argument (occurs with -N)
				if n == 1 {
					phi.Op = OpCopy
				}
				// Don't increment i in this case because we moved
				// an unprocessed predecessor down into slot i.
			} else {
				// splice it in
				p.Succs[pi] = Edge{d, 0}
				b.Preds[i] = Edge{d, 0}
				d.Preds = append(d.Preds, Edge{p, pi})
				d.Succs = append(d.Succs, Edge{b, i})
				i++
			}
		}
	}
}
