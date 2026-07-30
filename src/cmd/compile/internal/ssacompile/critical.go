// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// critical splits critical edges (those that go from a block with
// more than one outedge to a block with more than one inedge).
// Regalloc wants a critical-edge-free CFG so it can implement phi values.
func critical(f *ssacore.Func) {
	// maps from phi arg ID to the new block created for that argument
	blocks := f.Cache.AllocBlockSlice(f.NumValues())
	defer f.Cache.FreeBlockSlice(blocks)
	// need to iterate over f.Blocks without range, as we might
	// need to split critical edges on newly constructed blocks
	for j := 0; j < len(f.Blocks); j++ {
		b := f.Blocks[j]
		if len(b.Preds) <= 1 {
			continue
		}

		var phi *ssacore.Value
		// determine if we've only got a single phi in this
		// block, this is easier to handle than the general
		// case of a block with multiple phi values.
		for _, v := range b.Values {
			if v.Op == ssaop.OpPhi {
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
			p := e.B
			pi := e.I
			if p.Kind == block.BlockPlain {
				i++
				continue // only single output block
			}

			var d *ssacore.Block // new block used to remove critical edge
			reusedBlock := false // if true, then this is not the first use of this block
			if phi != nil {
				argID := phi.Args[i].ID
				// find or record the block that we used to split
				// critical edges for this argument
				if d = blocks[argID]; d == nil {
					// splitting doesn't necessarily remove the critical edge,
					// since we're iterating over len(f.Blocks) above, this forces
					// the new blocks to be re-examined.
					d = f.NewBlock(block.BlockPlain)
					d.Pos = p.Pos
					blocks[argID] = d
					if f.Pass.Debug > 0 {
						f.Warnl(p.Pos, "split critical edge")
					}
				} else {
					reusedBlock = true
				}
			} else {
				// no existing block, so allocate a new block
				// to place on the edge
				d = f.NewBlock(block.BlockPlain)
				d.Pos = p.Pos
				if f.Pass.Debug > 0 {
					f.Warnl(p.Pos, "split critical edge")
				}
			}

			// if this not the first argument for the
			// block, then we need to remove the
			// corresponding elements from the block
			// predecessors and phi args
			if reusedBlock {
				// Add p->d edge
				p.Succs[pi] = ssacore.Edge{B: d, I: len(d.Preds)}
				d.Preds = append(d.Preds, ssacore.Edge{B: p, I: pi})

				// Remove p as a predecessor from b.
				b.RemovePred(i)

				// Update corresponding phi args
				b.RemovePhiArg(phi, i)

				// splitting occasionally leads to a phi having
				// a single argument (occurs with -N)
				// Don't increment i in this case because we moved
				// an unprocessed predecessor down into slot i.
			} else {
				// splice it in
				p.Succs[pi] = ssacore.Edge{B: d, I: 0}
				b.Preds[i] = ssacore.Edge{B: d, I: 0}
				d.Preds = append(d.Preds, ssacore.Edge{B: p, I: pi})
				d.Succs = append(d.Succs, ssacore.Edge{B: b, I: i})
				i++
			}
		}
	}
}
