// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// trim removes blocks with no code in them.
// These blocks were inserted to remove critical edges.
func trim(f *Func) {
	n := 0
	for _, b := range f.Blocks {
		if !trimmableBlock(b) {
			f.Blocks[n] = b
			n++
			continue
		}

		// Splice b out of the graph. NOTE: `mergePhi` depends on the
		// order, in which the predecessors edges are merged here.
		p, i := b.Preds[0].b, b.Preds[0].i
		s, j := b.Succs[0].b, b.Succs[0].i
		ns := len(s.Preds)
		p.Succs[i] = Edge{s, j}
		s.Preds[j] = Edge{p, i}

		for _, e := range b.Preds[1:] {
			p, i := e.b, e.i
			p.Succs[i] = Edge{s, len(s.Preds)}
			s.Preds = append(s.Preds, Edge{p, i})
		}

		// If `s` had more than one predecessor, update its phi-ops to
		// account for the merge.
		if ns > 1 {
			for _, v := range s.Values {
				if v.Op == OpPhi {
					mergePhi(v, j, b)
				}
			}
			// Remove the phi-ops from `b` if they were merged into the
			// phi-ops of `s`.
			k := 0
			for _, v := range b.Values {
				if v.Op == OpPhi {
					if v.Uses == 0 {
						v.resetArgs()
						continue
					}
					// Pad the arguments of the remaining phi-ops so
					// they match the new predecessor count of `s`.
					// Since s did not have a Phi op corresponding to
					// the phi op in b, the other edges coming into s
					// must be loopback edges from s, so v is the right
					// argument to v!
					args := make([]*Value, len(v.Args))
					copy(args, v.Args)
					v.resetArgs()
					for x := 0; x < j; x++ {
						v.AddArg(v)
					}
					v.AddArg(args[0])
					for x := j + 1; x < ns; x++ {
						v.AddArg(v)
					}
					for _, a := range args[1:] {
						v.AddArg(a)
					}
				}
				b.Values[k] = v
				k++
			}
			b.Values = b.Values[:k]
		}

		// Merge the blocks' values.
		for _, v := range b.Values {
			v.Block = s
		}
		k := len(b.Values)
		m := len(s.Values)
		for i := 0; i < k; i++ {
			s.Values = append(s.Values, nil)
		}
		copy(s.Values[k:], s.Values[:m])
		copy(s.Values, b.Values)
	}
	if n < len(f.Blocks) {
		f.invalidateCFG()
		tail := f.Blocks[n:]
		for i := range tail {
			tail[i] = nil
		}
		f.Blocks = f.Blocks[:n]
	}
}

// emptyBlock returns true if the block does not contain actual
// instructions
func emptyBlock(b *Block) bool {
	for _, v := range b.Values {
		if v.Op != OpPhi {
			return false
		}
	}
	return true
}

// trimmableBlock returns true if the block can be trimmed from the CFG,
// subject to the following criteria:
//  - it should not be the first block
//  - it should be BlockPlain
//  - it should not loop back to itself
//  - it either is the single predecessor of the successor block or
//    contains no actual instructions
func trimmableBlock(b *Block) bool {
	if b.Kind != BlockPlain || b == b.Func.Entry {
		return false
	}
	s := b.Succs[0].b
	return s != b && (len(s.Preds) == 1 || emptyBlock(b))
}

// mergePhi adjusts the number of `v`s arguments to account for merge
// of `b`, which was `i`th predecessor of the `v`s block. Returns
// `v`.
func mergePhi(v *Value, i int, b *Block) *Value {
	u := v.Args[i]
	if u.Block == b {
		if u.Op != OpPhi {
			b.Func.Fatalf("value %s is not a phi operation", u.LongString())
		}
		// If the original block contained u = φ(u0, u1, ..., un) and
		// the current phi is
		//    v = φ(v0, v1, ..., u, ..., vk)
		// then the merged phi is
		//    v = φ(v0, v1, ..., u0, ..., vk, u1, ..., un)
		v.SetArg(i, u.Args[0])
		v.AddArgs(u.Args[1:]...)
	} else {
		// If the original block contained u = φ(u0, u1, ..., un) and
		// the current phi is
		//    v = φ(v0, v1, ...,  vi, ..., vk)
		// i.e. it does not use a value from the predecessor block,
		// then the merged phi is
		//    v = φ(v0, v1, ..., vk, vi, vi, ...)
		for j := 1; j < len(b.Preds); j++ {
			v.AddArg(v.Args[i])
		}
	}
	return v
}
