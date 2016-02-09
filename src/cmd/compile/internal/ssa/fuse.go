// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// fuse simplifies control flow by joining basic blocks.
func fuse(f *Func) {
	for changed := true; changed; {
		changed = false
		for _, b := range f.Blocks {
			changed = fuseBlockIf(b) || changed
			changed = fuseBlockPlain(b) || changed
		}
	}
}

// fuseBlockIf handles the following cases where s0 and s1 are empty blocks.
//
//   b       b          b
//  / \      | \      / |
// s0  s1    |  s1   s0 |
//  \ /      | /      \ |
//   ss      ss        ss
//
// If ss doesn't contain any Phi ops and s0 & s1 are empty then the branch
// can be dropped.
// TODO: If ss doesn't contain any Phi ops, are s0 and s1 dead code anyway?
func fuseBlockIf(b *Block) bool {
	if b.Kind != BlockIf {
		return false
	}

	var ss0, ss1 *Block
	s0 := b.Succs[0]
	if s0.Kind != BlockPlain || len(s0.Preds) != 1 || len(s0.Values) != 0 {
		s0, ss0 = nil, s0
	} else {
		ss0 = s0.Succs[0]
	}
	s1 := b.Succs[1]
	if s1.Kind != BlockPlain || len(s1.Preds) != 1 || len(s1.Values) != 0 {
		s1, ss1 = nil, s1
	} else {
		ss1 = s1.Succs[0]
	}

	if ss0 != ss1 {
		return false
	}
	ss := ss0

	// TODO: Handle OpPhi operations. We can still replace OpPhi if the
	// slots corresponding to b, s0 and s1 point to the same variable.
	for _, v := range ss.Values {
		if v.Op == OpPhi {
			return false
		}
	}

	// Now we have two following b->ss, b->s0->ss and b->s1->ss,
	// with s0 and s1 empty if exist.
	// We can replace it with b->ss without if ss has no phis
	// which is checked above.
	// No critical edge is introduced because b will have one successor.
	if s0 != nil {
		ss.removePred(s0)
	}
	if s1 != nil {
		ss.removePred(s1)
	}
	if s0 != nil && s1 != nil {
		// Add an edge if both edges are removed, otherwise b is no longer connected to ss.
		ss.Preds = append(ss.Preds, b)
	}
	b.Kind = BlockPlain
	b.Control = nil
	b.Succs = append(b.Succs[:0], ss)

	// Trash the empty blocks s0 & s1.
	if s0 != nil {
		s0.Kind = BlockInvalid
		s0.Values = nil
		s0.Succs = nil
		s0.Preds = nil
	}
	if s1 != nil {
		s1.Kind = BlockInvalid
		s1.Values = nil
		s1.Succs = nil
		s1.Preds = nil
	}
	return true
}

func fuseBlockPlain(b *Block) bool {
	if b.Kind != BlockPlain {
		return false
	}

	c := b.Succs[0]
	if len(c.Preds) != 1 {
		return false
	}

	// move all of b'c values to c.
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
	if f := b.Func; f.Entry == b {
		f.Entry = c
	}

	// trash b, just in case
	b.Kind = BlockInvalid
	b.Values = nil
	b.Preds = nil
	b.Succs = nil
	return true
}
