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
//   b        b        b      b
//  / \      | \      / |    | |
// s0  s1    |  s1   s0 |    | |
//  \ /      | /      \ |    | |
//   ss      ss        ss     ss
//
// If all Phi ops in ss have identical variables for slots corresponding to
// s0, s1 and b then the branch can be dropped.
// TODO: If ss doesn't contain any OpPhis, are s0 and s1 dead code anyway.
func fuseBlockIf(b *Block) bool {
	if b.Kind != BlockIf {
		return false
	}

	var ss0, ss1 *Block
	s0 := b.Succs[0]
	if s0.Kind != BlockPlain || len(s0.Preds) != 1 || len(s0.Values) != 0 {
		s0, ss0 = b, s0
	} else {
		ss0 = s0.Succs[0]
	}
	s1 := b.Succs[1]
	if s1.Kind != BlockPlain || len(s1.Preds) != 1 || len(s1.Values) != 0 {
		s1, ss1 = b, s1
	} else {
		ss1 = s1.Succs[0]
	}

	if ss0 != ss1 {
		return false
	}
	ss := ss0

	// s0 and s1 are equal with b if the corresponding block is missing
	// (2nd, 3rd and 4th case in the figure).
	i0, i1 := -1, -1
	for i, p := range ss.Preds {
		if p == s0 {
			i0 = i
		}
		if p == s1 {
			i1 = i
		}
	}
	if i0 == -1 || i1 == -1 {
		b.Fatalf("invalid predecessors")
	}
	for _, v := range ss.Values {
		if v.Op == OpPhi && v.Args[i0] != v.Args[i1] {
			return false
		}
	}

	// Now we have two of following b->ss, b->s0->ss and b->s1->ss,
	// with s0 and s1 empty if exist.
	// We can replace it with b->ss without if all OpPhis in ss
	// have identical predecessors (verified above).
	// No critical edge is introduced because b will have one successor.
	if s0 != b && s1 != b {
		ss.removePred(s0)

		// Replace edge b->s1->ss with b->ss.
		// We need to keep a slot for Phis corresponding to b.
		for i := range b.Succs {
			if b.Succs[i] == s1 {
				b.Succs[i] = ss
			}
		}
		for i := range ss.Preds {
			if ss.Preds[i] == s1 {
				ss.Preds[i] = b
			}
		}
	} else if s0 != b {
		ss.removePred(s0)
	} else if s1 != b {
		ss.removePred(s1)
	}
	b.Kind = BlockPlain
	b.SetControl(nil)
	b.Succs = append(b.Succs[:0], ss)

	// Trash the empty blocks s0 & s1.
	if s0 != b {
		s0.Kind = BlockInvalid
		s0.Values = nil
		s0.Succs = nil
		s0.Preds = nil
	}
	if s1 != b {
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
