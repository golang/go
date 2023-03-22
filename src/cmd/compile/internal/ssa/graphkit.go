// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// ----------------------------------------------------------------------------
// Graph transformation

// replaceUses replaces all uses of old in b with new.
func (b *Block) replaceUses(old, new *Value) {
	for _, v := range b.Values {
		for i, a := range v.Args {
			if a == old {
				v.SetArg(i, new)
			}
		}
	}
	for i, v := range b.ControlValues() {
		if v == old {
			b.ReplaceControl(i, new)
		}
	}
}

// moveTo moves v to dst, adjusting the appropriate Block.Values slices.
// The caller is responsible for ensuring that this is safe.
// i is the index of v in v.Block.Values.
func (v *Value) moveTo(dst *Block, i int) {
	if dst.Func.scheduled {
		v.Fatalf("moveTo after scheduling")
	}
	src := v.Block
	if src.Values[i] != v {
		v.Fatalf("moveTo bad index %d", v, i)
	}
	if src == dst {
		return
	}
	v.Block = dst
	dst.Values = append(dst.Values, v)
	last := len(src.Values) - 1
	src.Values[i] = src.Values[last]
	src.Values[last] = nil
	src.Values = src.Values[:last]
}
