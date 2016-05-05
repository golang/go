// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// phiopt eliminates boolean Phis based on the previous if.
//
// Main use case is to transform:
//   x := false
//   if b {
//     x = true
//   }
// into x = b.
//
// In SSA code this appears as
//
// b0
//   If b -> b1 b2
// b1
//   Plain -> b2
// b2
//   x = (OpPhi (ConstBool [true]) (ConstBool [false]))
//
// In this case we can replace x with a copy of b.
func phiopt(f *Func) {
	for _, b := range f.Blocks {
		if len(b.Preds) != 2 || len(b.Values) == 0 {
			// TODO: handle more than 2 predecessors, e.g. a || b || c.
			continue
		}

		pb0, b0 := b, b.Preds[0].b
		for len(b0.Succs) == 1 && len(b0.Preds) == 1 {
			pb0, b0 = b0, b0.Preds[0].b
		}
		if b0.Kind != BlockIf {
			continue
		}
		pb1, b1 := b, b.Preds[1].b
		for len(b1.Succs) == 1 && len(b1.Preds) == 1 {
			pb1, b1 = b1, b1.Preds[0].b
		}
		if b1 != b0 {
			continue
		}
		// b0 is the if block giving the boolean value.

		// reverse is the predecessor from which the truth value comes.
		var reverse int
		if b0.Succs[0].b == pb0 && b0.Succs[1].b == pb1 {
			reverse = 0
		} else if b0.Succs[0].b == pb1 && b0.Succs[1].b == pb0 {
			reverse = 1
		} else {
			b.Fatalf("invalid predecessors\n")
		}

		for _, v := range b.Values {
			if v.Op != OpPhi || !v.Type.IsBoolean() {
				continue
			}

			// Replaces
			//   if a { x = true } else { x = false } with x = a
			// and
			//   if a { x = false } else { x = true } with x = !a
			if v.Args[0].Op == OpConstBool && v.Args[1].Op == OpConstBool {
				if v.Args[reverse].AuxInt != v.Args[1-reverse].AuxInt {
					ops := [2]Op{OpNot, OpCopy}
					v.reset(ops[v.Args[reverse].AuxInt])
					v.AddArg(b0.Control)
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "converted OpPhi to %v", v.Op)
					}
					continue
				}
			}

			// Replaces
			//   if a { x = true } else { x = value } with x = a || value.
			// Requires that value dominates x, meaning that regardless of a,
			// value is always computed. This guarantees that the side effects
			// of value are not seen if a is false.
			if v.Args[reverse].Op == OpConstBool && v.Args[reverse].AuxInt == 1 {
				if tmp := v.Args[1-reverse]; f.sdom.isAncestorEq(tmp.Block, b) {
					v.reset(OpOrB)
					v.SetArgs2(b0.Control, tmp)
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "converted OpPhi to %v", v.Op)
					}
					continue
				}
			}

			// Replaces
			//   if a { x = value } else { x = false } with x = a && value.
			// Requires that value dominates x, meaning that regardless of a,
			// value is always computed. This guarantees that the side effects
			// of value are not seen if a is false.
			if v.Args[1-reverse].Op == OpConstBool && v.Args[1-reverse].AuxInt == 0 {
				if tmp := v.Args[reverse]; f.sdom.isAncestorEq(tmp.Block, b) {
					v.reset(OpAndB)
					v.SetArgs2(b0.Control, tmp)
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "converted OpPhi to %v", v.Op)
					}
					continue
				}
			}
		}
	}

}
