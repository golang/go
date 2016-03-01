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
			continue
		}

		pb0, b0 := b, b.Preds[0]
		for b0.Kind != BlockIf && len(b0.Preds) == 1 {
			pb0, b0 = b0, b0.Preds[0]
		}
		if b0.Kind != BlockIf {
			continue
		}
		pb1, b1 := b, b.Preds[1]
		for b1.Kind != BlockIf && len(b1.Preds) == 1 {
			pb1, b1 = b1, b1.Preds[0]
		}
		if b1 != b0 {
			continue
		}
		// b0 is the if block giving the boolean value.

		var reverse bool
		if b0.Succs[0] == pb0 && b0.Succs[1] == pb1 {
			reverse = false
		} else if b0.Succs[0] == pb1 && b0.Succs[1] == pb0 {
			reverse = true
		} else {
			b.Fatalf("invalid predecessors\n")
		}

		for _, v := range b.Values {
			if v.Op != OpPhi || !v.Type.IsBoolean() || v.Args[0].Op != OpConstBool || v.Args[1].Op != OpConstBool {
				continue
			}

			ok, isCopy := false, false
			if v.Args[0].AuxInt == 1 && v.Args[1].AuxInt == 0 {
				ok, isCopy = true, !reverse
			} else if v.Args[0].AuxInt == 0 && v.Args[1].AuxInt == 1 {
				ok, isCopy = true, reverse
			}

			// (Phi (ConstBool [x]) (ConstBool [x])) is already handled by opt / phielim.

			if ok && isCopy {
				if f.pass.debug > 0 {
					f.Config.Warnl(int(b.Line), "converted OpPhi to OpCopy")
				}
				v.reset(OpCopy)
				v.AddArg(b0.Control)
				continue
			}
			if ok && !isCopy {
				if f.pass.debug > 0 {
					f.Config.Warnl(int(b.Line), "converted OpPhi to OpNot")
				}
				v.reset(OpNot)
				v.AddArg(b0.Control)
				continue
			}
		}
	}

}
