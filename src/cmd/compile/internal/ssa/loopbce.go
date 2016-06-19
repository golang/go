package ssa

type indVar struct {
	ind   *Value // induction variable
	inc   *Value // increment, a constant
	nxt   *Value // ind+inc variable
	min   *Value // minimum value. inclusive,
	max   *Value // maximum value. exclusive.
	entry *Block // entry block in the loop.
	// Invariants: for all blocks dominated by entry:
	//	min <= ind < max
	//	min <= nxt <= max
}

// findIndVar finds induction variables in a function.
//
// Look for variables and blocks that satisfy the following
//
// loop:
//   ind = (Phi min nxt),
//   if ind < max
//     then goto enter_loop
//     else goto exit_loop
//
//   enter_loop:
//	do something
//      nxt = inc + ind
//	goto loop
//
// exit_loop:
//
//
// TODO: handle 32 bit operations
func findIndVar(f *Func) []indVar {
	var iv []indVar

nextb:
	for _, b := range f.Blocks {
		if b.Kind != BlockIf || len(b.Preds) != 2 {
			continue
		}

		var ind, max *Value // induction, and maximum
		entry := -1         // which successor of b enters the loop

		// Check thet the control if it either ind < max or max > ind.
		// TODO: Handle Leq64, Geq64.
		switch b.Control.Op {
		case OpLess64:
			entry = 0
			ind, max = b.Control.Args[0], b.Control.Args[1]
		case OpGreater64:
			entry = 0
			ind, max = b.Control.Args[1], b.Control.Args[0]
		default:
			continue nextb
		}

		// Check that the induction variable is a phi that depends on itself.
		if ind.Op != OpPhi {
			continue
		}

		// Extract min and nxt knowing that nxt is an addition (e.g. Add64).
		var min, nxt *Value // minimum, and next value
		if n := ind.Args[0]; n.Op == OpAdd64 && (n.Args[0] == ind || n.Args[1] == ind) {
			min, nxt = ind.Args[1], n
		} else if n := ind.Args[1]; n.Op == OpAdd64 && (n.Args[0] == ind || n.Args[1] == ind) {
			min, nxt = ind.Args[0], n
		} else {
			// Not a recognized induction variable.
			continue
		}

		var inc *Value
		if nxt.Args[0] == ind { // nxt = ind + inc
			inc = nxt.Args[1]
		} else if nxt.Args[1] == ind { // nxt = inc + ind
			inc = nxt.Args[0]
		} else {
			panic("unreachable") // one of the cases must be true from the above.
		}

		// Expect the increment to be a positive constant.
		// TODO: handle negative increment.
		if inc.Op != OpConst64 || inc.AuxInt <= 0 {
			continue
		}

		// Up to now we extracted the induction variable (ind),
		// the increment delta (inc), the temporary sum (nxt),
		// the mininum value (min) and the maximum value (max).
		//
		// We also know that ind has the form (Phi min nxt) where
		// nxt is (Add inc nxt) which means: 1) inc dominates nxt
		// and 2) there is a loop starting at inc and containing nxt.
		//
		// We need to prove that the induction variable is incremented
		// only when it's smaller than the maximum value.
		// Two conditions must happen listed below to accept ind
		// as an induction variable.

		// First condition: loop entry has a single predecessor, which
		// is the header block.  This implies that b.Succs[entry] is
		// reached iff ind < max.
		if len(b.Succs[entry].b.Preds) != 1 {
			// b.Succs[1-entry] must exit the loop.
			continue
		}

		// Second condition: b.Succs[entry] dominates nxt so that
		// nxt is computed when inc < max, meaning nxt <= max.
		if !f.sdom.isAncestorEq(b.Succs[entry].b, nxt.Block) {
			// inc+ind can only be reached through the branch that enters the loop.
			continue
		}

		// If max is c + SliceLen with c <= 0 then we drop c.
		// Makes sure c + SliceLen doesn't overflow when SliceLen == 0.
		// TODO: save c as an offset from max.
		if w, c := dropAdd64(max); (w.Op == OpStringLen || w.Op == OpSliceLen) && 0 >= c && -c >= 0 {
			max = w
		}

		// We can only guarantee that the loops runs within limits of induction variable
		// if the increment is 1 or when the limits are constants.
		if inc.AuxInt != 1 {
			ok := false
			if min.Op == OpConst64 && max.Op == OpConst64 {
				if max.AuxInt > min.AuxInt && max.AuxInt%inc.AuxInt == min.AuxInt%inc.AuxInt { // handle overflow
					ok = true
				}
			}
			if !ok {
				continue
			}
		}

		if f.pass.debug > 1 {
			if min.Op == OpConst64 {
				b.Func.Config.Warnl(b.Line, "Induction variable with minimum %d and increment %d", min.AuxInt, inc.AuxInt)
			} else {
				b.Func.Config.Warnl(b.Line, "Induction variable with non-const minimum and increment %d", inc.AuxInt)
			}
		}

		iv = append(iv, indVar{
			ind:   ind,
			inc:   inc,
			nxt:   nxt,
			min:   min,
			max:   max,
			entry: b.Succs[entry].b,
		})
		b.Logf("found induction variable %v (inc = %v, min = %v, max = %v)\n", ind, inc, min, max)
	}

	return iv
}

// loopbce performs loop based bounds check elimination.
func loopbce(f *Func) {
	ivList := findIndVar(f)

	m := make(map[*Value]indVar)
	for _, iv := range ivList {
		m[iv.ind] = iv
	}

	removeBoundsChecks(f, m)
}

// removesBoundsChecks remove IsInBounds and IsSliceInBounds based on the induction variables.
func removeBoundsChecks(f *Func, m map[*Value]indVar) {
	for _, b := range f.Blocks {
		if b.Kind != BlockIf {
			continue
		}

		v := b.Control

		// Simplify:
		// (IsInBounds ind max) where 0 <= const == min <= ind < max.
		// (IsSliceInBounds ind max) where 0 <= const == min <= ind < max.
		// Found in:
		//	for i := range a {
		//		use a[i]
		//		use a[i:]
		//		use a[:i]
		//	}
		if v.Op == OpIsInBounds || v.Op == OpIsSliceInBounds {
			ind, add := dropAdd64(v.Args[0])
			if ind.Op != OpPhi {
				goto skip1
			}
			if v.Op == OpIsInBounds && add != 0 {
				goto skip1
			}
			if v.Op == OpIsSliceInBounds && (0 > add || add > 1) {
				goto skip1
			}

			if iv, has := m[ind]; has && f.sdom.isAncestorEq(iv.entry, b) && isNonNegative(iv.min) {
				if v.Args[1] == iv.max {
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "Found redundant %s", v.Op)
					}
					goto simplify
				}
			}
		}
	skip1:

		// Simplify:
		// (IsSliceInBounds ind (SliceCap a)) where 0 <= min <= ind < max == (SliceLen a)
		// Found in:
		//	for i := range a {
		//		use a[:i]
		//		use a[:i+1]
		//	}
		if v.Op == OpIsSliceInBounds {
			ind, add := dropAdd64(v.Args[0])
			if ind.Op != OpPhi {
				goto skip2
			}
			if 0 > add || add > 1 {
				goto skip2
			}

			if iv, has := m[ind]; has && f.sdom.isAncestorEq(iv.entry, b) && isNonNegative(iv.min) {
				if v.Args[1].Op == OpSliceCap && iv.max.Op == OpSliceLen && v.Args[1].Args[0] == iv.max.Args[0] {
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "Found redundant %s (len promoted to cap)", v.Op)
					}
					goto simplify
				}
			}
		}
	skip2:

		// Simplify
		// (IsInBounds (Add64 ind) (Const64 [c])) where 0 <= min <= ind < max <= (Const64 [c])
		// (IsSliceInBounds ind (Const64 [c])) where 0 <= min <= ind < max <= (Const64 [c])
		if v.Op == OpIsInBounds || v.Op == OpIsSliceInBounds {
			ind, add := dropAdd64(v.Args[0])
			if ind.Op != OpPhi {
				goto skip3
			}

			// ind + add >= 0 <-> min + add >= 0 <-> min >= -add
			if iv, has := m[ind]; has && f.sdom.isAncestorEq(iv.entry, b) && isGreaterOrEqualThan(iv.min, -add) {
				if !v.Args[1].isGenericIntConst() || !iv.max.isGenericIntConst() {
					goto skip3
				}

				limit := v.Args[1].AuxInt
				if v.Op == OpIsSliceInBounds {
					// If limit++ overflows signed integer then 0 <= max && max <= limit will be false.
					limit++
				}

				if max := iv.max.AuxInt + add; 0 <= max && max <= limit { // handle overflow
					if f.pass.debug > 0 {
						f.Config.Warnl(b.Line, "Found redundant (%s ind %d), ind < %d", v.Op, v.Args[1].AuxInt, iv.max.AuxInt+add)
					}
					goto simplify
				}
			}
		}
	skip3:

		continue

	simplify:
		f.Logf("removing bounds check %v at %v in %s\n", b.Control, b, f.Name)
		b.Kind = BlockFirst
		b.SetControl(nil)
	}
}

func dropAdd64(v *Value) (*Value, int64) {
	if v.Op == OpAdd64 && v.Args[0].Op == OpConst64 {
		return v.Args[1], v.Args[0].AuxInt
	}
	if v.Op == OpAdd64 && v.Args[1].Op == OpConst64 {
		return v.Args[0], v.Args[1].AuxInt
	}
	return v, 0
}

func isGreaterOrEqualThan(v *Value, c int64) bool {
	if c == 0 {
		return isNonNegative(v)
	}
	if v.isGenericIntConst() && v.AuxInt >= c {
		return true
	}
	return false
}
