package ssa

import "fmt"

type indVarFlags uint8

const (
	indVarMinExc indVarFlags = 1 << iota // minimum value is exclusive (default: inclusive)
	indVarMaxInc                         // maximum value is inclusive (default: exclusive)
)

type indVar struct {
	ind   *Value // induction variable
	inc   *Value // increment, a constant
	nxt   *Value // ind+inc variable
	min   *Value // minimum value, inclusive/exclusive depends on flags
	max   *Value // maximum value, inclusive/exclusive depends on flags
	entry *Block // entry block in the loop.
	flags indVarFlags
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
	sdom := f.sdom()

nextb:
	for _, b := range f.Blocks {
		if b.Kind != BlockIf || len(b.Preds) != 2 {
			continue
		}

		var flags indVarFlags
		var ind, max *Value // induction, and maximum
		entry := -1         // which successor of b enters the loop

		// Check thet the control if it either ind </<= max or max >/>= ind.
		// TODO: Handle 32-bit comparisons.
		switch b.Control.Op {
		case OpLeq64:
			flags |= indVarMaxInc
			fallthrough
		case OpLess64:
			entry = 0
			ind, max = b.Control.Args[0], b.Control.Args[1]
		case OpGeq64:
			flags |= indVarMaxInc
			fallthrough
		case OpGreater64:
			entry = 0
			ind, max = b.Control.Args[1], b.Control.Args[0]
		default:
			continue nextb
		}

		// See if the arguments are reversed (i < len() <=> len() > i)
		if max.Op == OpPhi {
			ind, max = max, ind
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

		// Expect the increment to be a constant.
		if inc.Op != OpConst64 {
			continue
		}

		// If the increment is negative, swap min/max and their flags
		if inc.AuxInt <= 0 {
			min, max = max, min
			oldf := flags
			flags = 0
			if oldf&indVarMaxInc == 0 {
				flags |= indVarMinExc
			}
			if oldf&indVarMinExc == 0 {
				flags |= indVarMaxInc
			}
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
		if !sdom.isAncestorEq(b.Succs[entry].b, nxt.Block) {
			// inc+ind can only be reached through the branch that enters the loop.
			continue
		}

		// We can only guarantee that the loops runs within limits of induction variable
		// if the increment is ±1 or when the limits are constants.
		if inc.AuxInt != 1 && inc.AuxInt != -1 {
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

		if f.pass.debug >= 1 {
			mb1, mb2 := "[", "]"
			if flags&indVarMinExc != 0 {
				mb1 = "("
			}
			if flags&indVarMaxInc == 0 {
				mb2 = ")"
			}

			mlim1, mlim2 := fmt.Sprint(min.AuxInt), fmt.Sprint(max.AuxInt)
			if !min.isGenericIntConst() {
				if f.pass.debug >= 2 {
					mlim1 = fmt.Sprint(min)
				} else {
					mlim1 = "?"
				}
			}
			if !max.isGenericIntConst() {
				if f.pass.debug >= 2 {
					mlim2 = fmt.Sprint(max)
				} else {
					mlim2 = "?"
				}
			}
			b.Func.Warnl(b.Pos, "Induction variable: limits %v%v,%v%v, increment %d", mb1, mlim1, mlim2, mb2, inc.AuxInt)
		}

		iv = append(iv, indVar{
			ind:   ind,
			inc:   inc,
			nxt:   nxt,
			min:   min,
			max:   max,
			entry: b.Succs[entry].b,
			flags: flags,
		})
		b.Logf("found induction variable %v (inc = %v, min = %v, max = %v)\n", ind, inc, min, max)
	}

	return iv
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
