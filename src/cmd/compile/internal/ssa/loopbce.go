// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"fmt"
)

type indVarFlags uint8

const (
	indVarMinExc    indVarFlags = 1 << iota // minimum value is exclusive (default: inclusive)
	indVarMaxInc                            // maximum value is inclusive (default: exclusive)
	indVarCountDown                         // if set the iteration starts at max and count towards min (default: min towards max)
)

type indVar struct {
	ind   *Value // induction variable
	nxt   *Value // the incremented variable
	min   *Value // minimum value, inclusive/exclusive depends on flags
	max   *Value // maximum value, inclusive/exclusive depends on flags
	entry *Block // entry block in the loop.
	flags indVarFlags
	// Invariant: for all blocks strictly dominated by entry:
	//	min <= ind <  max    [if flags == 0]
	//	min <  ind <  max    [if flags == indVarMinExc]
	//	min <= ind <= max    [if flags == indVarMaxInc]
	//	min <  ind <= max    [if flags == indVarMinExc|indVarMaxInc]
}

// parseIndVar checks whether the SSA value passed as argument is a valid induction
// variable, and, if so, extracts:
//   - the minimum bound
//   - the increment value
//   - the "next" value (SSA value that is Phi'd into the induction variable every loop)
//
// Currently, we detect induction variables that match (Phi min nxt),
// with nxt being (Add inc ind).
// If it can't parse the induction variable correctly, it returns (nil, nil, nil).
func parseIndVar(ind *Value) (min, inc, nxt *Value) {
	if ind.Op != OpPhi {
		return
	}

	if n := ind.Args[0]; (n.Op == OpAdd64 || n.Op == OpAdd32 || n.Op == OpAdd16 || n.Op == OpAdd8) && (n.Args[0] == ind || n.Args[1] == ind) {
		min, nxt = ind.Args[1], n
	} else if n := ind.Args[1]; (n.Op == OpAdd64 || n.Op == OpAdd32 || n.Op == OpAdd16 || n.Op == OpAdd8) && (n.Args[0] == ind || n.Args[1] == ind) {
		min, nxt = ind.Args[0], n
	} else {
		// Not a recognized induction variable.
		return
	}

	if nxt.Args[0] == ind { // nxt = ind + inc
		inc = nxt.Args[1]
	} else if nxt.Args[1] == ind { // nxt = inc + ind
		inc = nxt.Args[0]
	} else {
		panic("unreachable") // one of the cases must be true from the above.
	}

	return
}

// findIndVar finds induction variables in a function.
//
// Look for variables and blocks that satisfy the following
//
//	 loop:
//	   ind = (Phi min nxt),
//	   if ind < max
//	     then goto enter_loop
//	     else goto exit_loop
//
//	   enter_loop:
//		do something
//	      nxt = inc + ind
//		goto loop
//
//	 exit_loop:
func findIndVar(f *Func) []indVar {
	var iv []indVar
	sdom := f.Sdom()

	for _, b := range f.Blocks {
		if b.Kind != BlockIf || len(b.Preds) != 2 {
			continue
		}

		var ind *Value   // induction variable
		var init *Value  // starting value
		var limit *Value // ending value

		// Check that the control if it either ind </<= limit or limit </<= ind.
		// TODO: Handle unsigned comparisons?
		c := b.Controls[0]
		inclusive := false
		switch c.Op {
		case OpLeq64, OpLeq32, OpLeq16, OpLeq8:
			inclusive = true
			fallthrough
		case OpLess64, OpLess32, OpLess16, OpLess8:
			ind, limit = c.Args[0], c.Args[1]
		default:
			continue
		}

		// See if this is really an induction variable
		less := true
		init, inc, nxt := parseIndVar(ind)
		if init == nil {
			// We failed to parse the induction variable. Before punting, we want to check
			// whether the control op was written with the induction variable on the RHS
			// instead of the LHS. This happens for the downwards case, like:
			//     for i := len(n)-1; i >= 0; i--
			init, inc, nxt = parseIndVar(limit)
			if init == nil {
				// No recognized induction variable on either operand
				continue
			}

			// Ok, the arguments were reversed. Swap them, and remember that we're
			// looking at an ind >/>= loop (so the induction must be decrementing).
			ind, limit = limit, ind
			less = false
		}

		if ind.Block != b {
			// TODO: Could be extended to include disjointed loop headers.
			// I don't think this is causing missed optimizations in real world code often.
			// See https://go.dev/issue/63955
			continue
		}

		// Expect the increment to be a nonzero constant.
		if !inc.isGenericIntConst() {
			continue
		}
		step := inc.AuxInt
		if step == 0 {
			continue
		}

		// Increment sign must match comparison direction.
		// When incrementing, the termination comparison must be ind </<= limit.
		// When decrementing, the termination comparison must be ind >/>= limit.
		// See issue 26116.
		if step > 0 && !less {
			continue
		}
		if step < 0 && less {
			continue
		}

		// Up to now we extracted the induction variable (ind),
		// the increment delta (inc), the temporary sum (nxt),
		// the initial value (init) and the limiting value (limit).
		//
		// We also know that ind has the form (Phi init nxt) where
		// nxt is (Add inc nxt) which means: 1) inc dominates nxt
		// and 2) there is a loop starting at inc and containing nxt.
		//
		// We need to prove that the induction variable is incremented
		// only when it's smaller than the limiting value.
		// Two conditions must happen listed below to accept ind
		// as an induction variable.

		// Identify which successor is the *loop entry* (where nxt is computed)
		var entry *Block
		if sdom.IsAncestorEq(b.Succs[0].b, nxt.Block) {
			entry = b.Succs[0].b
		} else if sdom.IsAncestorEq(b.Succs[1].b, nxt.Block) {
			entry = b.Succs[1].b
		} else {
			// Neither successor dominates nxt; shape is not a simple for-loop header.
			continue
		}

		// First condition: loop entry has exactly one predecessor (the header).
		if len(entry.Preds) != 1 {
			continue
		}

		// Second condition: entry must dominate nxt so that nxt = ind+inc
		// is only reachable when the loop is taken.
		if !sdom.IsAncestorEq(entry, nxt.Block) {
			continue
		}

		// Check for overflow/underflow. We need to make sure that inc never causes
		// the induction variable to wrap around.
		// We use a function wrapper here for easy return true / return false / keep going logic.
		// This function returns true if the increment will never overflow/underflow.
		ok := func() bool {
			if step > 0 {
				if limit.isGenericIntConst() {
					// Figure out the actual largest value.
					v := limit.AuxInt
					if !inclusive {
						if v == minSignedValue(limit.Type) {
							return false // < minint is never satisfiable.
						}
						v--
					}
					if init.isGenericIntConst() {
						// Use stride to compute a better lower limit.
						if init.AuxInt > v {
							return false
						}
						v = addU(init.AuxInt, diff(v, init.AuxInt)/uint64(step)*uint64(step))
					}
					if addWillOverflow(v, step) {
						return false
					}
					if inclusive && v != limit.AuxInt || !inclusive && v+1 != limit.AuxInt {
						// We know a better limit than the programmer did. Use our limit instead.
						limit = f.constVal(limit.Op, limit.Type, v, true)
						inclusive = true
					}
					return true
				}
				if step == 1 && !inclusive {
					// Can't overflow because maxint is never a possible value.
					return true
				}
				// If the limit is not a constant, check to see if it is a
				// negative offset from a known non-negative value.
				knn, k := findKNN(limit)
				if knn == nil || k < 0 {
					return false
				}
				// limit == (something nonnegative) - k. That subtraction can't underflow, so
				// we can trust it.
				if inclusive {
					// ind <= knn - k cannot overflow if step is at most k
					return step <= k
				}
				// ind < knn - k cannot overflow if step is at most k+1
				return step <= k+1 && k != maxSignedValue(limit.Type)
			} else { // step < 0
				if limit.Op == OpConst64 {
					// Figure out the actual smallest value.
					v := limit.AuxInt
					if !inclusive {
						if v == maxSignedValue(limit.Type) {
							return false // > maxint is never satisfiable.
						}
						v++
					}
					if init.isGenericIntConst() {
						// Use stride to compute a better lower limit.
						if init.AuxInt < v {
							return false
						}
						v = subU(init.AuxInt, diff(init.AuxInt, v)/uint64(-step)*uint64(-step))
					}
					if subWillUnderflow(v, -step) {
						return false
					}
					if inclusive && v != limit.AuxInt || !inclusive && v-1 != limit.AuxInt {
						// We know a better limit than the programmer did. Use our limit instead.
						limit = f.constVal(limit.Op, limit.Type, v, true)
						inclusive = true
					}
					return true
				}
				if step == -1 && !inclusive {
					// Can't underflow because minint is never a possible value.
					return true
				}
			}
			return false

		}

		if ok() {
			flags := indVarFlags(0)
			var min, max *Value
			if step > 0 {
				min = init
				max = limit
				if inclusive {
					flags |= indVarMaxInc
				}
			} else {
				min = limit
				max = init
				flags |= indVarMaxInc
				if !inclusive {
					flags |= indVarMinExc
				}
				flags |= indVarCountDown
				step = -step
			}
			if f.pass.debug >= 1 {
				printIndVar(b, ind, min, max, step, flags)
			}

			iv = append(iv, indVar{
				ind:   ind,
				nxt:   nxt,
				min:   min,
				max:   max,
				entry: entry,
				flags: flags,
			})
			b.Logf("found induction variable %v (inc = %v, min = %v, max = %v)\n", ind, inc, min, max)
		}

		// TODO: other unrolling idioms
		// for i := 0; i < KNN - KNN % k ; i += k
		// for i := 0; i < KNN&^(k-1) ; i += k // k a power of 2
		// for i := 0; i < KNN&(-k) ; i += k // k a power of 2
	}

	return iv
}

// addWillOverflow reports whether x+y would result in a value more than maxint.
func addWillOverflow(x, y int64) bool {
	return x+y < x
}

// subWillUnderflow reports whether x-y would result in a value less than minint.
func subWillUnderflow(x, y int64) bool {
	return x-y > x
}

// diff returns x-y as a uint64. Requires x>=y.
func diff(x, y int64) uint64 {
	if x < y {
		base.Fatalf("diff %d - %d underflowed", x, y)
	}
	return uint64(x - y)
}

// addU returns x+y. Requires that x+y does not overflow an int64.
func addU(x int64, y uint64) int64 {
	if y >= 1<<63 {
		if x >= 0 {
			base.Fatalf("addU overflowed %d + %d", x, y)
		}
		x += 1<<63 - 1
		x += 1
		y -= 1 << 63
	}
	if addWillOverflow(x, int64(y)) {
		base.Fatalf("addU overflowed %d + %d", x, y)
	}
	return x + int64(y)
}

// subU returns x-y. Requires that x-y does not underflow an int64.
func subU(x int64, y uint64) int64 {
	if y >= 1<<63 {
		if x < 0 {
			base.Fatalf("subU underflowed %d - %d", x, y)
		}
		x -= 1<<63 - 1
		x -= 1
		y -= 1 << 63
	}
	if subWillUnderflow(x, int64(y)) {
		base.Fatalf("subU underflowed %d - %d", x, y)
	}
	return x - int64(y)
}

// if v is known to be x - c, where x is known to be nonnegative and c is a
// constant, return x, c. Otherwise return nil, 0.
func findKNN(v *Value) (*Value, int64) {
	var x, y *Value
	x = v
	switch v.Op {
	case OpSub64, OpSub32, OpSub16, OpSub8:
		x = v.Args[0]
		y = v.Args[1]

	case OpAdd64, OpAdd32, OpAdd16, OpAdd8:
		x = v.Args[0]
		y = v.Args[1]
		if x.isGenericIntConst() {
			x, y = y, x
		}
	}
	switch x.Op {
	case OpSliceLen, OpStringLen, OpSliceCap:
	default:
		return nil, 0
	}
	if y == nil {
		return x, 0
	}
	if !y.isGenericIntConst() {
		return nil, 0
	}
	if v.Op == OpAdd64 || v.Op == OpAdd32 || v.Op == OpAdd16 || v.Op == OpAdd8 {
		return x, -y.AuxInt
	}
	return x, y.AuxInt
}

func printIndVar(b *Block, i, min, max *Value, inc int64, flags indVarFlags) {
	mb1, mb2 := "[", "]"
	if flags&indVarMinExc != 0 {
		mb1 = "("
	}
	if flags&indVarMaxInc == 0 {
		mb2 = ")"
	}

	mlim1, mlim2 := fmt.Sprint(min.AuxInt), fmt.Sprint(max.AuxInt)
	if !min.isGenericIntConst() {
		if b.Func.pass.debug >= 2 {
			mlim1 = fmt.Sprint(min)
		} else {
			mlim1 = "?"
		}
	}
	if !max.isGenericIntConst() {
		if b.Func.pass.debug >= 2 {
			mlim2 = fmt.Sprint(max)
		} else {
			mlim2 = "?"
		}
	}
	extra := ""
	if b.Func.pass.debug >= 2 {
		extra = fmt.Sprintf(" (%s)", i)
	}
	b.Func.Warnl(b.Pos, "Induction variable: limits %v%v,%v%v, increment %d%s", mb1, mlim1, mlim2, mb2, inc, extra)
}

func minSignedValue(t *types.Type) int64 {
	return -1 << (t.Size()*8 - 1)
}

func maxSignedValue(t *types.Type) int64 {
	return 1<<((t.Size()*8)-1) - 1
}
