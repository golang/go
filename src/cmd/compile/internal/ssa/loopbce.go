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
	indVarMinExc indVarFlags = 1 << iota // minimum value is exclusive (default: inclusive)
	indVarMaxInc                         // maximum value is inclusive (default: exclusive)
)

type indVar struct {
	ind   *Value // induction variable
	nxt   *Value // the incremented variable
	min   *Value // minimum value, inclusive/exclusive depends on flags
	max   *Value // maximum value, inclusive/exclusive depends on flags
	entry *Block // the block where the edge from the succeeded comparison of the induction variable goes to, means when the bound check has passed.
	step  int64  // it will always be positive.
	flags indVarFlags
	// Invariant: for all blocks dominated by entry:
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
//   - the header's edge returning from the body
//
// Currently, we detect induction variables that match (Phi min nxt),
// with nxt being (Add inc ind).
// If it can't parse the induction variable correctly, it returns (nil, nil, nil).
func parseIndVar(ind *Value) (min, inc, nxt *Value, loopReturn Edge) {
	if ind.Op != OpPhi {
		return
	}

	if n := ind.Args[0]; (n.Op == OpAdd64 || n.Op == OpAdd32 || n.Op == OpAdd16 || n.Op == OpAdd8) && (n.Args[0] == ind || n.Args[1] == ind) {
		min, nxt, loopReturn = ind.Args[1], n, ind.Block.Preds[0]
	} else if n := ind.Args[1]; (n.Op == OpAdd64 || n.Op == OpAdd32 || n.Op == OpAdd16 || n.Op == OpAdd8) && (n.Args[0] == ind || n.Args[1] == ind) {
		min, nxt, loopReturn = ind.Args[0], n, ind.Block.Preds[1]
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
//
// We may have more than one induction variables, the loop in the go
// source code may looks like this:
//
//	for i >= 0 && j >= 0 {
//		// use i and j
//		i--
//		j--
//	}
//
// So, also look for variables and blocks that satisfy the following
//
//	loop:
//	  i = (Phi maxi nxti)
//	  j = (Phi maxj nxtj)
//	  if i >= mini
//	    then goto check_j
//	    else goto exit_loop
//
//	check_j:
//	  if j >= minj
//	    then goto enter_loop
//	    else goto exit_loop
//
//	enter_loop:
//	  do something
//	  nxti = i - di
//	  nxtj = j - dj
//	  goto loop
//
//	exit_loop:
func findIndVar(f *Func) []indVar {
	var iv []indVar
	sdom := f.Sdom()

nextblock:
	for _, b := range f.Blocks {
		if b.Kind != BlockIf {
			continue
		}
		c := b.Controls[0]
		for idx := range 2 {
			// Check that the control if it either ind </<= limit or limit </<= ind.
			// TODO: Handle unsigned comparisons?
			inclusive := false
			switch c.Op {
			case OpLeq64, OpLeq32, OpLeq16, OpLeq8:
				inclusive = true
			case OpLess64, OpLess32, OpLess16, OpLess8:
			default:
				continue nextblock
			}

			less := idx == 0
			// induction variable, ending value
			ind, limit := c.Args[idx], c.Args[1-idx]
			// starting value, increment value, next value, loop return edge
			init, inc, nxt, loopReturn := parseIndVar(ind)
			if init == nil {
				continue // this is not an induction variable
			}

			// This is ind.Block.Preds, not b.Preds. That's a restriction on the loop header,
			// not the comparison block.
			if len(ind.Block.Preds) != 2 {
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
			// step == minInt64 cannot be safely negated below, because -step
			// overflows back to minInt64. The later underflow checks need a
			// positive magnitude, so reject this case here.
			if step == minSignedValue(ind.Type) {
				continue
			}

			// startBody is the edge that eventually returns to the loop header.
			var startBody Edge
			switch {
			case sdom.IsAncestorEq(b.Succs[0].b, loopReturn.b):
				startBody = b.Succs[0]
			case sdom.IsAncestorEq(b.Succs[1].b, loopReturn.b):
				// if x { goto exit } else { goto entry } is identical to if !x { goto entry } else { goto exit }
				startBody = b.Succs[1]
				less = !less
				inclusive = !inclusive
			default:
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

			// First condition: the entry block has a single predecessor.
			// The entry now means the in-loop edge where the induction variable
			// comparison succeeded. Its predecessor is not necessarily the header
			// block. This implies that b.Succs[0] is reached iff ind < limit.
			if len(startBody.b.Preds) != 1 {
				// the other successor must exit the loop.
				continue
			}

			// Second condition: startBody.b dominates nxt so that
			// nxt is computed when inc < limit.
			if !sdom.IsAncestorEq(startBody.b, nxt.Block) {
				// inc+ind can only be reached through the branch that confirmed the
				// induction variable is in bounds.
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
							// TODO(1.27): investigate passing a smaller-magnitude overflow limit to addU
							// for addWillOverflow.
							v = addU(init.AuxInt, diff(v, init.AuxInt)/uint64(step)*uint64(step))
						}
						if addWillOverflow(v, step, maxSignedValue(ind.Type)) {
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

					// TODO: other unrolling idioms
					// for i := 0; i < KNN - KNN % k ; i += k
					// for i := 0; i < KNN&^(k-1) ; i += k // k a power of 2
					// for i := 0; i < KNN&(-k) ; i += k // k a power of 2
				} else { // step < 0
					if limit.isGenericIntConst() {
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
							// TODO(1.27): investigate passing a smaller-magnitude underflow limit to subU
							// for subWillUnderflow.
							v = subU(init.AuxInt, diff(init.AuxInt, v)/uint64(-step)*uint64(-step))
						}
						if subWillUnderflow(v, -step, minSignedValue(ind.Type)) {
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
					step = -step
				}
				if f.pass.debug >= 1 {
					printIndVar(b, ind, min, max, step, flags)
				}

				iv = append(iv, indVar{
					ind: ind,
					nxt: nxt,
					min: min,
					max: max,
					// This is startBody.b, where startBody is the edge from the comparison for the
					// induction variable, not necessarily the in-loop edge from the loop header.
					// Induction variable bounds are not valid in the loop before this edge.
					entry: startBody.b,
					step:  step,
					flags: flags,
				})
				b.Logf("found induction variable %v (inc = %v, min = %v, max = %v)\n", ind, inc, min, max)
			}
		}
	}

	return iv
}

// subWillUnderflow checks if x - y underflows the min value.
// y must be positive.
func subWillUnderflow(x, y int64, min int64) bool {
	if y < 0 {
		base.Fatalf("expecting positive value")
	}
	return x < min+y
}

// addWillOverflow checks if x + y overflows the max value.
// y must be positive.
func addWillOverflow(x, y int64, max int64) bool {
	if y < 0 {
		base.Fatalf("expecting positive value")
	}
	return x > max-y
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
	// TODO(1.27): investigate passing a smaller-magnitude overflow limit in here.
	if addWillOverflow(x, int64(y), maxSignedValue(types.Types[types.TINT64])) {
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
	// TODO(1.27): investigate passing a smaller-magnitude underflow limit in here.
	if subWillUnderflow(x, int64(y), minSignedValue(types.Types[types.TINT64])) {
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
