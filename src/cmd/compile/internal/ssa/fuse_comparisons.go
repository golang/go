// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// fuseIntInRange transforms integer range checks to remove the short-circuit operator. For example,
// it would convert `if 1 <= x && x < 5 { ... }` into `if (1 <= x) & (x < 5) { ... }`. Rewrite rules
// can then optimize these into unsigned range checks, `if unsigned(x-1) < 4 { ... }` in this case.
func fuseIntInRange(b *ssacore.Block) bool {
	return fuseComparisons(b, canOptIntInRange)
}

// fuseNanCheck replaces the short-circuit operators between NaN checks and comparisons with
// constants. For example, it would transform `if x != x || x > 1.0 { ... }` into
// `if (x != x) | (x > 1.0) { ... }`. Rewrite rules can then merge the NaN check with the comparison,
// in this case generating `if !(x <= 1.0) { ... }`.
func fuseNanCheck(b *ssacore.Block) bool {
	return fuseComparisons(b, canOptNanCheck)
}

// fuseSingleBitDifference replaces the short-circuit operators between equality checks with
// constants that only differ by a single bit. For example, it would convert
// `if x == 4 || x == 6 { ... }` into `if (x == 4) | (x == 6) { ... }`. Rewrite rules can
// then optimize these using a bitwise operation, in this case generating `if x|2 == 6 { ... }`.
func fuseSingleBitDifference(b *ssacore.Block) bool {
	return fuseComparisons(b, canOptSingleBitDifference)
}

// fuseComparisons looks for control graphs that match this pattern:
//
//	p - predecessor
//	|\
//	| b - block
//	|/ \
//	s0 s1 - successors
//
// This pattern is typical for if statements such as `if x || y { ... }` and `if x && y { ... }`.
//
// If canOptControls returns true when passed the control values for p and b then fuseComparisons
// will try to convert p into a plain block with only one successor (b) and modify b's control
// value to include p's control value (effectively causing b to be speculatively executed).
//
// This transformation results in a control graph that will now look like this:
//
//	p
//	 \
//	  b
//	 / \
//	s0 s1
//
// Later passes will then fuse p and b.
//
// In other words `if x || y { ... }` will become `if x | y { ... }` and `if x && y { ... }` will
// become `if x & y { ... }`. This is a useful transformation because we can then use rewrite
// rules to optimize `x | y` and `x & y`.
func fuseComparisons(b *ssacore.Block, canOptControls func(a, b *ssacore.Value, op ssaop.Op) bool) bool {
	if len(b.Preds) != 1 {
		return false
	}
	p := b.Preds[0].Block()
	if b.Kind != block.BlockIf || p.Kind != block.BlockIf {
		return false
	}

	// Don't merge control values if b is likely to be bypassed anyway.
	if p.Likely == ssacore.BranchLikely && p.Succs[0].Block() != b {
		return false
	}
	if p.Likely == ssacore.BranchUnlikely && p.Succs[1].Block() != b {
		return false
	}

	// If the first (true) successors match then we have a disjunction (||).
	// If the second (false) successors match then we have a conjunction (&&).
	for i, op := range [2]ssaop.Op{ssaop.OpOrB, ssaop.OpAndB} {
		if p.Succs[i].Block() != b.Succs[i].Block() {
			continue
		}

		// Check if the control values can be usefully combined.
		bc := b.Controls[0]
		pc := p.Controls[0]
		if !canOptControls(bc, pc, op) {
			return false
		}

		// TODO(mundaym): should we also check the cost of executing b?
		// Currently we might speculatively execute b even if b contains
		// a lot of instructions. We could just check that len(b.Values)
		// is lower than a fixed amount. Bear in mind however that the
		// other optimization passes might yet reduce the cost of b
		// significantly so we shouldn't be overly conservative.
		if !canSpeculativelyExecute(b) {
			return false
		}

		// if the destination block has any phis that are differentiated between
		// the edge being removed and the other edge, removing it will change
		// the behavior of the program
		if hasDifferentiatedPhi(p.Succs[i], b.Succs[i]) {
			continue
		}

		// Logically combine the control values for p and b.
		v := b.NewValue0(bc.Pos, op, bc.Type)
		v.AddArg(pc)
		v.AddArg(bc)

		// Set the combined control value as the control value for b.
		b.SetControl(v)

		// Modify p so that it jumps directly to b.
		p.RemoveEdge(i)
		p.Kind = block.BlockPlain
		p.Likely = ssacore.BranchUnknown
		p.ResetControls()

		return true
	}

	// TODO: could negate condition(s) to merge controls.
	return false
}

func hasDifferentiatedPhi(x ssacore.Edge, y ssacore.Edge) bool {
	b := x.Block()
	if y.Block() != b {
		panic("non matching edges")
	}
	xi := x.I
	yi := y.I
	for _, v := range b.Values {
		if v.Op != ssaop.OpPhi {
			continue
		}
		if v.Args[xi] != v.Args[yi] {
			return true
		}
	}
	return false
}

// getConstIntArgIndex returns the index of the first argument that is a
// constant integer or -1 if no such argument exists.
func getConstIntArgIndex(v *ssacore.Value) int {
	for i, a := range v.Args {
		switch a.Op {
		case ssaop.OpConst8, ssaop.OpConst16, ssaop.OpConst32, ssaop.OpConst64:
			return i
		}
	}
	return -1
}

// isSignedInequality reports whether op represents the inequality < or ≤
// in the signed domain.
func isSignedInequality(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpLess64, ssaop.OpLess32, ssaop.OpLess16, ssaop.OpLess8,
		ssaop.OpLeq64, ssaop.OpLeq32, ssaop.OpLeq16, ssaop.OpLeq8:
		return true
	}
	return false
}

// isUnsignedInequality reports whether op represents the inequality < or ≤
// in the unsigned domain, including "x != 0", which is equivalent to the
// unsigned "0 < x".
func isUnsignedInequality(v *ssacore.Value) bool {
	switch v.Op {
	case ssaop.OpLess64U, ssaop.OpLess32U, ssaop.OpLess16U, ssaop.OpLess8U,
		ssaop.OpLeq64U, ssaop.OpLeq32U, ssaop.OpLeq16U, ssaop.OpLeq8U:
		return true
	case ssaop.OpNeq64, ssaop.OpNeq32, ssaop.OpNeq16, ssaop.OpNeq8:
		// "x != 0" is equivalent to the unsigned "0 < x", and is its
		// canonical form; see "prefer equalities with zero" in generic.rules.
		return isConstZero(v.Args[0]) || isConstZero(v.Args[1])
	}
	return false
}

func canOptIntInRange(x, y *ssacore.Value, op ssaop.Op) bool {
	// We need both inequalities to be either in the signed or unsigned domain.
	// TODO(mundaym): it would also be good to merge when we have an Eq op that
	// could be transformed into a Less/Leq. For example in the unsigned
	// domain 'x == 0 || 3 < x' is equivalent to 'x <= 0 || 3 < x'
	inequalityChecks := [...]func(*ssacore.Value) bool{
		isSignedInequality,
		isUnsignedInequality,
	}
	for _, f := range inequalityChecks {
		if !f(x) || !f(y) {
			continue
		}

		// Check that both inequalities are comparisons with constants.
		xi := getConstIntArgIndex(x)
		if xi < 0 {
			return false
		}
		yi := getConstIntArgIndex(y)
		if yi < 0 {
			return false
		}

		// Check that the non-constant arguments to the inequalities
		// are the same.
		return x.Args[xi^1] == y.Args[yi^1]
	}
	return false
}

// canOptNanCheck reports whether one of arguments is a NaN check and the other
// is a comparison with a constant that can be combined together.
//
// Examples (c must be a constant):
//
//	v != v || v <  c => !(c <= v)
//	v != v || v <= c => !(c <  v)
//	v != v || c <  v => !(v <= c)
//	v != v || c <= v => !(v <  c)
func canOptNanCheck(x, y *ssacore.Value, op ssaop.Op) bool {
	if op != ssaop.OpOrB {
		return false
	}

	for i := 0; i <= 1; i, x, y = i+1, y, x {
		if len(x.Args) != 2 || x.Args[0] != x.Args[1] {
			continue
		}
		v := x.Args[0]
		switch x.Op {
		case ssaop.OpNeq64F:
			if y.Op != ssaop.OpLess64F && y.Op != ssaop.OpLeq64F {
				return false
			}
			for j := 0; j <= 1; j++ {
				a, b := y.Args[j], y.Args[j^1]
				if a.Op != ssaop.OpConst64F {
					continue
				}
				// Sign bit operations not affect NaN check results. This special case allows us
				// to optimize statements like `if v != v || Abs(v) > c { ... }`.
				if (b.Op == ssaop.OpAbs || b.Op == ssaop.OpNeg64F) && b.Args[0] == v {
					return true
				}
				return b == v
			}
		case ssaop.OpNeq32F:
			if y.Op != ssaop.OpLess32F && y.Op != ssaop.OpLeq32F {
				return false
			}
			for j := 0; j <= 1; j++ {
				a, b := y.Args[j], y.Args[j^1]
				if a.Op != ssaop.OpConst32F {
					continue
				}
				// Sign bit operations not affect NaN check results. This special case allows us
				// to optimize statements like `if v != v || -v > c { ... }`.
				if b.Op == ssaop.OpNeg32F && b.Args[0] == v {
					return true
				}
				return b == v
			}
		}
	}
	return false
}

// canOptSingleBitDifference returns true if x op y matches either:
//
//	v == c || v == d
//	v != c && v != d
//
// Where c and d are constant values that differ by a single bit.
func canOptSingleBitDifference(x, y *ssacore.Value, op ssaop.Op) bool {
	if x.Op != y.Op {
		return false
	}
	switch x.Op {
	case ssaop.OpEq64, ssaop.OpEq32, ssaop.OpEq16, ssaop.OpEq8:
		if op != ssaop.OpOrB {
			return false
		}
	case ssaop.OpNeq64, ssaop.OpNeq32, ssaop.OpNeq16, ssaop.OpNeq8:
		if op != ssaop.OpAndB {
			return false
		}
	default:
		return false
	}

	xi := getConstIntArgIndex(x)
	if xi < 0 {
		return false
	}
	yi := getConstIntArgIndex(y)
	if yi < 0 {
		return false
	}
	if x.Args[xi^1] != y.Args[yi^1] {
		return false
	}
	return OneBit(x.Args[xi].AuxInt ^ y.Args[yi].AuxInt)
}
