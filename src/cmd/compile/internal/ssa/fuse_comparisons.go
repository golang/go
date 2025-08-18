// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// fuseIntInRange transforms integer range checks to remove the short-circuit operator. For example,
// it would convert `if 1 <= x && x < 5 { ... }` into `if (1 <= x) & (x < 5) { ... }`. Rewrite rules
// can then optimize these into unsigned range checks, `if unsigned(x-1) < 4 { ... }` in this case.
func fuseIntInRange(b *Block) bool {
	return fuseComparisons(b, canOptIntInRange)
}

// fuseNanCheck replaces the short-circuit operators between NaN checks and comparisons with
// constants. For example, it would transform `if x != x || x > 1.0 { ... }` into
// `if (x != x) | (x > 1.0) { ... }`. Rewrite rules can then merge the NaN check with the comparison,
// in this case generating `if !(x <= 1.0) { ... }`.
func fuseNanCheck(b *Block) bool {
	return fuseComparisons(b, canOptNanCheck)
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
func fuseComparisons(b *Block, canOptControls func(a, b *Value, op Op) bool) bool {
	if len(b.Preds) != 1 {
		return false
	}
	p := b.Preds[0].Block()
	if b.Kind != BlockIf || p.Kind != BlockIf {
		return false
	}

	// Don't merge control values if b is likely to be bypassed anyway.
	if p.Likely == BranchLikely && p.Succs[0].Block() != b {
		return false
	}
	if p.Likely == BranchUnlikely && p.Succs[1].Block() != b {
		return false
	}

	// If the first (true) successors match then we have a disjunction (||).
	// If the second (false) successors match then we have a conjunction (&&).
	for i, op := range [2]Op{OpOrB, OpAndB} {
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

		// Logically combine the control values for p and b.
		v := b.NewValue0(bc.Pos, op, bc.Type)
		v.AddArg(pc)
		v.AddArg(bc)

		// Set the combined control value as the control value for b.
		b.SetControl(v)

		// Modify p so that it jumps directly to b.
		p.removeEdge(i)
		p.Kind = BlockPlain
		p.Likely = BranchUnknown
		p.ResetControls()

		return true
	}

	// TODO: could negate condition(s) to merge controls.
	return false
}

// getConstIntArgIndex returns the index of the first argument that is a
// constant integer or -1 if no such argument exists.
func getConstIntArgIndex(v *Value) int {
	for i, a := range v.Args {
		switch a.Op {
		case OpConst8, OpConst16, OpConst32, OpConst64:
			return i
		}
	}
	return -1
}

// isSignedInequality reports whether op represents the inequality < or ≤
// in the signed domain.
func isSignedInequality(v *Value) bool {
	switch v.Op {
	case OpLess64, OpLess32, OpLess16, OpLess8,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8:
		return true
	}
	return false
}

// isUnsignedInequality reports whether op represents the inequality < or ≤
// in the unsigned domain.
func isUnsignedInequality(v *Value) bool {
	switch v.Op {
	case OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U:
		return true
	}
	return false
}

func canOptIntInRange(x, y *Value, op Op) bool {
	// We need both inequalities to be either in the signed or unsigned domain.
	// TODO(mundaym): it would also be good to merge when we have an Eq op that
	// could be transformed into a Less/Leq. For example in the unsigned
	// domain 'x == 0 || 3 < x' is equivalent to 'x <= 0 || 3 < x'
	inequalityChecks := [...]func(*Value) bool{
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
func canOptNanCheck(x, y *Value, op Op) bool {
	if op != OpOrB {
		return false
	}

	for i := 0; i <= 1; i, x, y = i+1, y, x {
		if len(x.Args) != 2 || x.Args[0] != x.Args[1] {
			continue
		}
		v := x.Args[0]
		switch x.Op {
		case OpNeq64F:
			if y.Op != OpLess64F && y.Op != OpLeq64F {
				return false
			}
			for j := 0; j <= 1; j++ {
				a, b := y.Args[j], y.Args[j^1]
				if a.Op != OpConst64F {
					continue
				}
				// Sign bit operations not affect NaN check results. This special case allows us
				// to optimize statements like `if v != v || Abs(v) > c { ... }`.
				if (b.Op == OpAbs || b.Op == OpNeg64F) && b.Args[0] == v {
					return true
				}
				return b == v
			}
		case OpNeq32F:
			if y.Op != OpLess32F && y.Op != OpLeq32F {
				return false
			}
			for j := 0; j <= 1; j++ {
				a, b := y.Args[j], y.Args[j^1]
				if a.Op != OpConst32F {
					continue
				}
				// Sign bit operations not affect NaN check results. This special case allows us
				// to optimize statements like `if v != v || -v > c { ... }`.
				if b.Op == OpNeg32F && b.Args[0] == v {
					return true
				}
				return b == v
			}
		}
	}
	return false
}
