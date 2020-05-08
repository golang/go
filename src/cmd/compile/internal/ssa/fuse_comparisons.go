// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// fuseIntegerComparisons optimizes inequalities such as '1 <= x && x < 5',
// which can be optimized to 'unsigned(x-1) < 4'.
//
// Look for branch structure like:
//
//   p
//   |\
//   | b
//   |/ \
//   s0 s1
//
// In our example, p has control '1 <= x', b has control 'x < 5',
// and s0 and s1 are the if and else results of the comparison.
//
// This will be optimized into:
//
//   p
//    \
//     b
//    / \
//   s0 s1
//
// where b has the combined control value 'unsigned(x-1) < 4'.
// Later passes will then fuse p and b.
func fuseIntegerComparisons(b *Block) bool {
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

	// Check if the control values combine to make an integer inequality that
	// can be further optimized later.
	bc := b.Controls[0]
	pc := p.Controls[0]
	if !areMergeableInequalities(bc, pc) {
		return false
	}

	// If the first (true) successors match then we have a disjunction (||).
	// If the second (false) successors match then we have a conjunction (&&).
	for i, op := range [2]Op{OpOrB, OpAndB} {
		if p.Succs[i].Block() != b.Succs[i].Block() {
			continue
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

func areMergeableInequalities(x, y *Value) bool {
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
