// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/base"

const logosCorrelatedMaxPredecessors = 8

// logosCorrelatedConstants folds an integer operation when it has the same
// constant result on every incoming edge. Valid SSA aligns the arguments of
// same-block phis by predecessor, so their values must be paired by that index.
// Evaluating their independent sets of possible values would lose this fact.
//
// Only scalar constants and one level of same-block phis are inspected. The
// predecessor cap bounds the work per candidate; loops are never unfolded.
func logosCorrelatedConstants(f *Func) {
	if base.Debug.LogosOpt <= 0 {
		return
	}
	for _, b := range f.Blocks {
		if len(b.Preds) < 2 || len(b.Preds) > logosCorrelatedMaxPredecessors {
			continue
		}
	values:
		for _, v := range b.Values {
			if !logosCorrelatedOp(v.Op) {
				continue
			}
			x, y := v.Args[0], v.Args[1]
			if x.Op != OpPhi && y.Op != OpPhi {
				continue
			}
			var result lattice
			for i := range b.Preds {
				xc := logosCorrelatedArg(x, b, i)
				yc := logosCorrelatedArg(y, b, i)
				if xc == nil || yc == nil {
					continue values
				}
				// Reuse the generic rules for exact width, wraparound, and
				// signed/unsigned comparison semantics. No division, shifts,
				// floating-point, pointer, or memory operations are admitted.
				folded := computeLattice(f, v, xc, yc)
				if folded.tag != constant || i != 0 && !equals(result, folded) {
					continue values
				}
				result = folded
			}
			v.reset(result.val.Op)
			v.AuxInt = result.val.AuxInt
			if f.pass.debug > 0 {
				f.Warnl(v.Pos, "correlated constant %v: %v", v, v.AuxInt)
			}
		}
	}
}

func logosCorrelatedArg(v *Value, b *Block, predecessor int) *Value {
	if v.Op == OpPhi {
		if v.Block != b {
			return nil
		}
		v = v.Args[predecessor]
	}
	switch v.Op {
	case OpConst8, OpConst16, OpConst32, OpConst64, OpConstBool:
		return v
	default:
		return nil
	}
}

func logosCorrelatedOp(op Op) bool {
	switch op {
	case OpAdd8, OpAdd16, OpAdd32, OpAdd64,
		OpSub8, OpSub16, OpSub32, OpSub64,
		OpMul8, OpMul16, OpMul32, OpMul64,
		OpAnd8, OpAnd16, OpAnd32, OpAnd64,
		OpOr8, OpOr16, OpOr32, OpOr64,
		OpXor8, OpXor16, OpXor32, OpXor64,
		OpEq8, OpEq16, OpEq32, OpEq64, OpEqB,
		OpNeq8, OpNeq16, OpNeq32, OpNeq64, OpNeqB,
		OpLess8, OpLess16, OpLess32, OpLess64,
		OpLess8U, OpLess16U, OpLess32U, OpLess64U,
		OpLeq8, OpLeq16, OpLeq32, OpLeq64,
		OpLeq8U, OpLeq16U, OpLeq32U, OpLeq64U:
		return true
	default:
		return false
	}
}
