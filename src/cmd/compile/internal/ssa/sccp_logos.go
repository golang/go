// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/base"

// logosSCCPPossibleConst admits only pure scalar operations. Constants become
// visible after the final generic rewrite pass when SCCP removes infeasible
// edges, so these operations must participate in its def-use worklist too.
func logosSCCPPossibleConst(v *Value) bool {
	if base.Debug.LogosOpt <= 0 {
		return false
	}
	switch v.Op {
	case OpBitLen8, OpBitLen16, OpBitLen32, OpBitLen64,
		OpPopCount8, OpPopCount16, OpPopCount32, OpPopCount64,
		OpBswap16, OpBswap32, OpBswap64,
		OpAndB, OpOrB:
		return true
	case OpCondSelect:
		return v.Type.IsInteger() || v.Type.IsBoolean()
	default:
		return false
	}
}

// logosSCCPLattice extends the existing lattice without changing the function.
// Every transfer is monotone: operands can move from top to a constant and
// then to bottom, including when a newly reachable loop edge changes a phi.
func (t *worklist) logosSCCPLattice(v *Value) lattice {
	switch v.Op {
	case OpBitLen8, OpBitLen16, OpBitLen32, OpBitLen64,
		OpPopCount8, OpPopCount16, OpPopCount32, OpPopCount64:
		x := t.getLatticeCell(v.Args[0])
		if x.tag == constant {
			// Reuse target-width-aware folding instead of defining a second
			// set of integer and intrinsic evaluation rules.
			return computeLattice(t.f, v, x.val)
		}
		return lattice{x.tag, nil}
	case OpBswap16, OpBswap32, OpBswap64:
		x := t.getLatticeCell(v.Args[0])
		if x.tag != constant {
			return lattice{x.tag, nil}
		}
		value, _ := logosBswap(v.Op, x.val.AuxInt)
		// Byte swaps have no generic constant rewrite. Keep the original
		// value intact until SCCP converges, and retain its result type
		// rather than assuming that it has the opcode's width.
		var c *Value
		switch v.Type.Size() {
		case 2:
			c = t.f.ConstInt16(v.Type, int16(value))
		case 4:
			c = t.f.ConstInt32(v.Type, int32(value))
		case 8:
			c = t.f.ConstInt64(v.Type, value)
		default:
			v.Fatalf("unexpected result type for %s: %v", v.Op, v.Type)
		}
		return lattice{constant, c}
	case OpCondSelect:
		condition := t.getLatticeCell(v.Args[2])
		if condition.tag == top {
			// Waiting preserves monotonicity even if an arm changes before
			// the condition is visited. The condition's uses reschedule us.
			return lattice{top, nil}
		}
		if condition.tag == constant {
			if condition.val.AuxInt != 0 {
				return t.getLatticeCell(v.Args[0])
			}
			return t.getLatticeCell(v.Args[1])
		}
		// An unknown condition can select either arm. Meet their lattices
		// with the same optimistic rules as a phi over executable edges.
		x := t.getLatticeCell(v.Args[0])
		y := t.getLatticeCell(v.Args[1])
		if x.tag == top {
			return y
		}
		if y.tag == top || equals(x, y) {
			return x
		}
		return lattice{bottom, nil}
	case OpAndB, OpOrB:
		x := t.getLatticeCell(v.Args[0])
		y := t.getLatticeCell(v.Args[1])
		if x.tag == top || y.tag == top {
			// Check top before controlling constants: returning a constant
			// here could otherwise move back to top when a loop phi changes.
			return lattice{top, nil}
		}
		identity := int64(1)
		if v.Op == OpOrB {
			identity = 0
		}
		if x.tag == constant {
			if x.val.AuxInt != identity {
				return x
			}
			return y
		}
		if y.tag == constant {
			if y.val.AuxInt != identity {
				return y
			}
			return x
		}
	}
	return lattice{bottom, nil}
}
