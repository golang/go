// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"math/bits"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
)

// returns the bitfield width of mask >> rshift for arm64 bitfield ops.
func arm64BFWidth(mask, rshift int64) int64 {
	shiftedMask := int64(uint64(mask) >> uint64(rshift))
	if shiftedMask == 0 {
		panic("ARM64 BF mask is zero")
	}
	return nto(shiftedMask)
}

// arm64Invert evaluates (InvertFlags op), which
// is the same as altering the condition codes such
// that the same result would be produced if the arguments
// to the flag-generating instruction were reversed, e.g.
// (InvertFlags (CMP x y)) -> (CMP y x)
func arm64Invert(op ssaop.Op) ssaop.Op {
	switch op {
	case ssaop.OpARM64LessThan:
		return ssaop.OpARM64GreaterThan
	case ssaop.OpARM64LessThanU:
		return ssaop.OpARM64GreaterThanU
	case ssaop.OpARM64GreaterThan:
		return ssaop.OpARM64LessThan
	case ssaop.OpARM64GreaterThanU:
		return ssaop.OpARM64LessThanU
	case ssaop.OpARM64LessEqual:
		return ssaop.OpARM64GreaterEqual
	case ssaop.OpARM64LessEqualU:
		return ssaop.OpARM64GreaterEqualU
	case ssaop.OpARM64GreaterEqual:
		return ssaop.OpARM64LessEqual
	case ssaop.OpARM64GreaterEqualU:
		return ssaop.OpARM64LessEqualU
	case ssaop.OpARM64Equal, ssaop.OpARM64NotEqual:
		return op
	case ssaop.OpARM64LessThanF:
		return ssaop.OpARM64GreaterThanF
	case ssaop.OpARM64GreaterThanF:
		return ssaop.OpARM64LessThanF
	case ssaop.OpARM64LessEqualF:
		return ssaop.OpARM64GreaterEqualF
	case ssaop.OpARM64GreaterEqualF:
		return ssaop.OpARM64LessEqualF
	case ssaop.OpARM64NotLessThanF:
		return ssaop.OpARM64NotGreaterThanF
	case ssaop.OpARM64NotGreaterThanF:
		return ssaop.OpARM64NotLessThanF
	case ssaop.OpARM64NotLessEqualF:
		return ssaop.OpARM64NotGreaterEqualF
	case ssaop.OpARM64NotGreaterEqualF:
		return ssaop.OpARM64NotLessEqualF
	default:
		panic("unreachable")
	}
}

// arm64Negate finds the complement to an ARM64 condition code,
// for example !Equal -> NotEqual or !LessThan -> GreaterEqual
//
// For floating point, it's more subtle because NaN is unordered. We do
// !LessThanF -> NotLessThanF, the latter takes care of NaNs.
func arm64Negate(op ssaop.Op) ssaop.Op {
	switch op {
	case ssaop.OpARM64LessThan:
		return ssaop.OpARM64GreaterEqual
	case ssaop.OpARM64LessThanU:
		return ssaop.OpARM64GreaterEqualU
	case ssaop.OpARM64GreaterThan:
		return ssaop.OpARM64LessEqual
	case ssaop.OpARM64GreaterThanU:
		return ssaop.OpARM64LessEqualU
	case ssaop.OpARM64LessEqual:
		return ssaop.OpARM64GreaterThan
	case ssaop.OpARM64LessEqualU:
		return ssaop.OpARM64GreaterThanU
	case ssaop.OpARM64GreaterEqual:
		return ssaop.OpARM64LessThan
	case ssaop.OpARM64GreaterEqualU:
		return ssaop.OpARM64LessThanU
	case ssaop.OpARM64Equal:
		return ssaop.OpARM64NotEqual
	case ssaop.OpARM64NotEqual:
		return ssaop.OpARM64Equal
	case ssaop.OpARM64LessThanF:
		return ssaop.OpARM64NotLessThanF
	case ssaop.OpARM64NotLessThanF:
		return ssaop.OpARM64LessThanF
	case ssaop.OpARM64LessEqualF:
		return ssaop.OpARM64NotLessEqualF
	case ssaop.OpARM64NotLessEqualF:
		return ssaop.OpARM64LessEqualF
	case ssaop.OpARM64GreaterThanF:
		return ssaop.OpARM64NotGreaterThanF
	case ssaop.OpARM64NotGreaterThanF:
		return ssaop.OpARM64GreaterThanF
	case ssaop.OpARM64GreaterEqualF:
		return ssaop.OpARM64NotGreaterEqualF
	case ssaop.OpARM64NotGreaterEqualF:
		return ssaop.OpARM64GreaterEqualF
	default:
		panic("unreachable")
	}
}

// evaluate an ARM64 op against a flags value
// that is potentially constant; return 1 for true,
// -1 for false, and 0 for not constant.
func ccARM64Eval(op ssaop.Op, flags *ssa.Value) int {
	fop := flags.Op
	if fop == ssaop.OpARM64InvertFlags {
		return -ccARM64Eval(op, flags.Args[0])
	}
	if fop != ssaop.OpARM64FlagConstant {
		return 0
	}
	fc := ssa.FlagConstant(flags.AuxInt)
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return -1
	}
	switch op {
	case ssaop.OpARM64Equal:
		return b2i(fc.Eq())
	case ssaop.OpARM64NotEqual:
		return b2i(fc.Ne())
	case ssaop.OpARM64LessThan:
		return b2i(fc.Lt())
	case ssaop.OpARM64LessThanU:
		return b2i(fc.Ult())
	case ssaop.OpARM64GreaterThan:
		return b2i(fc.Gt())
	case ssaop.OpARM64GreaterThanU:
		return b2i(fc.Ugt())
	case ssaop.OpARM64LessEqual:
		return b2i(fc.Le())
	case ssaop.OpARM64LessEqualU:
		return b2i(fc.Ule())
	case ssaop.OpARM64GreaterEqual:
		return b2i(fc.Ge())
	case ssaop.OpARM64GreaterEqualU:
		return b2i(fc.Uge())
	}
	return 0
}

// checks if mask >> rshift applied at lsb is a valid arm64 bitfield op mask.
func isARM64BFMask(lsb, mask, rshift int64) bool {
	shiftedMask := int64(uint64(mask) >> uint64(rshift))
	return shiftedMask != 0 && ssa.IsPowerOfTwo(shiftedMask+1) && nto(shiftedMask)+lsb < 64
}

// nto returns the number of trailing ones.
func nto(x int64) int64 {
	return int64(ssa.Ntz64(^x))
}

func rotateRight64(v, rotate int64) int64 {
	return int64(bits.RotateLeft64(uint64(v), int(-rotate)))
}
