// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "math/bits"

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
func arm64Invert(op Op) Op {
	switch op {
	case OpARM64LessThan:
		return OpARM64GreaterThan
	case OpARM64LessThanU:
		return OpARM64GreaterThanU
	case OpARM64GreaterThan:
		return OpARM64LessThan
	case OpARM64GreaterThanU:
		return OpARM64LessThanU
	case OpARM64LessEqual:
		return OpARM64GreaterEqual
	case OpARM64LessEqualU:
		return OpARM64GreaterEqualU
	case OpARM64GreaterEqual:
		return OpARM64LessEqual
	case OpARM64GreaterEqualU:
		return OpARM64LessEqualU
	case OpARM64Equal, OpARM64NotEqual:
		return op
	case OpARM64LessThanF:
		return OpARM64GreaterThanF
	case OpARM64GreaterThanF:
		return OpARM64LessThanF
	case OpARM64LessEqualF:
		return OpARM64GreaterEqualF
	case OpARM64GreaterEqualF:
		return OpARM64LessEqualF
	case OpARM64NotLessThanF:
		return OpARM64NotGreaterThanF
	case OpARM64NotGreaterThanF:
		return OpARM64NotLessThanF
	case OpARM64NotLessEqualF:
		return OpARM64NotGreaterEqualF
	case OpARM64NotGreaterEqualF:
		return OpARM64NotLessEqualF
	default:
		panic("unreachable")
	}
}

// arm64Negate finds the complement to an ARM64 condition code,
// for example !Equal -> NotEqual or !LessThan -> GreaterEqual
//
// For floating point, it's more subtle because NaN is unordered. We do
// !LessThanF -> NotLessThanF, the latter takes care of NaNs.
func arm64Negate(op Op) Op {
	switch op {
	case OpARM64LessThan:
		return OpARM64GreaterEqual
	case OpARM64LessThanU:
		return OpARM64GreaterEqualU
	case OpARM64GreaterThan:
		return OpARM64LessEqual
	case OpARM64GreaterThanU:
		return OpARM64LessEqualU
	case OpARM64LessEqual:
		return OpARM64GreaterThan
	case OpARM64LessEqualU:
		return OpARM64GreaterThanU
	case OpARM64GreaterEqual:
		return OpARM64LessThan
	case OpARM64GreaterEqualU:
		return OpARM64LessThanU
	case OpARM64Equal:
		return OpARM64NotEqual
	case OpARM64NotEqual:
		return OpARM64Equal
	case OpARM64LessThanF:
		return OpARM64NotLessThanF
	case OpARM64NotLessThanF:
		return OpARM64LessThanF
	case OpARM64LessEqualF:
		return OpARM64NotLessEqualF
	case OpARM64NotLessEqualF:
		return OpARM64LessEqualF
	case OpARM64GreaterThanF:
		return OpARM64NotGreaterThanF
	case OpARM64NotGreaterThanF:
		return OpARM64GreaterThanF
	case OpARM64GreaterEqualF:
		return OpARM64NotGreaterEqualF
	case OpARM64NotGreaterEqualF:
		return OpARM64GreaterEqualF
	default:
		panic("unreachable")
	}
}

// evaluate an ARM64 op against a flags value
// that is potentially constant; return 1 for true,
// -1 for false, and 0 for not constant.
func ccARM64Eval(op Op, flags *Value) int {
	fop := flags.Op
	if fop == OpARM64InvertFlags {
		return -ccARM64Eval(op, flags.Args[0])
	}
	if fop != OpARM64FlagConstant {
		return 0
	}
	fc := FlagConstant(flags.AuxInt)
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return -1
	}
	switch op {
	case OpARM64Equal:
		return b2i(fc.Eq())
	case OpARM64NotEqual:
		return b2i(fc.Ne())
	case OpARM64LessThan:
		return b2i(fc.Lt())
	case OpARM64LessThanU:
		return b2i(fc.Ult())
	case OpARM64GreaterThan:
		return b2i(fc.Gt())
	case OpARM64GreaterThanU:
		return b2i(fc.Ugt())
	case OpARM64LessEqual:
		return b2i(fc.Le())
	case OpARM64LessEqualU:
		return b2i(fc.Ule())
	case OpARM64GreaterEqual:
		return b2i(fc.Ge())
	case OpARM64GreaterEqualU:
		return b2i(fc.Uge())
	}
	return 0
}

// checks if mask >> rshift applied at lsb is a valid arm64 bitfield op mask.
func isARM64BFMask(lsb, mask, rshift int64) bool {
	shiftedMask := int64(uint64(mask) >> uint64(rshift))
	return shiftedMask != 0 && IsPowerOfTwo(shiftedMask+1) && nto(shiftedMask)+lsb < 64
}

// nto returns the number of trailing ones.
func nto(x int64) int64 {
	return int64(Ntz64(^x))
}

func rotateRight64(v, rotate int64) int64 {
	return int64(bits.RotateLeft64(uint64(v), int(-rotate)))
}
