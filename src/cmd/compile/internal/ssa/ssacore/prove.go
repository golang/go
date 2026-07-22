// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"
	"math"
	"math/bits"

	"cmd/compile/internal/ssa/ssaop"
)

// FitsInBitsU reports whether x fits in b bits (unsigned).
func FitsInBitsU(x uint64, b uint) bool {
	return x>>b == 0
}

// InitLimit sets initial constant limit for v.  This limit is based
// only on the operation itself, not any of its input arguments. This
// method is only used in two places, once when the prove pass startup
// and the other when a new ssa value is created, both for init. (unlike
// flowLimit, below, which computes additional constraints based on
// ranges of opcode arguments).
func InitLimit(v *Value) Limit {
	if v.Type.IsBoolean() {
		switch v.Op {
		case ssaop.OpConstBool:
			b := v.AuxInt
			return Limit{Min: b, Max: b, Umin: uint64(b), Umax: uint64(b)}
		default:
			return Limit{Min: 0, Max: 1, Umin: 0, Umax: 1}
		}
	}
	if v.Type.IsPtrShaped() { // These are the types that EqPtr/NeqPtr operate on, except uintptr.
		switch v.Op {
		case ssaop.OpConstNil:
			return Limit{Min: 0, Max: 0, Umin: 0, Umax: 0}
		case ssaop.OpAddr, ssaop.OpLocalAddr: // TODO: others?
			l := NoLimit()
			l.Umin = 1
			return l
		default:
			return NoLimit()
		}
	}
	if !v.Type.IsInteger() {
		return NoLimit()
	}

	// Default limits based on type.
	lim := NoLimitForBitsize(uint(v.Type.Size()) * 8)

	// Tighter limits on some opcodes.
	switch v.Op {
	// constants
	case ssaop.OpConst64:
		lim = Limit{Min: v.AuxInt, Max: v.AuxInt, Umin: uint64(v.AuxInt), Umax: uint64(v.AuxInt)}
	case ssaop.OpConst32:
		lim = Limit{Min: v.AuxInt, Max: v.AuxInt, Umin: uint64(uint32(v.AuxInt)), Umax: uint64(uint32(v.AuxInt))}
	case ssaop.OpConst16:
		lim = Limit{Min: v.AuxInt, Max: v.AuxInt, Umin: uint64(uint16(v.AuxInt)), Umax: uint64(uint16(v.AuxInt))}
	case ssaop.OpConst8:
		lim = Limit{Min: v.AuxInt, Max: v.AuxInt, Umin: uint64(uint8(v.AuxInt)), Umax: uint64(uint8(v.AuxInt))}

	// extensions
	case ssaop.OpZeroExt8to64, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to16:
		lim = lim.SignedMinMax(0, 1<<8-1)
		lim = lim.UnsignedMax(1<<8 - 1)
	case ssaop.OpZeroExt16to64, ssaop.OpZeroExt16to32:
		lim = lim.SignedMinMax(0, 1<<16-1)
		lim = lim.UnsignedMax(1<<16 - 1)
	case ssaop.OpZeroExt32to64:
		lim = lim.SignedMinMax(0, 1<<32-1)
		lim = lim.UnsignedMax(1<<32 - 1)
	case ssaop.OpSignExt8to64, ssaop.OpSignExt8to32, ssaop.OpSignExt8to16:
		lim = lim.SignedMinMax(math.MinInt8, math.MaxInt8)
	case ssaop.OpSignExt16to64, ssaop.OpSignExt16to32:
		lim = lim.SignedMinMax(math.MinInt16, math.MaxInt16)
	case ssaop.OpSignExt32to64:
		lim = lim.SignedMinMax(math.MinInt32, math.MaxInt32)

	// math/bits intrinsics
	case ssaop.OpCtz64, ssaop.OpBitLen64, ssaop.OpPopCount64,
		ssaop.OpCtz32, ssaop.OpBitLen32, ssaop.OpPopCount32,
		ssaop.OpCtz16, ssaop.OpBitLen16, ssaop.OpPopCount16,
		ssaop.OpCtz8, ssaop.OpBitLen8, ssaop.OpPopCount8:
		lim = lim.UnsignedMax(uint64(v.Args[0].Type.Size() * 8))

	// bool to uint8 conversion
	case ssaop.OpCvtBoolToUint8:
		lim = lim.UnsignedMax(1)

	// length operations
	case ssaop.OpSliceLen, ssaop.OpSliceCap:
		f := v.Block.Func
		elemSize := uint64(v.Args[0].Type.Elem().Size())
		if elemSize > 0 {
			heapSize := uint64(1)<<(uint64(f.Config.PtrSize)*8) - 1
			maximumElementsFittingInHeap := heapSize / elemSize
			lim = lim.UnsignedMax(maximumElementsFittingInHeap)
		}
		fallthrough
	case ssaop.OpStringLen:
		lim = lim.signedMin(0)
	}

	// signed <-> unsigned propagation
	if lim.Min >= 0 {
		lim = lim.UnsignedMinMax(uint64(lim.Min), uint64(lim.Max))
	}
	if FitsInBitsU(lim.Umax, uint(8*v.Type.Size()-1)) {
		lim = lim.SignedMinMax(int64(lim.Umin), int64(lim.Umax))
	}

	return lim
}

// a Limit records known upper and lower bounds for a value.
//
// If we have min>max or umin>umax, then this Limit is
// called "unsatisfiable". When we encounter such a Limit, we
// know that any code for which that Limit applies is unreachable.
// We don't particularly care how unsatisfiable limits propagate,
// including becoming satisfiable, because any optimization
// decisions based on those limits only apply to unreachable code.
type Limit struct {
	Min, Max   int64  // min <= value <= max, signed
	Umin, Umax uint64 // umin <= value <= umax, unsigned
	// For booleans, we use 0==false, 1==true for both ranges
	// For pointers, we use 0,0,0,0 for nil and minInt64,maxInt64,1,maxUint64 for nonnil
}

func NoLimit() Limit {
	return NoLimitForBitsize(64)
}

// If x and y can add without overflow or underflow
// (using b bits), SafeAdd returns x+y, true.
// Otherwise, returns 0, false.
func SafeAdd(x, y int64, b uint) (int64, bool) {
	s := x + y
	if x >= 0 && y >= 0 && s < 0 {
		return 0, false // 64-bit overflow
	}
	if x < 0 && y < 0 && s >= 0 {
		return 0, false // 64-bit underflow
	}
	if !fitsInBits(s, b) {
		return 0, false
	}
	return s, true
}

// same as safeAdd but for subtraction.
func SafeSub(x, y int64, b uint) (int64, bool) {
	if y == math.MinInt64 {
		if x == math.MaxInt64 {
			return 0, false // 64-bit overflow
		}
		x++
		y++
	}
	return SafeAdd(x, -y, b)
}

// same as safeAddU but for subtraction.
func SafeSubU(x, y uint64, b uint) (uint64, bool) {
	if x < y {
		return 0, false // 64-bit underflow
	}
	s := x - y
	if !FitsInBitsU(s, b) {
		return 0, false
	}
	return s, true
}

func ConvertIntWithBitsize[Target uint64 | int64, Source uint64 | int64](x Source, bitsize uint) Target {
	if Target(0)-1 < 0 {
		// Signed target: sign-extend the low bitsize bits.
		switch bitsize {
		case 64:
			return Target(int64(x))
		case 32:
			return Target(int32(x))
		case 16:
			return Target(int16(x))
		case 8:
			return Target(int8(x))
		}
	} else {
		// Unsigned target: zero-extend the low bitsize bits.
		switch bitsize {
		case 64:
			return Target(uint64(x))
		case 32:
			return Target(uint32(x))
		case 16:
			return Target(uint16(x))
		case 8:
			return Target(uint8(x))
		}
	}
	panic("unreachable")
}

// fitsInBits reports whether x fits in b bits (signed).
func fitsInBits(x int64, b uint) bool {
	if b == 64 {
		return true
	}
	m := int64(-1) << (b - 1)
	M := -m - 1
	return x >= m && x <= M
}

func NoLimitForBitsize(bitsize uint) Limit {
	return Limit{Min: -(1 << (bitsize - 1)), Max: 1<<(bitsize-1) - 1, Umin: 0, Umax: 1<<bitsize - 1}
}

// same as safeAdd for unsigned arithmetic.
func safeAddU(x, y uint64, b uint) (uint64, bool) {
	s := x + y
	if s < x || s < y {
		return 0, false // 64-bit overflow
	}
	if !FitsInBitsU(s, b) {
		return 0, false
	}
	return s, true
}

func (l Limit) String() string {
	return fmt.Sprintf("sm,SM=%d,%d um,UM=%d,%d", l.Min, l.Max, l.Umin, l.Umax)
}

func (l Limit) Intersect(l2 Limit) Limit {
	l.Min = max(l.Min, l2.Min)
	l.Umin = max(l.Umin, l2.Umin)
	l.Max = min(l.Max, l2.Max)
	l.Umax = min(l.Umax, l2.Umax)
	return l
}

func (l Limit) signedMin(m int64) Limit {
	l.Min = max(l.Min, m)
	return l
}

func (l Limit) SignedMinMax(minimum, maximum int64) Limit {
	l.Min = max(l.Min, minimum)
	l.Max = min(l.Max, maximum)
	return l
}

func (l Limit) UnsignedMin(m uint64) Limit {
	l.Umin = max(l.Umin, m)
	return l
}

func (l Limit) UnsignedMax(m uint64) Limit {
	l.Umax = min(l.Umax, m)
	return l
}

func (l Limit) UnsignedMinMax(minimum, maximum uint64) Limit {
	l.Umin = max(l.Umin, minimum)
	l.Umax = min(l.Umax, maximum)
	return l
}

func (l Limit) nonzero() bool {
	return l.Min > 0 || l.Umin > 0 || l.Max < 0
}

func (l Limit) MaybeZero() bool {
	return !l.nonzero()
}

func (l Limit) Nonnegative() bool {
	return l.Min >= 0
}

func (l Limit) Unsat() bool {
	return l.Min > l.Max || l.Umin > l.Umax
}

// UnsignedFixedLeadingBits extracts the all the most significant fixed bits from the limit.
// fixed and count are an other way to represent a limit, you can convert them to a limit as follows:
//
//	umin = fixed
//	umax = fixed | (1<<(64-count) - 1)
//
// In order to be useful for bitmanip analysis fixed and count are a coarser tool than a limit:
// 1. the varying section (umax-umin) is always one less than a power of two
// 2. that section is naturally aligned inside the 64-bit space
func (l Limit) UnsignedFixedLeadingBits() (fixed uint64, count uint) {
	varying := uint(bits.Len64(l.Umin ^ l.Umax))
	count = uint(bits.LeadingZeros64(l.Umin ^ l.Umax))
	fixed = l.Umin &^ (1<<varying - 1)
	return
}

// Add returns the limit obtained by adding a value with limit l
// to a value with limit l2. The result must fit in b bits.
func (l Limit) Add(l2 Limit, b uint) Limit {
	var isLConst, isL2Const bool
	var lConst, l2Const uint64
	if l.Min == l.Max {
		isLConst = true
		lConst = ConvertIntWithBitsize[uint64](l.Min, b)
	} else if l.Umin == l.Umax {
		isLConst = true
		lConst = l.Umin
	}
	if l2.Min == l2.Max {
		isL2Const = true
		l2Const = ConvertIntWithBitsize[uint64](l2.Min, b)
	} else if l2.Umin == l2.Umax {
		isL2Const = true
		l2Const = l2.Umin
	}
	if isLConst && isL2Const {
		r := lConst + l2Const
		r &= (uint64(1) << b) - 1
		int64r := ConvertIntWithBitsize[int64](r, b)
		return Limit{Min: int64r, Max: int64r, Umin: r, Umax: r}
	}

	r := NoLimit()
	min, minOk := SafeAdd(l.Min, l2.Min, b)
	max, maxOk := SafeAdd(l.Max, l2.Max, b)
	if minOk && maxOk {
		r.Min = min
		r.Max = max
	}
	umin, uminOk := safeAddU(l.Umin, l2.Umin, b)
	umax, umaxOk := safeAddU(l.Umax, l2.Umax, b)
	if uminOk && umaxOk {
		r.Umin = umin
		r.Umax = umax
	}
	return r
}

// same as add but for subtraction.
func (l Limit) Sub(l2 Limit, b uint) Limit {
	r := NoLimit()
	min, minOk := SafeSub(l.Min, l2.Max, b)
	max, maxOk := SafeSub(l.Max, l2.Min, b)
	if minOk && maxOk {
		r.Min = min
		r.Max = max
	}
	umin, uminOk := SafeSubU(l.Umin, l2.Umax, b)
	umax, umaxOk := SafeSubU(l.Umax, l2.Umin, b)
	if uminOk && umaxOk {
		r.Umin = umin
		r.Umax = umax
	}
	return r
}

// same as add but for multiplication.
func (l Limit) Mul(l2 Limit, b uint) Limit {
	r := NoLimit()
	umaxhi, umaxlo := bits.Mul64(l.Umax, l2.Umax)
	if umaxhi == 0 && FitsInBitsU(umaxlo, b) {
		r.Umax = umaxlo
		r.Umin = l.Umin * l2.Umin
		// Note: if the code containing this multiply is
		// unreachable, then we may have umin>umax, and this
		// multiply may overflow.  But that's ok for
		// unreachable code. If this code is reachable, we
		// know umin<=umax, so this multiply will not overflow
		// because the max multiply didn't.
	}
	// Signed is harder, so don't bother. The only useful
	// case is when we know both multiplicands are nonnegative,
	// but that case is handled above because we would have then
	// previously propagated signed info to the unsigned domain,
	// and will propagate it back after the multiply.
	return r
}

// Similar to add, but compute 1 << l if it fits without overflow in b bits.
func (l Limit) Exp2(b uint) Limit {
	r := NoLimit()
	if l.Umax < uint64(b) {
		r.Umin = 1 << l.Umin
		r.Umax = 1 << l.Umax
		// Same as above in mul, signed<->unsigned propagation
		// will handle the signed case for us.
	}
	return r
}

// Similar to add, but computes the complement of the limit for bitsize b.
func (l Limit) Com(b uint) Limit {
	switch b {
	case 64:
		return Limit{
			Min:  ^l.Max,
			Max:  ^l.Min,
			Umin: ^l.Umax,
			Umax: ^l.Umin,
		}
	case 32:
		return Limit{
			Min:  int64(^int32(l.Max)),
			Max:  int64(^int32(l.Min)),
			Umin: uint64(^uint32(l.Umax)),
			Umax: uint64(^uint32(l.Umin)),
		}
	case 16:
		return Limit{
			Min:  int64(^int16(l.Max)),
			Max:  int64(^int16(l.Min)),
			Umin: uint64(^uint16(l.Umax)),
			Umax: uint64(^uint16(l.Umin)),
		}
	case 8:
		return Limit{
			Min:  int64(^int8(l.Max)),
			Max:  int64(^int8(l.Min)),
			Umin: uint64(^uint8(l.Umax)),
			Umax: uint64(^uint8(l.Umin)),
		}
	default:
		panic("unreachable")
	}
}

// Similar to add, but computes the negation of the limit for bitsize b.
func (l Limit) Neg(b uint) Limit {
	return l.Com(b).Add(Limit{Min: 1, Max: 1, Umin: 1, Umax: 1}, b)
}

// Similar to add, but computes the TrailingZeros of the limit for bitsize b.
func (l Limit) Ctz(b uint) Limit {
	fixed, fixedCount := l.UnsignedFixedLeadingBits()
	if fixedCount == 64 {
		constResult := min(uint(bits.TrailingZeros64(fixed)), b)
		return Limit{Min: int64(constResult), Max: int64(constResult), Umin: uint64(constResult), Umax: uint64(constResult)}
	}

	varying := 64 - fixedCount
	if l.Umin&((1<<varying)-1) != 0 {
		// there will always be at least one non-zero bit in the varying part
		varying--
		return NoLimit().UnsignedMax(uint64(varying))
	}
	return NoLimit().UnsignedMax(uint64(min(uint(bits.TrailingZeros64(fixed)), b)))
}

// Similar to add, but computes the Len of the limit for bitsize b.
func (l Limit) Bitlen(b uint) Limit {
	return NoLimit().UnsignedMinMax(
		uint64(bits.Len64(l.Umin)),
		uint64(bits.Len64(l.Umax)),
	)
}

// Similar to add, but computes the PopCount of the limit for bitsize b.
func (l Limit) Popcount(b uint) Limit {
	fixed, fixedCount := l.UnsignedFixedLeadingBits()
	varying := 64 - fixedCount
	fixedContribution := uint64(bits.OnesCount64(fixed))

	min := fixedContribution
	max := fixedContribution + uint64(varying)

	varyingMask := uint64(1)<<varying - 1

	if varyingPartOfUmax := l.Umax & varyingMask; uint(bits.OnesCount64(varyingPartOfUmax)) != varying {
		// there is at least one zero bit in the varying part
		max--
	}
	if varyingPartOfUmin := l.Umin & varyingMask; varyingPartOfUmin != 0 {
		// there is at least one non-zero bit in the varying part
		min++
	}

	return NoLimit().UnsignedMinMax(min, max)
}

func (l Limit) ConstValue() (_ int64, ok bool) {
	switch {
	case l.Min == l.Max:
		return l.Min, true
	case l.Umin == l.Umax:
		return int64(l.Umin), true
	default:
		return 0, false
	}
}
