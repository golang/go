// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/compile/internal/base"

// foldLogos extends the existing bit analysis without changing its lattice or
// cycle handling. fold normalizes the returned facts to the result width.
func (kb *knownBitsState) foldLogos(v *Value) (value, known int64) {
	if base.Debug.LogosOpt <= 0 {
		return 0, 0
	}
	switch v.Op {
	case OpBswap16, OpBswap32, OpBswap64:
		return kb.foldLogosBswap(v)
	case OpBitLen8, OpBitLen16, OpBitLen32, OpBitLen64,
		OpCtz8, OpCtz16, OpCtz32, OpCtz64,
		OpCtz8NonZero, OpCtz16NonZero, OpCtz32NonZero, OpCtz64NonZero,
		OpPopCount8, OpPopCount16, OpPopCount32, OpPopCount64:
		return kb.foldLogosBitIntrinsic(v)
	case OpAdd8, OpAdd16, OpAdd32, OpAdd64:
		x, xk := kb.fold(v.Args[0])
		y, yk := kb.fold(v.Args[1])
		return logosKnownBitsAdd(x, xk, y, yk, 0)
	case OpSub8, OpSub16, OpSub32, OpSub64:
		x, xk := kb.fold(v.Args[0])
		y, yk := kb.fold(v.Args[1])
		return logosKnownBitsAdd(x, xk, ^y&yk, yk, 1)
	case OpNeg8, OpNeg16, OpNeg32, OpNeg64:
		x, xk := kb.fold(v.Args[0])
		return logosKnownBitsAdd(0, -1, ^x&xk, xk, 1)
	case OpRotateLeft8, OpRotateLeft16, OpRotateLeft32, OpRotateLeft64:
		x, xk := kb.fold(v.Args[0])
		if xk == 0 {
			return 0, 0
		}
		y, yk := kb.fold(v.Args[1])
		width := uint(v.Type.Size() * 8)
		mask := ^uint64(0) >> (64 - width)
		data, dataKnown := uint64(x)&mask, uint64(xk)&mask

		// Only the low log2(width) count bits matter, regardless of the
		// count's type or sign. At most width (and thus 64) residues remain.
		countMask := int64(width - 1)
		y &= countMask
		yk |= ^countMask
		first := true
		for count := range allPossibleValues(y, yk) {
			shift := uint(count)
			a := int64((data<<shift | data>>(width-shift)) & mask)
			k := int64((dataKnown<<shift | dataKnown>>(width-shift)) & mask)
			if first {
				value, known = a, k
				first = false
			} else {
				known &= k &^ (value ^ a)
			}
			if known == 0 {
				break
			}
		}
		return value & known, known
	default:
		return 0, 0
	}
}

// rewriteLogosRotateProjection removes a rotate followed by a logical right
// shift by the same literal count k. For 0 <= k < width, the pair equals
// x & (widthMask >> k), so it equals x when x's top k bits are known zero.
func (kb *knownBitsState) rewriteLogosRotateProjection(v *Value) {
	if base.Debug.LogosOpt <= 0 {
		return
	}
	var width uint
	var rotate Op
	switch v.Op {
	case OpRsh8Ux8, OpRsh8Ux16, OpRsh8Ux32, OpRsh8Ux64:
		width, rotate = 8, OpRotateLeft8
	case OpRsh16Ux8, OpRsh16Ux16, OpRsh16Ux32, OpRsh16Ux64:
		width, rotate = 16, OpRotateLeft16
	case OpRsh32Ux8, OpRsh32Ux16, OpRsh32Ux32, OpRsh32Ux64:
		width, rotate = 32, OpRotateLeft32
	case OpRsh64Ux8, OpRsh64Ux16, OpRsh64Ux32, OpRsh64Ux64:
		width, rotate = 64, OpRotateLeft64
	default:
		return
	}
	r := v.Args[0]
	if r.Op != rotate {
		return
	}
	shiftCount, rotateCount := v.Args[1], r.Args[1]
	if !shiftCount.isGenericIntConst() || !rotateCount.isGenericIntConst() {
		return
	}
	k := shiftCount.AuxInt
	if k < 0 || k >= int64(width) || k != rotateCount.AuxInt {
		return
	}
	input := r.Args[0]
	value, known := kb.fold(input)
	widthMask := ^uint64(0) >> (64 - width)
	highMask := widthMask &^ (widthMask >> uint(k))
	if uint64(known)&highMask != highMask || uint64(value)&highMask != 0 {
		return
	}
	// The value is equivalent, so its cached facts remain sound.
	v.copyOf(input)
}

// logosKnownBitsAdd computes facts for x + y + carry. Unknown input bits must
// be zero in x and y; carry must be the known bit 0 or 1. Arithmetic wraps,
// including signed overflow, so the low result bits also serve narrower types.
//
// Adding every unknown input bit identifies the possible carry changes. A bit
// unknown in either input or changed by those carries stays unknown. This is
// the tristate addition construction proved sound and optimal in
// "Sound, Precise, and Fast Abstract Interpretation with Tristate Numbers"
// (CGO 2022): https://people.cs.rutgers.edu/~santosh.nagarakatte/papers/cgo-2022.pdf.
func logosKnownBitsAdd(x, xk, y, yk, carry int64) (value, known int64) {
	sum := x + y + carry
	sumWithUnknowns := sum + ^xk + ^yk
	unknown := ^xk | ^yk | (sum ^ sumWithUnknowns)
	return sum &^ unknown, ^unknown
}
