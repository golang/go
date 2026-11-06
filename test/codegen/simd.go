// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of simd peephole optimizations.

//go:build goexperiment.simd && amd64

package codegen

import (
	"math"
	"simd/archsimd"
)

func vptest1() bool {
	v1 := archsimd.LoadUint64x2([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2([]uint64{0, 0})
	// amd64:`VPTEST (.*)(.*)$`
	// amd64:`SETCS (.*)$`
	return v1.AndNot(v2).IsZero()
}

func vptest2() bool {
	v1 := archsimd.LoadUint64x2([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2([]uint64{0, 0})
	// amd64:`VPTEST (.*)(.*)$`
	// amd64:`SETEQ (.*)$`
	return v1.And(v2).IsZero()
}

type Args2 struct {
	V0 archsimd.Uint8x32
	V1 archsimd.Uint8x32
	x  string
}

//go:noinline
func simdStructNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	return a.V0.Xor(a.V1)
}

func simdStructWrapperNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	a.x = "test"
	return simdStructNoSpill(a)
}

//go:noinline
func simdArrayNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	return a[0].V0.Xor(a[0].V1)
}

func simdArrayWrapperNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	a[0].x = "test"
	return simdArrayNoSpill(a)
}

func simdFeatureGuardedMaskOpt() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Masked(mask) // amd64:`VPADDW.Z .*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Masked(mask) // amd64:`VPAND .*$`
}

func simdMaskedMerge() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Merge(x, mask) // amd64:-`VPBLENDVB .*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Merge(x, mask) // amd64:`VPBLENDVB .*$`
}

func simdIfElseVPBLENDM(x, y archsimd.Int32x4, bits uint8) archsimd.Int32x4 {
	if archsimd.X86.AVX512() {
		// Check that 128/256-bit IfElse with an AVX-512 K-mask blends directly with VPBLENDM.
		// amd64:`KMOVB` `VPBLENDMD X[0-9]+, X[0-9]+, K[1-7], X[0-9]+` -`VPMOVM2D` -`VPBLENDVB`
		return x.IfElse(archsimd.Mask32x4FromBits(bits), y)
	}
	// amd64:`VPBLENDVB`
	return x.IfElse(archsimd.Mask32x4FromBits(bits), y)
}

func simdFloat32x8Masked(x, y archsimd.Float32x8) archsimd.Float32x8 {
	if archsimd.X86.AVX512() {
		// Check that Masked folds into a zeroing AVX-512 op.
		mask := x.Less(y)
		// amd64:`VADDPS.Z .*$` -`VANDPS` -`VPAND`
		return x.Add(y).Masked(mask)
	}
	// Check that we produce the AVX1 floating-point AND, not the AVX2 integer AND.
	mask := x.Less(y)
	// amd64:`VANDPS .*$` -`VPAND`
	return x.Add(y).Masked(mask)
}

func simdFloat64x4Masked(x, y archsimd.Float64x4) archsimd.Float64x4 {
	if archsimd.X86.AVX512() {
		// Check that Masked folds into a zeroing AVX-512 op.
		mask := x.Less(y)
		// amd64:`VADDPD.Z .*$` -`VANDPD` -`VPAND`
		return x.Add(y).Masked(mask)
	}
	// Check that we produce the AVX1 floating-point AND, not the AVX2 integer AND.
	mask := x.Less(y)
	// amd64:`VANDPD .*$` -`VPAND`
	return x.Add(y).Masked(mask)
}

func simdFloat32x8IfElse(x, y archsimd.Float32x8) archsimd.Float32x8 {
	if archsimd.X86.AVX512() {
		// Check that a comparison mask feeds directly into a merging AVX-512 op.
		mask := x.Less(y)
		// amd64:`VADDPS .*, K[1-7], .*$` -`VBLENDVPS` -`VPBLEND`
		return x.Add(y).IfElse(mask, x)
	}
	// Check that we produce the AVX1 floating-point blend, not the AVX2 integer blend.
	mask := x.Less(y)
	// amd64:`VBLENDVPS .*$` -`VPBLENDVB`
	return x.Add(y).IfElse(mask, x)
}

func simdFloat64x4IfElse(x, y archsimd.Float64x4) archsimd.Float64x4 {
	if archsimd.X86.AVX512() {
		// Check that a comparison mask feeds directly into a merging AVX-512 op.
		mask := x.Less(y)
		// amd64:`VADDPD .*, K[1-7], .*$` -`VBLENDVPD` -`VPBLEND`
		return x.Add(y).IfElse(mask, x)
	}
	// Check that we produce the AVX1 floating-point blend, not the AVX2 integer blend.
	mask := x.Less(y)
	// amd64:`VBLENDVPD .*$` -`VPBLENDVB`
	return x.Add(y).IfElse(mask, x)
}

func simdCompareToKMaskMerge(x, y, dst, a, b archsimd.Int32x8) archsimd.Int32x8 {
	if !archsimd.X86.AVX512() {
		return dst
	}
	// Check that a 256-bit comparison used as a merge mask compares into a K-mask
	// and folds into a merging AVX-512 op.
	// amd64:`VPCMPD \$6, Y[0-9]+, Y[0-9]+, K[1-7]` `VPADDD Y[0-9]+, Y[0-9]+, K[1-7], Y[0-9]+` -`VPCMPGTD` -`VPBLEND`
	return x.Add(y).Merge(dst, a.Less(b))
}

func simdCompareToKMaskAndIfElse(x, y, a, b, c, d archsimd.Float64x4) archsimd.Float64x4 {
	if !archsimd.X86.AVX512() {
		return y
	}
	// Check that an And of two 256-bit comparisons used as an IfElse mask compares
	// into K-masks, combines them with KANDB, and blends with VPBLENDMQ.
	// amd64:`VCMPPD \$[01], Y[0-9]+, Y[0-9]+, K[1-7]` `KANDB` `VPBLENDMQ Y[0-9]+, Y[0-9]+, K[1-7], Y[0-9]+` -`VANDPD` -`VBLENDVPD`
	return x.IfElse(a.Less(b).And(c.Equal(d)), y)
}

func simdAbsInt64x2(x archsimd.Int64x2) archsimd.Uint64x2 {
	if archsimd.X86.AVX512() {
		// Check that the AVX emulation is turned back into VPABSQ when AVX-512 is known.
		// amd64:`VPABSQ X[0-9]+, X[0-9]+` -`VPCMPGTQ`
		return x.Abs()
	}
	// amd64:`VPCMPGTQ` -`VPABSQ`
	return x.Abs()
}

func simdAbsInt64x4(x archsimd.Int64x4) archsimd.Uint64x4 {
	if archsimd.X86.AVX512() {
		// Check that the AVX2 emulation is turned back into VPABSQ when AVX-512 is known.
		// amd64:`VPABSQ Y[0-9]+, Y[0-9]+` -`VPCMPGTQ`
		return x.Abs()
	}
	// amd64:`VPCMPGTQ` -`VPABSQ`
	return x.Abs()
}

var nan = math.NaN()
var floats64s = []float64{0, 1, 2, nan, 4, nan, 6, 7, 8, 9, 10, 11, nan, 13, 14, 15}
var sinkInt64s = make([]int64, 100)

func simdIsNaN() {
	x := archsimd.LoadFloat64x4(floats64s)
	y := archsimd.LoadFloat64x4(floats64s[4:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x4().Store(sinkInt64s)
}

func simdIsNaN512() {
	x := archsimd.LoadFloat64x8(floats64s)
	y := archsimd.LoadFloat64x8(floats64s[8:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x8().Store(sinkInt64s)
}

func sftImmVPSRL() archsimd.Uint32x4 {
	var x archsimd.Uint32x4
	// amd64:`VPSRLD \$1, .*$`
	return x.ShiftAllRight(1)
}

func sftImmZeroMasked(x archsimd.Int16x32, bits uint32) archsimd.Int16x32 {
	// Check that a zero-masked shift by 0 still zeroes masked-off lanes instead of folding to x.
	mask := archsimd.Mask16x32FromBits(bits)
	// amd64:`VMOVDQU16\.Z` -`VPSLLW`
	return x.ShiftAllLeft(0).Masked(mask)
}

func sftImmZeroMerge(x, dst archsimd.Int16x32, bits uint32) archsimd.Int16x32 {
	// Check that a merge-masked shift by 0 folds to a blend instead of shifting by 0.
	mask := archsimd.Mask16x32FromBits(bits)
	// amd64:`VPBLENDMW` -`VPSLLW`
	return x.ShiftAllLeft(0).IfElse(mask, dst)
}

func aLtbLtc8_avx512(a, b, c archsimd.Int8x64) archsimd.Mask8x64 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KANDQ`
	return a.Less(b).And(b.Less(c))
}

func aLtbORbLtc8_avx512(a, b, c archsimd.Int8x64) archsimd.Mask8x64 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KORQ`
	return a.Less(b).Or(b.Less(c))
}

func aLtbLtc64_avx512(a, b, c archsimd.Int64x8) archsimd.Mask64x8 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KANDB`
	return a.Less(b).And(b.Less(c))
}

func aLtbORbLtc64_avx512(a, b, c archsimd.Int64x8) archsimd.Mask64x8 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KORB`
	return a.Less(b).Or(b.Less(c))
}

var globalSlice = []uint32{1, 2, 3, 4, 5, 6, 7, 8}

func simdMemoryOperandMerge() archsimd.Uint32x4 {
	a := archsimd.BroadcastUint32x4(1)
	// amd64:`VPADDD \([A-Z]+\), X\d, X\d`
	a = a.Add(archsimd.LoadUint32x4(globalSlice[0:4]))
	// amd64:`VPADDD 16\([A-Z]+\), X\d, X\d`
	a = a.Add(archsimd.LoadUint32x4(globalSlice[4:8]))
	return a
}

func simdZeroingUsesVEX(x archsimd.Uint64x8) uint64 {
	// The SIMD-typed value in this function implies AVX is present,
	// so the zeroing of t must not use legacy-SSE encodings, which
	// would incur AVX-SSE transition penalties (issue 80835).
	// amd64:`VMOVUPS X15` -`\bMOVUPS`
	var t [8]uint64
	acc := archsimd.LoadUint64x8(t[:])
	acc = acc.Add(x)
	acc.Store(t[:])
	var n uint64
	for _, v := range t {
		n += v
	}
	return n
}

func simdShiftScalarUsesVEX(x archsimd.Uint64x2, n uint64) archsimd.Uint64x2 {
	// The scalar shift count must be moved to a vector register with
	// a VEX encoding, not a legacy-SSE MOVQ (issue 80835).
	// amd64:`VMOVQ` -`\bMOVQ [A-Z]+, X`
	return x.ShiftAllLeft(n)
}

func simdMoveUsesVEX(x archsimd.Uint64x8, p *[8]uint64) uint64 {
	// The 64-byte array copy is a lowered Move; its 16-byte chunks
	// must use VEX encodings here (issue 80835).
	var t [8]uint64
	acc := archsimd.LoadUint64x8(t[:])
	acc = acc.Add(x)
	acc.Store(t[:])
	// amd64:`VMOVUPS \([A-Z0-9]+\), X14` -`\bMOVUPS`
	*p = t
	return p[0]
}

func simdMove16UsesVEX(x archsimd.Uint64x2, d, s *[16]byte) uint64 {
	// The 16-byte block copy is lowered to MOVOload/MOVOstore, which
	// must use VEX encodings here (issue 80835).
	// amd64:`VMOVUPS \([A-Z0-9]+\), X` -`\bMOVUPS`
	*d = *s
	t := x.Add(x)
	var out [2]uint64
	t.Store(out[:])
	return out[0]
}

//go:noinline
func simdOpaque() {}

func simdSpillUsesVEX(a, b archsimd.Uint64x2) archsimd.Uint64x2 {
	// x is live across the call, so it is spilled and reloaded; both
	// must use VEX encodings here (issue 80835).
	// amd64:`VMOVUPS [^,]*\(SP\), X` -`\bMOVUPS`
	x := a.Add(b)
	simdOpaque()
	return x.Add(x)
}

func simdShuffleCopyUsesVEX(a, b archsimd.Uint64x2, n int) archsimd.Uint64x2 {
	// The swap makes regalloc place shuffle copies in the block
	// splitting the loop back edge; they must use VEX encodings here.
	// This also relies on split-edge blocks inheriting the CPU
	// features of their edge (issue 80835).
	x, y := a, b
	// amd64:`VMOVUPS X[0-9]+, X[0-9]+` -`\bMOVUPS`
	for i := 0; i < n; i++ {
		x, y = y, x
	}
	return x.Add(y)
}

func simdMaskSpillUsesK(z archsimd.Float32x8, m archsimd.Mask32x8, n int) archsimd.Float32x8 {
	// The closure keeps the mask itself live across the call, so it is
	// spilled. It must be reloaded straight into a K register, rather
	// than into a general purpose one and moved from there.
	_ = func() {
		z = z.Compress(m)
		simdOpaque()
		// amd64:`KMOVQ [^,]*\(SP\), K[0-7]`
		for range n {
			// amd64:-`KMOVQ [A-Z]+, K[0-7]`
			z = z.Compress(m)
		}
	}
	return z
}

// 128-bit broadcasts need only AVX: without AVX2 they use AVX shuffles, and
// with AVX2 (known from GOAMD64 or a feature check) they use VPBROADCAST.

func broadcast128Int8(x int8) archsimd.Int8x16 {
	// amd64/v1:`VPSHUFB` -`VPBROADCASTB`
	// amd64/v3:`VPBROADCASTB` -`VPSHUFB`
	return archsimd.BroadcastInt8x16(x)
}

func broadcast128Int16(x int16) archsimd.Int16x8 {
	// amd64/v1:`VPUNPCKLWD` `VPSHUFD \$0,` -`VPBROADCASTW`
	// amd64/v3:`VPBROADCASTW` -`VPSHUFD`
	return archsimd.BroadcastInt16x8(x)
}

func broadcast128Int32(x int32) archsimd.Int32x4 {
	// amd64/v1:`VPSHUFD \$0,` -`VPBROADCASTD`
	// amd64/v3:`VPBROADCASTD` -`VPSHUFD`
	return archsimd.BroadcastInt32x4(x)
}

func broadcast128Float32(x float32) archsimd.Float32x4 {
	// amd64/v1:`VPSHUFD \$0,` -`VBROADCASTSS`
	// amd64/v3:`VBROADCASTSS` -`VPSHUFD`
	return archsimd.BroadcastFloat32x4(x)
}

func broadcast128Int64(x int64) archsimd.Int64x2 {
	// amd64/v1:`VPSHUFD \$68,` -`VPBROADCASTQ`
	// amd64/v3:`VPBROADCASTQ` -`VPSHUFD`
	return archsimd.BroadcastInt64x2(x)
}

func broadcast128AVX2Checked(x int32) archsimd.Int32x4 {
	if archsimd.X86.AVX2() {
		// amd64:`VPBROADCASTD` -`VPSHUFD`
		return archsimd.BroadcastInt32x4(x)
	}
	return archsimd.Int32x4{}
}

func broadcast128MulToShift(x archsimd.Int32x4) archsimd.Int32x4 {
	// amd64:`VPSLLD \$3,` -`VPMULLD`
	return x.Mul(archsimd.BroadcastInt32x4(8))
}
