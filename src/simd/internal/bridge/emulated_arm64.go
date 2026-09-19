// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package bridge

import "simd/archsimd"

func (xx Uint64x2nclm) CarrylessMultiplyEven(yy Uint64x2nclm) Uint64x2nclm {
	x := archsimd.Uint64x2(xx)
	y := archsimd.Uint64x2(yy)

	return Uint64x2nclm(carrylessMultiply_(x, y))
}

func (xx Uint64x2nclm) CarrylessMultiplyOdd(yy Uint64x2nclm) Uint64x2nclm {
	x := archsimd.Uint64x2(xx)
	y := archsimd.Uint64x2(yy)

	x = x.SetElem(0, x.GetElem(1))
	y = y.SetElem(0, y.GetElem(1))
	return Uint64x2nclm(carrylessMultiply_(x, y))
}

func new64x2(lo, hi uint64) archsimd.Uint64x2 {
	return archsimd.Uint64x2{}.SetElem(0, lo).SetElem(1, hi)
}

// These masks all have 4 zeroes between 1s.
var m0_ = new64x2(0x1084210842108421, 0x2108421084210842)
var m1_ = new64x2(0x2108421084210842, 0x4210842108421084)
var m2_ = new64x2(0x4210842108421084, 0x8421084210842108)
var m3_ = new64x2(0x8421084210842108, 0x0842108421084210)
var m4_ = new64x2(0x0842108421084210, 0x1084210842108421)

// Selects the middle 64 bits of a 128-bit simd value
var middle = new64x2(0xffffffff00000000, 0x00000000ffffffff)

// mwl_ is a 64x64 into 128 multiply that is missing
// some carries that we don't need for CLMUL emulation.
// The high 64 bits of each input are ignored.
// Also just for fun, accumulate sums with Xor.
func mwl(x, y archsimd.Uint64x2) archsimd.Uint64x2 {
	// reshape input into Uint32x4
	// input is  {a b _ _}.mwl_{c d _ _}
	// need the sum of
	// ac0_ac1
	//   0 ad0_ad1
	//   0 bc0_bc1
	//   0   0 bd0_bd1
	// This "sum" is where the carries (not propagated
	// across lanes) are lost.
	ab__ := x.ReshapeToUint32s()
	cd__ := y.ReshapeToUint32s()
	ac0_ac1_bd0_bd1 := ab__.MulWidenLo(cd__)

	dc__ := y.RotateAllLeft(32).ReshapeToUint32s()
	ad0_ad1_bc0_bc1 := ab__.MulWidenLo(dc__)
	//
	// have        ad0, ad1, bc0, bc1
	// want        0, ad0+bc0, ad1+bc1, 0
	// to add to    ac0_ac1_bd0_bd1
	//
	// swap 64-bit halves of ad0_ad1_bc0_bc1
	// to get   bc0_bc1_ad0_ad1
	bc0_bc1_ad0_ad1 := archsimd.Uint64x2{}.SetElem(0, ad0_ad1_bc0_bc1.GetElem(1)).SetElem(1, ad0_ad1_bc0_bc1.GetElem(0))

	// added to ad0_ad1_bc0_bc1 yields
	//   bc0+ad0, bc1+ad1, bc0+ad0, bc1+ad1
	// rotate 32 (within the two 64-bit elements) yields
	//   bc1+ad1, bc0+ad0, bc1+ad1, bc0+ad0
	// and then intersect with mask:
	//   0      , bc0+ad0, bc1+ad1, 0
	//
	// use xor to make it a worse multiply
	zzz_adPbc0_adPbc1_zzz := bc0_bc1_ad0_ad1.Xor(ad0_ad1_bc0_bc1).RotateAllLeft(32).And(middle)
	return ac0_ac1_bd0_bd1.Xor(zzz_adPbc0_adPbc1_zzz)
}

// carrylessMultiply is constant time carrless multiply implemented with an
// absurd number of multiplication given that the emulation platforms only have
// 32x32 into 64, it might make sense to rework this into that primitive, but,
// for now this works and is easily tested in scalar Go.
func carrylessMultiply_(x, y archsimd.Uint64x2) archsimd.Uint64x2 {

	// This by masking the two inputs into 5 thinned inputs, with
	// 4 zeroes separating any 2 set bits.  Multiply will potentially
	// set more bits with addition of overlapping terms, however this
	// technique allows as many as 31 additions (filling all 4 separation
	// positions with 1) without perturbing the bits we care about.  Since
	// there's at most 13 set bits in a thinned input, 31 is not a problem.
	// If there were only 3 set bits, there are 16 1s per thinned input and
	// only 15 additions can be tolerated -- so that's not possible.

	// This is also discussed at
	// https://timtaubert.de/blog/2017/06/verified-binary-multiplication-for-ghash/

	x0 := x.And(m0_)
	x1 := x.And(m1_)
	x2 := x.And(m2_)
	x3 := x.And(m3_)
	x4 := x.And(m4_)

	y0 := y.And(m0_)
	y1 := y.And(m1_)
	y2 := y.And(m2_)
	y3 := y.And(m3_)
	y4 := y.And(m4_)

	var z archsimd.Uint64x2
	// for a given line, combining (xI).mwl_(yJ) terms, I+J == K mod 5; mask index = K
	z = (mwl(x0, y0)).Xor(mwl(x1, y4)).Xor(mwl(x4, y1)).Xor(mwl(x2, y3)).Xor(mwl(x3, y2)).And(m0_)
	z = (mwl(x3, y3)).Xor(mwl(x2, y4)).Xor(mwl(x4, y2)).Xor(mwl(x0, y1)).Xor(mwl(x1, y0)).And(m1_).Or(z)
	z = (mwl(x1, y1)).Xor(mwl(x3, y4)).Xor(mwl(x4, y3)).Xor(mwl(x0, y2)).Xor(mwl(x2, y0)).And(m2_).Or(z)
	z = (mwl(x4, y4)).Xor(mwl(x0, y3)).Xor(mwl(x3, y0)).Xor(mwl(x1, y2)).Xor(mwl(x2, y1)).And(m3_).Or(z)
	z = (mwl(x2, y2)).Xor(mwl(x0, y4)).Xor(mwl(x4, y0)).Xor(mwl(x1, y3)).Xor(mwl(x3, y1)).And(m4_).Or(z)

	return z
}
