// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package archsimd

func new64x2(lo, hi uint64) Uint64x2 {
	return Uint64x2{}.SetElem(0, lo).SetElem(1, hi)
}

// These masks all have 4 zeroes between 1s.
var m0 = new64x2(0x1084210842108421, 0x2108421084210842)
var m1 = new64x2(0x2108421084210842, 0x4210842108421084)
var m2 = new64x2(0x4210842108421084, 0x8421084210842108)
var m3 = new64x2(0x8421084210842108, 0x0842108421084210)
var m4 = new64x2(0x0842108421084210, 0x1084210842108421)

// Selects the middle 64 bits of a 128-bit simd value
var middle = new64x2(0xffffffff00000000, 0x00000000ffffffff)

// mwl is a 64x64 into 128 multiply that is missing
// some carries that we don't need for CLMUL emulation.
// The high 64 bits of each input are ignored.
// Also just for fun, accumulate sums with Xor.
func (x Uint64x2) mwl(y Uint64x2) Uint64x2 {
	// reshape input into Uint32x4
	// input is  {a b _ _}.mwl{c d _ _}
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
	bc0_bc1_ad0_ad1 := Uint64x2{}.SetElem(0, ad0_ad1_bc0_bc1.GetElem(1)).SetElem(1, ad0_ad1_bc0_bc1.GetElem(0))

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
func (x Uint64x2) carrylessMultiply(y Uint64x2) Uint64x2 {

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

	x0 := x.And(m0)
	x1 := x.And(m1)
	x2 := x.And(m2)
	x3 := x.And(m3)
	x4 := x.And(m4)

	y0 := y.And(m0)
	y1 := y.And(m1)
	y2 := y.And(m2)
	y3 := y.And(m3)
	y4 := y.And(m4)

	var z Uint64x2
	// for a given line, combining (xI).mwl(yJ) terms, I+J == K mod 5; mask index = K
	z = (x0.mwl(y0)).Xor(x1.mwl(y4)).Xor(x4.mwl(y1)).Xor(x2.mwl(y3)).Xor(x3.mwl(y2)).And(m0)
	z = (x3.mwl(y3)).Xor(x2.mwl(y4)).Xor(x4.mwl(y2)).Xor(x0.mwl(y1)).Xor(x1.mwl(y0)).And(m1).Or(z)
	z = (x1.mwl(y1)).Xor(x3.mwl(y4)).Xor(x4.mwl(y3)).Xor(x0.mwl(y2)).Xor(x2.mwl(y0)).And(m2).Or(z)
	z = (x4.mwl(y4)).Xor(x0.mwl(y3)).Xor(x3.mwl(y0)).Xor(x1.mwl(y2)).Xor(x2.mwl(y1)).And(m3).Or(z)
	z = (x2.mwl(y2)).Xor(x0.mwl(y4)).Xor(x4.mwl(y0)).Xor(x1.mwl(y3)).Xor(x3.mwl(y1)).And(m4).Or(z)

	return z
}
