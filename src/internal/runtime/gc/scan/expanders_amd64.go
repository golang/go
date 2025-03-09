// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package scan

import (
	"simd"
	"unsafe"
)

var gcExpandersAVX512 = [68]func(unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8){
	nil,
	expandAVX512_1,
	expandAVX512_2,
	expandAVX512_3,
	expandAVX512_4,
	expandAVX512_6,
	expandAVX512_8,
	expandAVX512_10,
	expandAVX512_12,
	expandAVX512_14,
	expandAVX512_16,
	expandAVX512_18,
	expandAVX512_20,
	expandAVX512_22,
	expandAVX512_24,
	expandAVX512_26,
	expandAVX512_28,
	expandAVX512_30,
	expandAVX512_32,
	expandAVX512_36,
	expandAVX512_40,
	expandAVX512_44,
	expandAVX512_48,
	expandAVX512_52,
	expandAVX512_56,
	expandAVX512_60,
	expandAVX512_64,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
	nil,
}

func expandAVX512_1(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	x := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	y := simd.LoadUint64x8((*[8]uint64)(unsafe.Pointer(uintptr(src) + 64))).AsUint8x64()
	return x.AsUint64x8(), y.AsUint64x8()
}

var expandAVX512_2_mat0 = [8]uint64{
	0x0101020204040808, 0x1010202040408080, 0x0101020204040808, 0x1010202040408080,
	0x0101020204040808, 0x1010202040408080, 0x0101020204040808, 0x1010202040408080,
}
var expandAVX512_2_inShuf0 = [8]uint64{
	0x0706050403020100, 0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908,
	0x1716151413121110, 0x1716151413121110, 0x1f1e1d1c1b1a1918, 0x1f1e1d1c1b1a1918,
}
var expandAVX512_2_inShuf1 = [8]uint64{
	0x2726252423222120, 0x2726252423222120, 0x2f2e2d2c2b2a2928, 0x2f2e2d2c2b2a2928,
	0x3736353433323130, 0x3736353433323130, 0x3f3e3d3c3b3a3938, 0x3f3e3d3c3b3a3938,
}
var expandAVX512_2_outShufLo = [8]uint64{
	0x0b030a0209010800, 0x0f070e060d050c04, 0x1b131a1219111810, 0x1f171e161d151c14,
	0x2b232a2229212820, 0x2f272e262d252c24, 0x3b333a3239313830, 0x3f373e363d353c34,
}

func expandAVX512_2(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_2_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_2_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_2_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_2_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}

var expandAVX512_3_mat0 = [8]uint64{
	0x0101010202020404, 0x0408080810101020, 0x2020404040808080, 0x0101010202020404,
	0x0408080810101020, 0x2020404040808080, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_3_inShuf0 = [8]uint64{
	0x0706050403020100, 0x0706050403020100, 0x0706050403020100, 0x0f0e0d0c0b0a0908,
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_3_inShuf1 = [8]uint64{
	0x1716151413121110, 0x1716151413121110, 0x1716151413121110, 0x1f1e1d1c1b1a1918,
	0x1f1e1d1c1b1a1918, 0x1f1e1d1c1b1a1918, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_3_inShuf2 = [8]uint64{
	0x2726252423222120, 0x2726252423222120, 0x2726252423222120, 0xffffffffff2a2928,
	0xffffffffff2a2928, 0xffffffffffff2928, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_3_outShufLo = [8]uint64{
	0x0a02110901100800, 0x05140c04130b0312, 0x170f07160e06150d, 0x221a292119282018,
	0x1d2c241c2b231b2a, 0x2f271f2e261e2d25, 0x4a42514941504840, 0x45544c44534b4352,
}
var expandAVX512_3_outShufHi = [8]uint64{
	0x170f07160e06150d, 0x221a292119282018, 0x1d2c241c2b231b2a, 0x2f271f2e261e2d25,
	0x4a42514941504840, 0x45544c44534b4352, 0x574f47564e46554d, 0x625a696159686058,
}

func expandAVX512_3(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_3_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_3_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_3_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_3_inShuf2).AsUint8x64()
	v11 := simd.LoadUint64x8(&expandAVX512_3_outShufLo).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_3_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v0.Permute(v8)
	v10 := v9.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v12 := v4.ConcatPermute(v7, v11)
	v14 := v7.ConcatPermute(v10, v13)
	return v12.AsUint64x8(), v14.AsUint64x8()
}

var expandAVX512_4_mat0 = [8]uint64{
	0x0101010102020202, 0x0404040408080808, 0x1010101020202020, 0x4040404080808080,
	0x0101010102020202, 0x0404040408080808, 0x1010101020202020, 0x4040404080808080,
}
var expandAVX512_4_inShuf0 = [8]uint64{
	0x0706050403020100, 0x0706050403020100, 0x0706050403020100, 0x0706050403020100,
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908,
}
var expandAVX512_4_inShuf1 = [8]uint64{
	0x1716151413121110, 0x1716151413121110, 0x1716151413121110, 0x1716151413121110,
	0x1f1e1d1c1b1a1918, 0x1f1e1d1c1b1a1918, 0x1f1e1d1c1b1a1918, 0x1f1e1d1c1b1a1918,
}
var expandAVX512_4_outShufLo = [8]uint64{
	0x1911090118100800, 0x1b130b031a120a02, 0x1d150d051c140c04, 0x1f170f071e160e06,
	0x3931292138302820, 0x3b332b233a322a22, 0x3d352d253c342c24, 0x3f372f273e362e26,
}

func expandAVX512_4(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_4_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_4_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_4_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_4_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}

var expandAVX512_6_mat0 = [8]uint64{
	0x0101010101010202, 0x0202020204040404, 0x0404080808080808, 0x1010101010102020,
	0x2020202040404040, 0x4040808080808080, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_6_inShuf0 = [8]uint64{
	0x0706050403020100, 0x0706050403020100, 0x0706050403020100, 0x0706050403020100,
	0x0706050403020100, 0x0706050403020100, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_6_inShuf1 = [8]uint64{
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908,
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_6_inShuf2 = [8]uint64{
	0xffff151413121110, 0xffff151413121110, 0xffffff1413121110, 0xffffff1413121110,
	0xffffff1413121110, 0xffffff1413121110, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_6_outShufLo = [8]uint64{
	0x0901282018100800, 0x1a120a0229211911, 0x2b231b130b032a22, 0x0d052c241c140c04,
	0x1e160e062d251d15, 0x2f271f170f072e26, 0x4941686058504840, 0x5a524a4269615951,
}
var expandAVX512_6_outShufHi = [8]uint64{
	0x2b231b130b032a22, 0x0d052c241c140c04, 0x1e160e062d251d15, 0x2f271f170f072e26,
	0x4941686058504840, 0x5a524a4269615951, 0x6b635b534b436a62, 0x4d456c645c544c44,
}

func expandAVX512_6(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_6_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_6_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_6_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_6_inShuf2).AsUint8x64()
	v11 := simd.LoadUint64x8(&expandAVX512_6_outShufLo).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_6_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v0.Permute(v8)
	v10 := v9.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v12 := v4.ConcatPermute(v7, v11)
	v14 := v7.ConcatPermute(v10, v13)
	return v12.AsUint64x8(), v14.AsUint64x8()
}

var expandAVX512_8_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_8_inShuf0 = [8]uint64{
	0x0706050403020100, 0x0706050403020100, 0x0706050403020100, 0x0706050403020100,
	0x0706050403020100, 0x0706050403020100, 0x0706050403020100, 0x0706050403020100,
}
var expandAVX512_8_inShuf1 = [8]uint64{
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908,
	0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908, 0x0f0e0d0c0b0a0908,
}
var expandAVX512_8_outShufLo = [8]uint64{
	0x3830282018100800, 0x3931292119110901, 0x3a322a221a120a02, 0x3b332b231b130b03,
	0x3c342c241c140c04, 0x3d352d251d150d05, 0x3e362e261e160e06, 0x3f372f271f170f07,
}

func expandAVX512_8(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_8_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_8_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_8_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_8_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}

var expandAVX512_10_mat0 = [8]uint64{
	0x0101010101010101, 0x0101020202020202, 0x0202020204040404, 0x0404040404040808,
	0x0808080808080808, 0x1010101010101010, 0x1010202020202020, 0x2020202040404040,
}
var expandAVX512_10_inShuf0 = [8]uint64{
	0xff06050403020100, 0xff06050403020100, 0xff06050403020100, 0xff06050403020100,
	0xffff050403020100, 0xffff050403020100, 0xffff050403020100, 0xffff050403020100,
}
var expandAVX512_10_mat1 = [8]uint64{
	0x4040404040408080, 0x8080808080808080, 0x0808080808080808, 0x1010101010101010,
	0x1010202020202020, 0x2020202040404040, 0x4040404040408080, 0x8080808080808080,
}
var expandAVX512_10_inShuf1 = [8]uint64{
	0xffff050403020100, 0xffff050403020100, 0xff0c0b0a09080706, 0xff0c0b0a09080706,
	0xff0c0b0a09080706, 0xff0c0b0a09080706, 0xffff0b0a09080706, 0xffff0b0a09080706,
}
var expandAVX512_10_mat2 = [8]uint64{
	0x0101010101010101, 0x0101020202020202, 0x0202020204040404, 0x0404040404040808,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_10_inShuf2 = [8]uint64{
	0xffff0c0b0a090807, 0xffff0c0b0a090807, 0xffff0c0b0a090807, 0xffff0c0b0a090807,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_10_outShufLo = [8]uint64{
	0x3830282018100800, 0x2921191109014840, 0x1a120a0249413931, 0x0b034a423a322a22,
	0x4b433b332b231b13, 0x3c342c241c140c04, 0x2d251d150d054c44, 0x1e160e064d453d35,
}
var expandAVX512_10_outShufHi = [8]uint64{
	0x4840383028201810, 0x3931292119115850, 0x2a221a1259514941, 0x1b135a524a423a32,
	0x5b534b433b332b23, 0x4c443c342c241c14, 0x3d352d251d155c54, 0x2e261e165d554d45,
}

func expandAVX512_10(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_10_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_10_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_10_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_10_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_10_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_10_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_10_outShufLo).AsUint8x64()
	v15 := simd.LoadUint64x8(&expandAVX512_10_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v14 := v4.ConcatPermute(v8, v13)
	v16 := v8.ConcatPermute(v12, v15)
	return v14.AsUint64x8(), v16.AsUint64x8()
}

var expandAVX512_12_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_12_inShuf0 = [8]uint64{
	0xffff050403020100, 0xffff050403020100, 0xffff050403020100, 0xffff050403020100,
	0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100,
}
var expandAVX512_12_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_12_inShuf1 = [8]uint64{
	0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100,
	0xffff0a0908070605, 0xffff0a0908070605, 0xffff0a0908070605, 0xffff0a0908070605,
}
var expandAVX512_12_mat2 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
}
var expandAVX512_12_inShuf2 = [8]uint64{
	0xffffff0908070605, 0xffffff0908070605, 0xffffff0908070605, 0xffffff0908070605,
	0xffffff0a09080706, 0xffffff0a09080706, 0xffffff0a09080706, 0xffffff0a09080706,
}
var expandAVX512_12_outShufLo = [8]uint64{
	0x3830282018100800, 0x1911090158504840, 0x5951494139312921, 0x3a322a221a120a02,
	0x1b130b035a524a42, 0x5b534b433b332b23, 0x3c342c241c140c04, 0x1d150d055c544c44,
}
var expandAVX512_12_outShufHi = [8]uint64{
	0x5850484038302820, 0x3931292178706860, 0x7971696159514941, 0x5a524a423a322a22,
	0x3b332b237a726a62, 0x7b736b635b534b43, 0x5c544c443c342c24, 0x3d352d257c746c64,
}

func expandAVX512_12(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_12_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_12_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_12_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_12_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_12_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_12_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_12_outShufLo).AsUint8x64()
	v15 := simd.LoadUint64x8(&expandAVX512_12_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v14 := v4.ConcatPermute(v8, v13)
	v16 := v8.ConcatPermute(v12, v15)
	return v14.AsUint64x8(), v16.AsUint64x8()
}

var expandAVX512_14_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010101010202, 0x0202020202020202, 0x0202020204040404,
	0x0404040404040404, 0x0404080808080808, 0x0808080808080808, 0x1010101010101010,
}
var expandAVX512_14_inShuf0 = [8]uint64{
	0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100,
	0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100, 0xffffff0403020100,
}
var expandAVX512_14_mat1 = [8]uint64{
	0x1010101010102020, 0x2020202020202020, 0x2020202040404040, 0x4040404040404040,
	0x4040808080808080, 0x8080808080808080, 0x1010101010102020, 0x2020202020202020,
}
var expandAVX512_14_inShuf1 = [8]uint64{
	0xffffffff03020100, 0xffffffff03020100, 0xffffffff03020100, 0xffffffff03020100,
	0xffffffff03020100, 0xffffffff03020100, 0xffffff0807060504, 0xffffff0807060504,
}
var expandAVX512_14_mat2 = [8]uint64{
	0x2020202040404040, 0x4040404040404040, 0x4040808080808080, 0x8080808080808080,
	0x0101010101010101, 0x0101010101010202, 0x0202020202020202, 0x0202020204040404,
}
var expandAVX512_14_inShuf2 = [8]uint64{
	0xffffff0807060504, 0xffffff0807060504, 0xffffff0807060504, 0xffffff0807060504,
	0xffffff0908070605, 0xffffff0908070605, 0xffffffff08070605, 0xffffffff08070605,
}
var expandAVX512_14_mat3 = [8]uint64{
	0x0404040404040404, 0x0404080808080808, 0x0808080808080808, 0x1010101010101010,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_14_inShuf3 = [8]uint64{
	0xffffffff08070605, 0xffffffff08070605, 0xffffffff08070605, 0xffffffff08070605,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_14_outShufLo = [8]uint64{
	0x3830282018100800, 0x0901686058504840, 0x4941393129211911, 0x1a120a0269615951,
	0x5a524a423a322a22, 0x2b231b130b036a62, 0x6b635b534b433b33, 0x3c342c241c140c04,
}
var expandAVX512_14_outShufHi0 = [8]uint64{
	0x6860585048403830, 0x3931ffffffff7870, 0x7971696159514941, 0x4a423a32ffffffff,
	0xffff7a726a625a52, 0x5b534b433b33ffff, 0xffffffff7b736b63, 0x6c645c544c443c34,
}
var expandAVX512_14_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffff18100800ffff, 0xffffffffffffffff, 0xffffffff19110901,
	0x0a02ffffffffffff, 0xffffffffffff1a12, 0x1b130b03ffffffff, 0xffffffffffffffff,
}

func expandAVX512_14(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_14_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_14_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_14_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_14_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_14_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_14_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_14_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_14_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_14_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_14_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_14_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xff0ffc3ff0ffc3ff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0xf003c00f003c00)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_16_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_16_inShuf0 = [8]uint64{
	0x0303020201010000, 0x0303020201010000, 0x0303020201010000, 0x0303020201010000,
	0x0303020201010000, 0x0303020201010000, 0x0303020201010000, 0x0303020201010000,
}
var expandAVX512_16_inShuf1 = [8]uint64{
	0x0707060605050404, 0x0707060605050404, 0x0707060605050404, 0x0707060605050404,
	0x0707060605050404, 0x0707060605050404, 0x0707060605050404, 0x0707060605050404,
}
var expandAVX512_16_outShufLo = [8]uint64{
	0x1918111009080100, 0x3938313029282120, 0x1b1a13120b0a0302, 0x3b3a33322b2a2322,
	0x1d1c15140d0c0504, 0x3d3c35342d2c2524, 0x1f1e17160f0e0706, 0x3f3e37362f2e2726,
}

func expandAVX512_16(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_16_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_16_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_16_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_16_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}

var expandAVX512_18_mat0 = [8]uint64{
	0x0101010101010101, 0x0101020202020202, 0x0202020202020202, 0x0202020204040404,
	0x0404040404040404, 0x0404040404040808, 0x0808080808080808, 0x1010101010101010,
}
var expandAVX512_18_inShuf0 = [8]uint64{
	0x0303020201010000, 0xffffffff03020100, 0xffffffff03020100, 0xffffffff03020100,
	0xffffffff03020100, 0xffffffff03020100, 0x0303020201010000, 0xff03020201010000,
}
var expandAVX512_18_mat1 = [8]uint64{
	0x1010202020202020, 0x2020202020202020, 0x2020202040404040, 0x4040404040404040,
	0x4040404040408080, 0x8080808080808080, 0x1010101010101010, 0x1010202020202020,
}
var expandAVX512_18_inShuf1 = [8]uint64{
	0xffffffffff020100, 0xffffffffff020100, 0xffffffffff020100, 0xffffffffff020100,
	0xffffffffff020100, 0xffff020201010000, 0xff06060505040403, 0xffffffff06050403,
}
var expandAVX512_18_mat2 = [8]uint64{
	0x2020202020202020, 0x2020202040404040, 0x4040404040404040, 0x4040404040408080,
	0x8080808080808080, 0x0101010101010101, 0x0101020202020202, 0x0202020202020202,
}
var expandAVX512_18_inShuf2 = [8]uint64{
	0xffffffff06050403, 0xffffffff06050403, 0xffffffff06050403, 0xffffffff06050403,
	0x0606050504040303, 0x0707060605050404, 0xffffffffff060504, 0xffffffffff060504,
}
var expandAVX512_18_mat3 = [8]uint64{
	0x0202020204040404, 0x0404040404040404, 0x0404040404040808, 0x0808080808080808,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_18_inShuf3 = [8]uint64{
	0xffffffffff060504, 0xffffffffff060504, 0xffffffffff060504, 0xffff060605050404,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_18_outShufLo = [8]uint64{
	0x3028201810080100, 0x6058504840393831, 0x2119110903026968, 0x5149413b3a333229,
	0x120a05046b6a6159, 0x423d3c35342a221a, 0x07066d6c625a524a, 0x3e37362b231b130b,
}
var expandAVX512_18_outShufHi0 = [8]uint64{
	0x6160585048403830, 0xffffffff78706968, 0x59514941393231ff, 0xffff79716b6a6362,
	0x4a423a3433ffffff, 0x7a726d6c65645a52, 0x3b3635ffffffffff, 0x6f6e67665b534b43,
}
var expandAVX512_18_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0x18100800ffffffff, 0xffffffffffffff19, 0x0901ffffffffffff,
	0xffffffffff1b1a11, 0xffffffffffffffff, 0xffffff1d1c120a02, 0xffffffffffffffff,
}

func expandAVX512_18(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_18_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_18_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_18_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_18_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_18_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_18_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_18_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_18_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_18_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_18_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_18_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xffe0fff83ffe0fff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x1f0007c001f000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_20_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_20_inShuf0 = [8]uint64{
	0x0303020201010000, 0xffffffff03020100, 0xff03020201010000, 0xffff020201010000,
	0xffffffffff020100, 0xffff020201010000, 0xffff020201010000, 0xffffffffff020100,
}
var expandAVX512_20_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0202020202020202, 0x0404040404040404, 0x0404040408080808, 0x0808080808080808,
}
var expandAVX512_20_inShuf1 = [8]uint64{
	0xffff020201010000, 0xffff020201010000, 0xffffffffff020100, 0xffff020201010000,
	0xff06060505040403, 0x0606050504040303, 0xffffffff06050403, 0xffff050504040303,
}
var expandAVX512_20_mat2 = [8]uint64{
	0x1010101010101010, 0x1010101020202020, 0x2020202020202020, 0x4040404040404040,
	0x4040404080808080, 0x8080808080808080, 0x0101010101010101, 0x0101010102020202,
}
var expandAVX512_20_inShuf2 = [8]uint64{
	0xffff050504040303, 0xffffffffff050403, 0xffff050504040303, 0xffff050504040303,
	0xffffffffff050403, 0xffff050504040303, 0xffff060605050404, 0xffffffffff060504,
}
var expandAVX512_20_outShufLo = [8]uint64{
	0x2019181110080100, 0x4841403831302928, 0x1209030259585049, 0x33322b2a211b1a13,
	0x5b5a514b4a434239, 0x221d1c15140a0504, 0x4c45443a35342d2c, 0x160b07065d5c524d,
}
var expandAVX512_20_outShufHi = [8]uint64{
	0x4140393830292820, 0x6968605958515048, 0x312b2a2221787170, 0x5a53524943423b3a,
	0x237973726b6a615b, 0x45443d3c322d2c24, 0x6d6c625d5c55544a, 0x332f2e26257a7574,
}

func expandAVX512_20(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_20_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_20_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_20_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_20_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_20_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_20_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_20_outShufLo).AsUint8x64()
	v15 := simd.LoadUint64x8(&expandAVX512_20_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v14 := v4.ConcatPermute(v8, v13)
	v16 := v8.ConcatPermute(v12, v15)
	return v14.AsUint64x8(), v16.AsUint64x8()
}

var expandAVX512_22_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010101010202, 0x0202020202020202, 0x0202020204040404,
	0x0404040404040404, 0x0404080808080808, 0x0808080808080808, 0x1010101010101010,
}
var expandAVX512_22_inShuf0 = [8]uint64{
	0xffff020201010000, 0xffffffffff020100, 0xffff020201010000, 0xffffffffff020100,
	0xffff020201010000, 0xffffffffff020100, 0xffff020201010000, 0xffff020201010000,
}
var expandAVX512_22_mat1 = [8]uint64{
	0x1010101010102020, 0x2020202020202020, 0x2020202040404040, 0x4040404040404040,
	0x4040808080808080, 0x8080808080808080, 0x8080808080808080, 0x0101010101010101,
}
var expandAVX512_22_inShuf1 = [8]uint64{
	0xffffffffff020100, 0xffff020201010000, 0xffffffffff020100, 0xffff020201010000,
	0xffffffffff020100, 0xffffffff01010000, 0xffff040403030202, 0xffff050504040303,
}
var expandAVX512_22_mat2 = [8]uint64{
	0x0101010101010202, 0x0202020202020202, 0x0202020204040404, 0x0404040404040404,
	0x0404080808080808, 0x0808080808080808, 0x1010101010101010, 0x1010101010102020,
}
var expandAVX512_22_inShuf2 = [8]uint64{
	0xffffffffff050403, 0xffff050504040303, 0xffffffffff050403, 0xffff050504040303,
	0xffffffffff050403, 0xffff050504040303, 0xffff050504040303, 0xffffffffff050403,
}
var expandAVX512_22_mat3 = [8]uint64{
	0x2020202020202020, 0x2020202040404040, 0x4040404040404040, 0x4040808080808080,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_22_inShuf3 = [8]uint64{
	0xffff050504040303, 0xffffffffff050403, 0xffffff0504040303, 0xffffffffffff0403,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_22_outShufLo = [8]uint64{
	0x2120181110080100, 0x4948403938313028, 0x0302696860595850, 0x3229232219131209,
	0x5a514b4a413b3a33, 0x140a05046b6a615b, 0x3c35342a25241a15, 0x625d5c524d4c423d,
}
var expandAVX512_22_outShufHi0 = [8]uint64{
	0x5049484039383130, 0x7871706968605958, 0x3332ffffffffffff, 0x5b5a514b4a413b3a,
	0xffff7973726b6a61, 0x3d3c3534ffffffff, 0x6c625d5c524d4c42, 0xffffffff7a75746d,
}
var expandAVX512_22_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffff181110080100, 0xffffffffffffffff,
	0x0302ffffffffffff, 0xffffffff19131209, 0xffffffffffffffff, 0x140a0504ffffffff,
}

func expandAVX512_22(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_22_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_22_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_22_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_22_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_22_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_22_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_22_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_22_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_22_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_22_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_22_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xffff03fffc0ffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0xf0000fc0003f0000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_24_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_24_inShuf0 = [8]uint64{
	0x0202010101000000, 0x0202010101000000, 0x0202010101000000, 0x0202010101000000,
	0x0202010101000000, 0xff02010101000000, 0xffff010101000000, 0xffff010101000000,
}
var expandAVX512_24_inShuf1 = [8]uint64{
	0xffffffffffffff02, 0xffffffffffffff02, 0xffffffffffffff02, 0xffffffffffffff02,
	0xffffffffffffff02, 0x0404040303030202, 0x0404030303020202, 0x0404030303020202,
}
var expandAVX512_24_mat2 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x4040404040404040, 0x8080808080808080, 0x0101010101010101,
}
var expandAVX512_24_inShuf2 = [8]uint64{
	0x0505040404030303, 0x0505040404030303, 0x0505040404030303, 0xffff040404030303,
	0xffff040404030303, 0xffffffffffffff04, 0xffffffffffffff04, 0xffffffffffffff05,
}
var expandAVX512_24_mat3 = [8]uint64{
	0x0202020202020202, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_24_inShuf3 = [8]uint64{
	0xffffffffffffff05, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_24_outShufLo = [8]uint64{
	0x11100a0908020100, 0x282221201a191812, 0x3a39383231302a29, 0x14130d0c0b050403,
	0x2b2524231d1c1b15, 0x3d3c3b3534332d2c, 0x1716480f0e400706, 0x2e602726581f1e50,
}
var expandAVX512_24_outShufHi0 = [8]uint64{
	0x3a39383231302928, 0x51504a4948424140, 0x2a6261605a595852, 0x3d3c3b3534332c2b,
	0x54534d4c4b454443, 0x2d6564635d5c5b55, 0x703f3e6837362f2e, 0x5756ff4f4e784746,
}
var expandAVX512_24_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffff00ffffffffff,
}

func expandAVX512_24(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_24_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_24_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_24_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_24_mat2).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_24_inShuf2).AsUint8x64()
	v12 := simd.LoadUint64x8(&expandAVX512_24_mat3).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_24_inShuf3).AsUint8x64()
	v16 := simd.LoadUint64x8(&expandAVX512_24_outShufLo).AsUint8x64()
	v18 := simd.LoadUint64x8(&expandAVX512_24_outShufHi0).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_24_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v10 := v0.Permute(v9)
	v11 := v10.GaloisFieldAffineTransform(v8.AsUint64x8(), 0)
	v14 := v0.Permute(v13)
	v15 := v14.GaloisFieldAffineTransform(v12.AsUint64x8(), 0)
	v17 := v4.ConcatPermute(v7, v16)
	u0 := uint64(0xdfffffffffffffff)
	m0 := simd.Mask8x64FromBits(u0)
	v20 := v7.ConcatPermute(v11, v18).Masked(m0)
	u1 := uint64(0x2000000000000000)
	m1 := simd.Mask8x64FromBits(u1)
	v21 := v15.Permute(v19).Masked(m1)
	v22 := v20.Or(v21)
	return v17.AsUint64x8(), v22.AsUint64x8()
}

var expandAVX512_26_mat0 = [8]uint64{
	0x0101010101010101, 0x0101020202020202, 0x0202020202020202, 0x0202020204040404,
	0x0404040404040404, 0x0404040404040808, 0x0808080808080808, 0x1010101010101010,
}
var expandAVX512_26_inShuf0 = [8]uint64{
	0x0202010101000000, 0xffffffffff020100, 0xffff020201010000, 0xffffffffff020100,
	0xffff020201010000, 0xffffffffff020100, 0x0202010101000000, 0xffff010101000000,
}
var expandAVX512_26_mat1 = [8]uint64{
	0x1010202020202020, 0x2020202020202020, 0x2020202040404040, 0x4040404040404040,
	0x4040404040408080, 0x8080808080808080, 0x0101010101010101, 0x0808080808080808,
}
var expandAVX512_26_inShuf1 = [8]uint64{
	0xffffffffffff0100, 0xffffffff01010000, 0xffffffffffff0100, 0xffffffff01010000,
	0xffffffffffff0100, 0xffff010101000000, 0xffffffffffffff02, 0xff04040403030302,
}
var expandAVX512_26_mat2 = [8]uint64{
	0x1010101010101010, 0x1010202020202020, 0x2020202020202020, 0x2020202040404040,
	0x4040404040404040, 0x4040404040408080, 0x8080808080808080, 0x0101010101010101,
}
var expandAVX512_26_inShuf2 = [8]uint64{
	0x0404030303020202, 0xffffffffff040302, 0xffff040403030202, 0xffffffffff040302,
	0xffff040403030202, 0xffffffffff040302, 0xff04030303020202, 0xffff040404030303,
}
var expandAVX512_26_mat3 = [8]uint64{
	0x0101020202020202, 0x0202020202020202, 0x0202020204040404, 0x0404040404040404,
	0x0404040404040808, 0x1010101010101010, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_26_inShuf3 = [8]uint64{
	0xffffffffffff0403, 0xffffffff04040303, 0xffffffffffff0403, 0xffffffff04040303,
	0xffffffffffff0403, 0xffffffffffffff04, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_26_outShufLo = [8]uint64{
	0x2018111008020100, 0x3a39383231302821, 0x6860595850494840, 0x1312090504036a69,
	0x3b35343329232219, 0x5b5a514b4a413d3c, 0x0a7007066d6c6b61, 0x37362a25241a1514,
}
var expandAVX512_26_outShufHi0 = [8]uint64{
	0x5851504842414038, 0x7978727170686160, 0xffffffffffffff7a, 0x52494544433b3a39,
	0x7574736963625953, 0xffffffffff7d7c7b, 0xff47463e3d3cffff, 0x766a65645a55544a,
}
var expandAVX512_26_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0x20191810090800ff, 0xffffffffffffffff,
	0xffffffffffffffff, 0x1a110b0a01ffffff, 0x28ffffffffff211b, 0xffffffffffffffff,
}

func expandAVX512_26(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_26_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_26_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_26_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_26_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_26_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_26_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_26_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_26_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_26_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_26_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_26_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xff7c07ffff01ffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x83f80000fe0000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_28_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_28_inShuf0 = [8]uint64{
	0x0202010101000000, 0xffffffffff020100, 0x0202010101000000, 0xff02010101000000,
	0xffffffffffff0100, 0xffff010101000000, 0xffff010101000000, 0xffffffffffff0100,
}
var expandAVX512_28_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0404040408080808,
}
var expandAVX512_28_inShuf1 = [8]uint64{
	0xffff010101000000, 0xffff010101000000, 0xffffffffffff0100, 0xffff010101000000,
	0xffffffffffffff02, 0xffffffffffffff02, 0x0404040303030202, 0xffffffffff040302,
}
var expandAVX512_28_mat2 = [8]uint64{
	0x0808080808080808, 0x1010101010101010, 0x1010101020202020, 0x2020202020202020,
	0x4040404040404040, 0x4040404080808080, 0x8080808080808080, 0x0101010101010101,
}
var expandAVX512_28_inShuf2 = [8]uint64{
	0x0404030303020202, 0x0404030303020202, 0xffffffffffff0302, 0xffff030303020202,
	0xffff030303020202, 0xffffffffffff0302, 0xffff030303020202, 0xffff040404030303,
}
var expandAVX512_28_mat3 = [8]uint64{
	0x0101010102020202, 0x0202020202020202, 0x0808080808080808, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_28_inShuf3 = [8]uint64{
	0xffffffffffff0403, 0xffff040404030303, 0xffffffffffffff04, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_28_outShufLo = [8]uint64{
	0x1812111008020100, 0x31302a2928201a19, 0x4a49484241403832, 0x090504035a595850,
	0x2b211d1c1b151413, 0x4443393534332d2c, 0x5d5c5b514d4c4b45, 0x1e6817160a600706,
}
var expandAVX512_28_outShufHi0 = [8]uint64{
	0x4948424140383130, 0x6261605a5958504a, 0xff7a797872717068, 0x4339343332ffffff,
	0x5c5b514d4c4b4544, 0x757473696564635d, 0x35ffffffff7d7c7b, 0x4f4eff47463a3736,
}
var expandAVX512_28_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0x00ffffffffffffff, 0xffffffffff0a0908,
	0xffffffffffffffff, 0xffffffffffffffff, 0xff0d0c0b01ffffff, 0xffff10ffffffffff,
}

func expandAVX512_28(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_28_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_28_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_28_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_28_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_28_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_28_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_28_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_28_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_28_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_28_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_28_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xdf87fffff87fffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x2078000007800000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_30_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010101010202, 0x0202020202020202, 0x0202020204040404,
	0x0404040404040404, 0x0404080808080808, 0x0808080808080808, 0x1010101010101010,
}
var expandAVX512_30_inShuf0 = [8]uint64{
	0x0202010101000000, 0xffffffffff020100, 0xffff010101000000, 0xffffffffffff0100,
	0xffff010101000000, 0xffffffffffff0100, 0xffff010101000000, 0xffff010101000000,
}
var expandAVX512_30_mat1 = [8]uint64{
	0x1010101010102020, 0x2020202020202020, 0x2020202040404040, 0x4040404040404040,
	0x4040808080808080, 0x8080808080808080, 0x0101010101010101, 0x0202020202020202,
}
var expandAVX512_30_inShuf1 = [8]uint64{
	0xffffffffffff0100, 0xffff010101000000, 0xffffffffffff0100, 0xffff010101000000,
	0xffffffffffff0100, 0xffff010101000000, 0xffffffffffffff02, 0x0404030303020202,
}
var expandAVX512_30_mat2 = [8]uint64{
	0x0202020204040404, 0x0404040404040404, 0x0404080808080808, 0x0808080808080808,
	0x1010101010101010, 0x1010101010102020, 0x2020202020202020, 0x2020202040404040,
}
var expandAVX512_30_inShuf2 = [8]uint64{
	0xffffffffff040302, 0xffff030303020202, 0xffffffffffff0302, 0xffff030303020202,
	0xffff030303020202, 0xffffffffffff0302, 0xffff030303020202, 0xffffffffffff0302,
}
var expandAVX512_30_mat3 = [8]uint64{
	0x4040404040404040, 0x4040808080808080, 0x8080808080808080, 0x0101010101010101,
	0x0101010101010202, 0x0202020202020202, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_30_inShuf3 = [8]uint64{
	0xffff030303020202, 0xffffffffffff0302, 0xffff030303020202, 0xffff040404030303,
	0xffffffffffff0403, 0xffffffffffffff04, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_30_outShufLo = [8]uint64{
	0x1812111008020100, 0x3832313028222120, 0x58504a4948403a39, 0x04036a6968605a59,
	0x2423191514130905, 0x3d3c3b3534332925, 0x5d5c5b514d4c4b41, 0x0a7007066d6c6b61,
}
var expandAVX512_30_outShufHi0 = [8]uint64{
	0x504a4948403a3938, 0x70686261605a5958, 0xffffffffff787271, 0x3c3bffffffffffff,
	0x5c5b514d4c4b413d, 0x757473696564635d, 0xffffffffffffff79, 0x42ff3f3effffffff,
}
var expandAVX512_30_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0x1008020100ffffff, 0xffff201a19181211,
	0xffffffffffffffff, 0xffffffffffffffff, 0x15141309050403ff, 0xff28ffff211d1c1b,
}

func expandAVX512_30(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_30_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_30_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_30_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_30_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_30_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_30_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_30_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_30_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_30_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_30_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_30_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xb001ffffc007ffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x4ffe00003ff80000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_32_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_32_inShuf0 = [8]uint64{
	0x0101010100000000, 0x0101010100000000, 0x0101010100000000, 0x0101010100000000,
	0x0101010100000000, 0x0101010100000000, 0x0101010100000000, 0x0101010100000000,
}
var expandAVX512_32_inShuf1 = [8]uint64{
	0x0303030302020202, 0x0303030302020202, 0x0303030302020202, 0x0303030302020202,
	0x0303030302020202, 0x0303030302020202, 0x0303030302020202, 0x0303030302020202,
}
var expandAVX512_32_outShufLo = [8]uint64{
	0x0b0a090803020100, 0x1b1a191813121110, 0x2b2a292823222120, 0x3b3a393833323130,
	0x0f0e0d0c07060504, 0x1f1e1d1c17161514, 0x2f2e2d2c27262524, 0x3f3e3d3c37363534,
}

func expandAVX512_32(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_32_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_32_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_32_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_32_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}

var expandAVX512_36_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_36_inShuf0 = [8]uint64{
	0x0101010100000000, 0xffffffffffff0100, 0x0101010100000000, 0x0101010100000000,
	0xffffffffffff0100, 0x0101010100000000, 0x0101010100000000, 0xffffffffffff0100,
}
var expandAVX512_36_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x4040404040404040, 0x4040404080808080, 0x8080808080808080, 0x0101010101010101,
}
var expandAVX512_36_inShuf1 = [8]uint64{
	0x0101010100000000, 0xffffff0100000000, 0xffffffffffffff00, 0xffffffff00000000,
	0xff02020202010101, 0xffffffffffff0201, 0x0202020201010101, 0x0303030302020202,
}
var expandAVX512_36_mat2 = [8]uint64{
	0x0101010102020202, 0x0202020202020202, 0x0404040404040404, 0x0404040408080808,
	0x0808080808080808, 0x1010101010101010, 0x1010101020202020, 0x2020202020202020,
}
var expandAVX512_36_inShuf2 = [8]uint64{
	0xffffffffffff0302, 0x0303030302020202, 0x0303030302020202, 0xffffffffffff0302,
	0x0303030302020202, 0xffff030302020202, 0xffffffffffffff02, 0xffffffff02020202,
}
var expandAVX512_36_outShufLo = [8]uint64{
	0x1211100803020100, 0x2928201b1a191813, 0x4038333231302b2a, 0x504b4a4948434241,
	0x070605045b5a5958, 0x1e1d1c1716151409, 0x35342f2e2d2c211f, 0x4c47464544393736,
}
var expandAVX512_36_outShufHi = [8]uint64{
	0x3332313028222120, 0x4a4948403b3a3938, 0x616058535251504b, 0x78706b6a69686362,
	0x29262524237b7a79, 0x3f3e3d3c37363534, 0x5655544f4e4d4c41, 0x6d6c676665645957,
}

func expandAVX512_36(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_36_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_36_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_36_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_36_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_36_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_36_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_36_outShufLo).AsUint8x64()
	v15 := simd.LoadUint64x8(&expandAVX512_36_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v14 := v4.ConcatPermute(v8, v13)
	v16 := v8.ConcatPermute(v12, v15)
	return v14.AsUint64x8(), v16.AsUint64x8()
}

var expandAVX512_40_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_40_inShuf0 = [8]uint64{
	0x0101010000000000, 0x0101010000000000, 0x0101010000000000, 0x0101010000000000,
	0x0101010000000000, 0xffffff0000000000, 0xffffff0000000000, 0xffffff0000000000,
}
var expandAVX512_40_mat1 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x1010101010101010, 0x2020202020202020, 0x4040404040404040,
}
var expandAVX512_40_inShuf1 = [8]uint64{
	0xffffffffffff0101, 0xffffffffffff0101, 0xffffffffffff0101, 0xffffffffffff0101,
	0xffffffffffffff01, 0xffff020202020201, 0x0202020101010101, 0x0202020101010101,
}
var expandAVX512_40_mat2 = [8]uint64{
	0x8080808080808080, 0x0101010101010101, 0x0202020202020202, 0x0404040404040404,
	0x0808080808080808, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_40_inShuf2 = [8]uint64{
	0x0202020101010101, 0x0303030202020202, 0x0303030202020202, 0xffffff0202020202,
	0xffffff0202020202, 0xffffffffffff0202, 0xffffffffffff0202, 0xffffffffffff0202,
}
var expandAVX512_40_mat3 = [8]uint64{
	0x0101010101010101, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_40_inShuf3 = [8]uint64{
	0xffffffffffff0303, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_40_outShufLo = [8]uint64{
	0x0a09080403020100, 0x1814131211100c0b, 0x232221201c1b1a19, 0x31302c2b2a292824,
	0x3c3b3a3938343332, 0x0f0e0d4140070605, 0x1d51501716154948, 0x6027262559581f1e,
}
var expandAVX512_40_outShufHi0 = [8]uint64{
	0x3938343332313028, 0x44434241403c3b3a, 0x5251504c4b4a4948, 0x605c5b5a59585453,
	0x2c2b2a2964636261, 0x3e3d69683736352d, 0x797847464571703f, 0x575655ffff4f4e4d,
}
var expandAVX512_40_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffff0100ffffff,
}

func expandAVX512_40(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_40_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_40_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_40_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_40_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_40_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_40_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_40_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_40_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_40_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_40_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_40_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xe7ffffffffffffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x1800000000000000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_44_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_44_inShuf0 = [8]uint64{
	0x0101010000000000, 0xffffffffffff0100, 0x0101010000000000, 0x0101010000000000,
	0xffffffffffff0100, 0x0101010000000000, 0xffffff0000000000, 0xffffffffffffff00,
}
var expandAVX512_44_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
}
var expandAVX512_44_inShuf1 = [8]uint64{
	0xffffff0000000000, 0xffffff0000000000, 0xffffffffffffff00, 0xffffff0000000000,
	0xffffffffffff0101, 0xffffffffffff0101, 0xffffffffffff0101, 0xff02020202020101,
}
var expandAVX512_44_mat2 = [8]uint64{
	0x1010101010101010, 0x1010101020202020, 0x2020202020202020, 0x4040404040404040,
	0x4040404080808080, 0x8080808080808080, 0x0101010101010101, 0x0101010102020202,
}
var expandAVX512_44_inShuf2 = [8]uint64{
	0x0202020101010101, 0xffffffffffff0201, 0x0202020101010101, 0x0202020101010101,
	0xffffffffffff0201, 0xffff020101010101, 0xffffff0202020202, 0xffffffffffffff02,
}
var expandAVX512_44_mat3 = [8]uint64{
	0x0202020202020202, 0x0404040404040404, 0x0404040408080808, 0x1010101010101010,
	0x2020202020202020, 0x4040404040404040, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_44_inShuf3 = [8]uint64{
	0xffffff0202020202, 0xffffff0202020202, 0xffffffffffffff02, 0xffffffffffff0202,
	0xffffffffffff0202, 0xffffffffffff0202, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_44_outShufLo = [8]uint64{
	0x1110080403020100, 0x1c1b1a1918141312, 0x31302c2b2a292820, 0x4342414038343332,
	0x58504c4b4a494844, 0x600706055c5b5a59, 0x1d69681716150961, 0x2f2e2d2171701f1e,
}
var expandAVX512_44_outShufHi0 = [8]uint64{
	0x4844434241403938, 0x5a59585453525150, 0x6c6b6a6968605c5b, 0xffff787473727170,
	0xffffffffffffffff, 0x46453e3d3c3b3aff, 0xff57565549ffff47, 0x6d61ffff5f5e5dff,
}
var expandAVX512_44_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x0100ffffffffffff,
	0x0c0b0a0908040302, 0xffffffffffffff10, 0x20ffffffff1918ff, 0xffff2928ffffff21,
}

func expandAVX512_44(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_44_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_44_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_44_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_44_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_44_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_44_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_44_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_44_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_44_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_44_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_44_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0xce79fe003fffffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x318601ffc0000000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_48_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_48_inShuf0 = [8]uint64{
	0x0101000000000000, 0x0101000000000000, 0x0101000000000000, 0xffff000000000000,
	0xffff000000000000, 0xffff000000000000, 0xffff000000000000, 0xffff000000000000,
}
var expandAVX512_48_mat1 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0404040404040404,
	0x0808080808080808, 0x1010101010101010, 0x2020202020202020, 0x4040404040404040,
}
var expandAVX512_48_inShuf1 = [8]uint64{
	0xffffffff01010101, 0xffffffff01010101, 0xffffffffffff0101, 0x0202020202020101,
	0x0202010101010101, 0x0202010101010101, 0x0202010101010101, 0xffff010101010101,
}
var expandAVX512_48_mat2 = [8]uint64{
	0x8080808080808080, 0x0101010101010101, 0x0202020202020202, 0x0808080808080808,
	0x1010101010101010, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_48_inShuf2 = [8]uint64{
	0xffff010101010101, 0xffff020202020202, 0xffff020202020202, 0xffffffff02020202,
	0xffffffff02020202, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_48_outShufLo = [8]uint64{
	0x0908050403020100, 0x131211100d0c0b0a, 0x1d1c1b1a19181514, 0x2928252423222120,
	0x333231302d2c2b2a, 0x3d3c3b3a39383534, 0x0f0e434241400706, 0x515017164b4a4948,
}
var expandAVX512_48_outShufHi = [8]uint64{
	0x2524232221201918, 0x31302d2c2b2a2928, 0x3b3a393835343332, 0x4544434241403d3c,
	0x51504d4c4b4a4948, 0x1d1c1b1a55545352, 0x5b5a595827261f1e, 0x3736636261602f2e,
}

func expandAVX512_48(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_48_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_48_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_48_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_48_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_48_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_48_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_48_outShufLo).AsUint8x64()
	v15 := simd.LoadUint64x8(&expandAVX512_48_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v14 := v4.ConcatPermute(v8, v13)
	v16 := v8.ConcatPermute(v12, v15)
	return v14.AsUint64x8(), v16.AsUint64x8()
}

var expandAVX512_52_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_52_inShuf0 = [8]uint64{
	0x0101000000000000, 0xffffffffffff0100, 0x0101000000000000, 0xffff000000000000,
	0xffffffffffffff00, 0xffff000000000000, 0xffff000000000000, 0xffffffffffffff00,
}
var expandAVX512_52_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0101010101010101, 0x0202020202020202, 0x0202020202020202, 0x0404040404040404,
}
var expandAVX512_52_inShuf1 = [8]uint64{
	0xffff000000000000, 0xffff000000000000, 0xffffffffffffff00, 0xffff000000000000,
	0xffffffff01010101, 0xffffffffff010101, 0xff02020202020201, 0x0202010101010101,
}
var expandAVX512_52_mat2 = [8]uint64{
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
}
var expandAVX512_52_inShuf2 = [8]uint64{
	0xffffffffffff0201, 0x0202010101010101, 0xffff010101010101, 0xffffffffffffff01,
	0xffff010101010101, 0xffff010101010101, 0xffffffffffffff01, 0xffff010101010101,
}
var expandAVX512_52_mat3 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0404040404040404, 0x0808080808080808,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_52_inShuf3 = [8]uint64{
	0xffff020202020202, 0xffffffffffffff02, 0xffffffff02020202, 0xffffffffffff0202,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_52_outShufLo = [8]uint64{
	0x1008050403020100, 0x1a19181514131211, 0x2b2a2928201d1c1b, 0x3534333231302d2c,
	0x4845444342414038, 0x5958504d4c4b4a49, 0x616007065d5c5b5a, 0x6a69681716096362,
}
var expandAVX512_52_outShufHi0 = [8]uint64{
	0x403d3c3b3a393830, 0x51504d4c4b4a4948, 0x6261605855545352, 0x6c6b6a6968656463,
	0x7d7c7b7a7978706d, 0x31ffffffffffffff, 0xff3f3e3635343332, 0xffff4f4e41ffffff,
}
var expandAVX512_52_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xff08050403020100, 0x10ffffffffffffff, 0x1918ffffff131211,
}

func expandAVX512_52(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_52_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_52_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_52_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_52_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_52_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_52_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_52_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_52_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_52_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_52_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_52_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0x387f80ffffffffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0xc7807f0000000000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_56_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_56_inShuf0 = [8]uint64{
	0x0100000000000000, 0x0100000000000000, 0xff00000000000000, 0xff00000000000000,
	0xff00000000000000, 0xff00000000000000, 0xff00000000000000, 0xff00000000000000,
}
var expandAVX512_56_inShuf1 = [8]uint64{
	0xffff010101010101, 0x0202010101010101, 0x0201010101010101, 0xff01010101010101,
	0xff01010101010101, 0xff01010101010101, 0xff01010101010101, 0xff01010101010101,
}
var expandAVX512_56_mat2 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_56_inShuf2 = [8]uint64{
	0xff02020202020202, 0xffffff0202020202, 0xffffffffffffff02, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_56_outShufLo = [8]uint64{
	0x0806050403020100, 0x11100e0d0c0b0a09, 0x1a19181615141312, 0x232221201e1d1c1b,
	0x2c2b2a2928262524, 0x3534333231302e2d, 0x3e3d3c3b3a393836, 0x0f45444342414007,
}
var expandAVX512_56_outShufHi = [8]uint64{
	0x11100d0c0b0a0908, 0x1a19181615141312, 0x232221201e1d1c1b, 0x2c2b2a2928262524,
	0x3534333231302e2d, 0x3e3d3c3b3a393836, 0x0e46454443424140, 0x50174c4b4a49480f,
}

func expandAVX512_56(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_56_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_56_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_56_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_56_mat2).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_56_inShuf2).AsUint8x64()
	v12 := simd.LoadUint64x8(&expandAVX512_56_outShufLo).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_56_outShufHi).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v10 := v0.Permute(v9)
	v11 := v10.GaloisFieldAffineTransform(v8.AsUint64x8(), 0)
	v13 := v4.ConcatPermute(v7, v12)
	v15 := v7.ConcatPermute(v11, v14)
	return v13.AsUint64x8(), v15.AsUint64x8()
}

var expandAVX512_60_mat0 = [8]uint64{
	0x0101010101010101, 0x0101010102020202, 0x0202020202020202, 0x0404040404040404,
	0x0404040408080808, 0x0808080808080808, 0x1010101010101010, 0x1010101020202020,
}
var expandAVX512_60_inShuf0 = [8]uint64{
	0x0100000000000000, 0xffffffffffffff00, 0xff00000000000000, 0xff00000000000000,
	0xffffffffffffff00, 0xff00000000000000, 0xff00000000000000, 0xffffffffffffff00,
}
var expandAVX512_60_mat1 = [8]uint64{
	0x2020202020202020, 0x4040404040404040, 0x4040404080808080, 0x8080808080808080,
	0x0101010101010101, 0x0101010101010101, 0x0101010102020202, 0x0202020202020202,
}
var expandAVX512_60_inShuf1 = [8]uint64{
	0xff00000000000000, 0xff00000000000000, 0xffffffffffffff00, 0xff00000000000000,
	0xffffffffff010101, 0x0202020202010101, 0xffffffffffff0201, 0xff01010101010101,
}
var expandAVX512_60_mat2 = [8]uint64{
	0x0404040404040404, 0x0404040408080808, 0x0808080808080808, 0x1010101010101010,
	0x1010101020202020, 0x2020202020202020, 0x4040404040404040, 0x4040404080808080,
}
var expandAVX512_60_inShuf2 = [8]uint64{
	0xff01010101010101, 0xffffffffffffff01, 0xff01010101010101, 0xff01010101010101,
	0xffffffffffffff01, 0xff01010101010101, 0xff01010101010101, 0xffffffffffffff01,
}
var expandAVX512_60_mat3 = [8]uint64{
	0x8080808080808080, 0x0101010101010101, 0x0000000000000000, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_60_inShuf3 = [8]uint64{
	0xff01010101010101, 0xffffffffffff0202, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
}
var expandAVX512_60_outShufLo = [8]uint64{
	0x0806050403020100, 0x1816151413121110, 0x28201e1d1c1b1a19, 0x31302e2d2c2b2a29,
	0x4140383635343332, 0x4a49484645444342, 0x5a5958504e4d4c4b, 0x626160075e5d5c5b,
}
var expandAVX512_60_outShufHi0 = [8]uint64{
	0x3b3a3938302a2928, 0x44434241403e3d3c, 0x5453525150484645, 0x5d5c5b5a59585655,
	0x6d6c6b6a6968605e, 0x767574737271706e, 0xffffffffffffff78, 0x31ffff2f2e2d2c2b,
}
var expandAVX512_60_outShufHi1 = [8]uint64{
	0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
	0xffffffffffffffff, 0xffffffffffffffff, 0x06050403020100ff, 0xff0908ffffffffff,
}

func expandAVX512_60(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_60_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_60_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_60_mat1).AsUint8x64()
	v6 := simd.LoadUint64x8(&expandAVX512_60_inShuf1).AsUint8x64()
	v9 := simd.LoadUint64x8(&expandAVX512_60_mat2).AsUint8x64()
	v10 := simd.LoadUint64x8(&expandAVX512_60_inShuf2).AsUint8x64()
	v13 := simd.LoadUint64x8(&expandAVX512_60_mat3).AsUint8x64()
	v14 := simd.LoadUint64x8(&expandAVX512_60_inShuf3).AsUint8x64()
	v17 := simd.LoadUint64x8(&expandAVX512_60_outShufLo).AsUint8x64()
	v19 := simd.LoadUint64x8(&expandAVX512_60_outShufHi0).AsUint8x64()
	v20 := simd.LoadUint64x8(&expandAVX512_60_outShufHi1).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v7 := v0.Permute(v6)
	v8 := v7.GaloisFieldAffineTransform(v5.AsUint64x8(), 0)
	v11 := v0.Permute(v10)
	v12 := v11.GaloisFieldAffineTransform(v9.AsUint64x8(), 0)
	v15 := v0.Permute(v14)
	v16 := v15.GaloisFieldAffineTransform(v13.AsUint64x8(), 0)
	v18 := v4.ConcatPermute(v8, v17)
	u0 := uint64(0x9f01ffffffffffff)
	m0 := simd.Mask8x64FromBits(u0)
	v21 := v8.ConcatPermute(v12, v19).Masked(m0)
	u1 := uint64(0x60fe000000000000)
	m1 := simd.Mask8x64FromBits(u1)
	v22 := v16.Permute(v20).Masked(m1)
	v23 := v21.Or(v22)
	return v18.AsUint64x8(), v23.AsUint64x8()
}

var expandAVX512_64_mat0 = [8]uint64{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}
var expandAVX512_64_inShuf0 = [8]uint64{
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
	0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
}
var expandAVX512_64_inShuf1 = [8]uint64{
	0x0101010101010101, 0x0101010101010101, 0x0101010101010101, 0x0101010101010101,
	0x0101010101010101, 0x0101010101010101, 0x0101010101010101, 0x0101010101010101,
}
var expandAVX512_64_outShufLo = [8]uint64{
	0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918,
	0x2726252423222120, 0x2f2e2d2c2b2a2928, 0x3736353433323130, 0x3f3e3d3c3b3a3938,
}

func expandAVX512_64(src unsafe.Pointer) (simd.Uint64x8, simd.Uint64x8) {
	v0 := simd.LoadUint64x8((*[8]uint64)(src)).AsUint8x64()
	v1 := simd.LoadUint64x8(&expandAVX512_64_mat0).AsUint8x64()
	v2 := simd.LoadUint64x8(&expandAVX512_64_inShuf0).AsUint8x64()
	v5 := simd.LoadUint64x8(&expandAVX512_64_inShuf1).AsUint8x64()
	v8 := simd.LoadUint64x8(&expandAVX512_64_outShufLo).AsUint8x64()
	v3 := v0.Permute(v2)
	v4 := v3.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v6 := v0.Permute(v5)
	v7 := v6.GaloisFieldAffineTransform(v1.AsUint64x8(), 0)
	v9 := v4.Permute(v8)
	v10 := v7.Permute(v8)
	return v9.AsUint64x8(), v10.AsUint64x8()
}
