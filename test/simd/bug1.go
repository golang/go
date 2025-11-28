// compile

//go:build amd64 && goexperiment.simd

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for ICE on picking the wrong type for the spill slot.

package p

import (
	"simd"
	"unsafe"
)

func F(
	dst *[2][4][4]float32,
	tos *[2][4][4]float32,
	blend int,
) {
	tiny := simd.BroadcastFloat32x8(0)
	for {
		dstCol12 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[0][0:]))))
		dstCol34 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[0][2:]))))
		dstCol56 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[1][0:]))))
		dstCol78 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[1][2:]))))

		tosCol12 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(tos[0][0:]))))
		tosCol34 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(tos[0][2:]))))
		tosCol56 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(tos[1][0:]))))
		tosCol78 := simd.LoadFloat32x8((*[8]float32)(unsafe.Pointer((*[2][4]float32)(tos[1][2:]))))

		var Cr0, Cr1, Cr2 simd.Float32x8
		if blend != 0 {
			invas := tosCol78.Max(tiny)
			invad := dstCol78.Max(tiny)
			Cd0 := dstCol12.Mul(invad)
			Cd1 := dstCol34.Mul(invad)
			Cd2 := dstCol56.Mul(invad)
			Cs0 := tosCol12.Mul(invas)
			Cs1 := tosCol34.Mul(invas)
			Cs2 := tosCol56.Mul(invas)
			var Cm0, Cm1, Cm2 simd.Float32x8
			switch blend {
			case 4:
			case 10:
			case 11:
			case 8:
			case 5:
			case 1:
			case 0:
				Cm1 = Cs1
			case 2:
				Cm0 = Cd0.Add(Cs0)
				Cm1 = Cd1.Add(Cs1)
				Cm2 = Cd2.Add(Cs2)
			}
			Cr0 = dstCol78.Mul(Cs0).Mul(Cm0)
			Cr1 = dstCol78.Mul(Cs1).Mul(Cm1)
			Cr2 = dstCol78.Mul(Cs2).Mul(Cm2)
		}
		var resR, resG, resB, resA simd.Float32x8
		if blend == 0 {
			resR = tosCol12
			resG = tosCol34
			resB = tosCol56
			resA = tosCol78
		} else {
			resR = Cr0.Add(dstCol12)
			resG = Cr1.Add(dstCol34)
			resB = Cr2.Add(dstCol56)
		}

		resR.Store((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[0][0:2]))))
		resG.Store((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[0][2:4]))))
		resB.Store((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[1][0:2]))))
		resA.Store((*[8]float32)(unsafe.Pointer((*[2][4]float32)(dst[1][2:4]))))
	}
}
