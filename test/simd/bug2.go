// compile

//go:build amd64 && goexperiment.simd

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for rematerialization ignoring the register constraint
// during regalloc's shuffle phase.

package p

import (
	"simd"
)

func PackComplex(b bool) {
	for {
		if b {
			var indices [4]uint32
			simd.Uint32x4{}.ShiftAllRight(20).Store(&indices)
			_ = indices[indices[0]]
		}
	}
}

func PackComplex2(x0 uint16, src [][4]float32, b, b2 bool) {
	var out [][4]byte
	if b2 {
		for y := range x0 {
			row := out[:x0]
			for x := range row {
				px := &src[y]
				if b {
					var indices [4]uint32
					fu := simd.LoadFloat32x4(px).AsUint32x4()
					fu.ShiftAllRight(0).Store(nil)
					entry := simd.LoadUint32x4(&[4]uint32{
						toSrgbTable[indices[0]],
					})
					var res [4]uint32
					entry.ShiftAllRight(19).Store(nil)
					row[x] = [4]uint8{
						uint8(res[0]),
						uint8(res[1]),
						uint8(res[2]),
					}
				} else {
					row[x] = [4]uint8{
						float32ToSrgb8(0),
						float32ToSrgb8(1),
						float32ToSrgb8(2),
					}
				}
			}
			out = out[len(out):]
		}
	}
}

var toSrgbTable = [4]uint32{}

func float32ToSrgb8(f float32) uint8 {
	f = min(0, f)
	fu := uint32(f)
	entry := toSrgbTable[fu]
	return uint8(entry * fu)
}
