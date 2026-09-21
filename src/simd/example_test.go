// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"fmt"
	"simd"
)

func ExampleInt8s_Add() {
	// Initialize slice of 64 int8s (max vector size under AVX512).
	in1 := []int8{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	in2 := []int8{
		10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 100, 10, 10, 10, 16,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}

	// Load slices into vectors.
	v1 := simd.LoadInt8s(in1)
	v2 := simd.LoadInt8s(in2)

	// Add the vectors.
	sum := v1.Add(v2)

	// Store the result back to a slice.
	out := make([]int8, sum.Len())
	sum.Store(out)

	// Print the first 16 elements (minimum vector width across architectures).
	fmt.Println(out[:16])
	// Output: [11 22 33 44 55 66 77 88 99 110 121 112 23 24 25 32]
}

func ExampleInt8s_Masked() {
	// Load vectors of 64 elements.
	v1 := simd.LoadInt8s([]int8{
		1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	})
	var v2 simd.Int8s // zero value

	// Create a mask where elements in v1 are greater than zero.
	mask := v1.Greater(v2)

	// Keep elements of v1 where the mask is true, zero out elsewhere.
	res := v1.Masked(mask)

	out := make([]int8, res.Len())
	res.Store(out)

	// Print the first 16 elements.
	fmt.Println(out[:16])
	// Output: [1 0 3 0 5 0 7 0 9 0 11 0 13 0 15 0]
}

func ExampleLoadInt8sPart() {
	// Slice smaller than the full vector length.
	s := []int8{1, 2, 3, 4, 5}

	// Load partial slice.
	v, n := simd.LoadInt8sPart(s)
	fmt.Printf("Loaded %d elements\n", n)

	// Store only the loaded elements.
	out := make([]int8, n)
	v.StorePart(out)
	fmt.Println(out)

	// Output:
	// Loaded 5 elements
	// [1 2 3 4 5]
}

func ExampleFloat32s_MulAdd() {
	// Float32s on 512-bit vector has 16 elements.
	v1 := simd.LoadFloat32s([]float32{
		1.5, 2.5, 3.5, 4.5,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	})
	v2 := simd.LoadFloat32s([]float32{
		2.0, 2.0, 2.0, 2.0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	})
	v3 := simd.LoadFloat32s([]float32{
		1.0, 2.0, 3.0, 4.0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	})

	// Perform element-wise v1 * v2 + v3.
	res := v1.MulAdd(v2, v3)

	out := make([]float32, res.Len())
	res.Store(out)

	// Print the first 4 elements.
	fmt.Println(out[:4])
	// Output: [4 7 10 13]
}

func ExampleInt16s_ShiftAllLeft() {
	// Int16s on 512-bit vector has 32 elements.
	in := []int16{
		1, 2, 4, 8, 16, 32, 64, 128,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}
	v := simd.LoadInt16s(in)

	// Shift all elements left by 2 bits.
	res := v.ShiftAllLeft(2)

	out := make([]int16, res.Len())
	res.Store(out)

	// Print the first 8 elements.
	fmt.Println(out[:8])
	// Output: [4 8 16 32 64 128 256 512]
}

func ExampleInt16s_RotateAllLeft() {
	// Int16s on 512-bit vector has 32 elements.
	in := []int16{
		0x00f0, 0x1234, 0, 0, 0, 0, 0, 0x7000,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}
	v := simd.LoadInt16s(in)

	// Rotate all elements left by 4 bits.
	res := v.RotateAllLeft(4)

	out := make([]int16, res.Len())
	res.Store(out)

	fmt.Printf("%#04x\n", out[:8])
	// Output: [0x0f00 0x2341 0x0000 0x0000 0x0000 0x0000 0x0000 0x0007]
}
