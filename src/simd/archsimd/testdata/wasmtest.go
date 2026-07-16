// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package main

import (
	"fmt"
	"simd/archsimd"
)

//go:noinline
func id[T any](x T) T {
	return x
}

func main() {
	var x, _ = archsimd.LoadInt16x8Part([]int16{10, 21, 32, 43, 54, 65, 76})
	var y, _ = archsimd.LoadInt16x8Part([]int16{200, 300, 400, 500, 600, 700, 800})
	z := x.Add(y)
	fmt.Println("z =", z.String())
	m1 := z.Equal(z)
	fmt.Println("m1 =", m1.ToInt16x8().String())
	fmt.Println("m1.String() = ", m1.String())
	i := m1.ToInt16x8().Neg()
	fmt.Println("i =", i.String())
	i0 := i.ShiftAllLeft(2)
	fmt.Println("i0 =", i0.String())
	i1 := i.ShiftAllLeft(id(uint64(2)))
	fmt.Println("i1 =", i1.String())
	i2 := i.ShiftAllLeft(id(uint64(15)))
	fmt.Println("i2 =", i2.String())
	i3 := i2.ShiftAllRight(id(uint64(16)))
	fmt.Println("i3 =", i3.String())
	m2 := z.And(i0).Equal(i1)
	fmt.Println("m2 =", m2.ToInt16x8().String())
	zz := z.IfElse(m2, y)
	fmt.Printf("zz = %v\n", zz)
	zl := zz.ExtendLo4ToInt32()
	fmt.Printf("zl-extended = %v\n", zl)
	zh := zz.ExtendHi4ToInt32()
	fmt.Printf("zh-extended = %v\n", zh)

	w := make([]int16, 7, 7)
	zz.StorePart(w)
	fmt.Printf("w = %v\n", w)

	fmt.Println()

	for j := uint64(0); j < 18; j++ {
		fmt.Printf("%v.RAL(%d)=%v\n", i, j, i.RotateAllLeft(j))
	}

	fmt.Println()

	for j := uint64(0); j < 18; j++ {
		fmt.Printf("%v.RAr(%d)=%v\n", i, j, i.RotateAllRight(j))
	}

	{
		var a = archsimd.LoadInt8x16([]int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		// var b = archsimd.LoadInt8x16([]int8{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 32})
		var i = archsimd.LoadInt8x16([]int8{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30})
		fmt.Println(a.LookupOrZero(i))
		// fmt.Println(a.Shuffle(i, b))
	}

}
