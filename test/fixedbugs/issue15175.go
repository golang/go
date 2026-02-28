// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure unsigned shift results get sign-extended correctly.
package main

import "fmt"

func main() {
	failed := false
	a6 := uint8(253)
	if got := a6 >> 0; got != 253 {
		fmt.Printf("uint8(253)>>0 = %v, wanted 253\n", got)
		failed = true
	}
	if got := f1(0, 2, 1, 0, 0, 1, true); got != 255 {
		fmt.Printf("f1(...) = %v, wanted 255\n", got)
		failed = true
	}
	if got := f2(1); got != 242 {
		fmt.Printf("f2(...) = %v, wanted 242\n", got)
		failed = true
	}
	if got := f3(false, 0, 0); got != 254 {
		fmt.Printf("f3(...) = %v, wanted 254\n", got)
		failed = true
	}
	if failed {
		panic("bad")
	}
}

func f1(a1 uint, a2 int8, a3 int8, a4 int8, a5 uint8, a6 int, a7 bool) uint8 {
	a5--
	a4 += (a2 << a1 << 2) | (a4 ^ a4<<(a1&a1)) - a3                              // int8
	a6 -= a6 >> (2 + uint32(a2)>>3)                                              // int
	a1 += a1                                                                     // uint
	a3 *= a4 << (a1 | a1) << (uint16(3) >> 2 & (1 - 0) & (uint16(1) << a5 << 3)) // int8
	a7 = a7 || ((a2 == a4) || (a7 && a7) || ((a5 == a5) || (a7 || a7)))          // bool
	return a5 >> a1
}

func f2(a1 uint8) uint8 {
	a1--
	a1--
	a1 -= a1 + (a1 << 1) - (a1*a1*a1)<<(2-0+(3|3)-1)                // uint8
	v1 := 0 * ((2 * 1) ^ 1) & ((uint(0) >> a1) + (2+0)*(uint(2)+0)) // uint
	_ = v1
	return a1 >> (((2 ^ 2) >> (v1 | 2)) + 0)
}

func f3(a1 bool, a2 uint, a3 int64) uint8 {
	a3--
	v1 := 1 & (2 & 1 * (1 ^ 2) & (uint8(3*1) >> 0)) // uint8
	_ = v1
	v1 += v1 - (v1 >> a2) + (v1 << (a2 ^ a2) & v1) // uint8
	v1 *= v1                                       // uint8
	a3--
	v1 += v1 & v1 // uint8
	v1--
	v1 = ((v1 << 0) | v1>>0) + v1 // uint8
	return v1 >> 0
}
