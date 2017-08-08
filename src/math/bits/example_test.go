// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bits_test

import (
	"fmt"
	"math/bits"
)

func ExampleLeadingZeros8() {
	fmt.Printf("LeadingZeros8(%08b) = %d\n", 1, bits.LeadingZeros8(1))
	// Output:
	// LeadingZeros8(00000001) = 7
}

func ExampleLeadingZeros16() {
	fmt.Printf("LeadingZeros16(%016b) = %d\n", 1, bits.LeadingZeros16(1))
	// Output:
	// LeadingZeros16(0000000000000001) = 15
}

func ExampleLeadingZeros32() {
	fmt.Printf("LeadingZeros32(%032b) = %d\n", 1, bits.LeadingZeros32(1))
	// Output:
	// LeadingZeros32(00000000000000000000000000000001) = 31
}

func ExampleLeadingZeros64() {
	fmt.Printf("LeadingZeros64(%064b) = %d\n", 1, bits.LeadingZeros64(1))
	// Output:
	// LeadingZeros64(0000000000000000000000000000000000000000000000000000000000000001) = 63
}

func ExampleOnesCount8() {
	fmt.Printf("OnesCount8(%08b) = %d\n", 14, bits.OnesCount8(14))
	// Output:
	// OnesCount8(00001110) = 3
}

func ExampleOnesCount16() {
	fmt.Printf("OnesCount16(%016b) = %d\n", 14, bits.OnesCount16(14))
	// Output:
	// OnesCount16(0000000000001110) = 3
}

func ExampleOnesCount32() {
	fmt.Printf("OnesCount32(%032b) = %d\n", 14, bits.OnesCount32(14))
	// Output:
	// OnesCount32(00000000000000000000000000001110) = 3
}

func ExampleOnesCount64() {
	fmt.Printf("OnesCount64(%064b) = %d\n", 14, bits.OnesCount64(14))
	// Output:
	// OnesCount64(0000000000000000000000000000000000000000000000000000000000001110) = 3
}

func ExampleTrailingZeros8() {
	fmt.Printf("TrailingZeros8(%08b) = %d\n", 8, bits.TrailingZeros8(8))
	// Output:
	// TrailingZeros8(00001000) = 3
}

func ExampleTrailingZeros16() {
	fmt.Printf("TrailingZeros16(%016b) = %d\n", 8, bits.TrailingZeros16(8))
	// Output:
	// TrailingZeros16(0000000000001000) = 3
}

func ExampleTrailingZeros32() {
	fmt.Printf("TrailingZeros32(%032b) = %d\n", 8, bits.TrailingZeros32(8))
	// Output:
	// TrailingZeros32(00000000000000000000000000001000) = 3
}

func ExampleTrailingZeros64() {
	fmt.Printf("TrailingZeros64(%064b) = %d\n", 8, bits.TrailingZeros64(8))
	// Output:
	// TrailingZeros64(0000000000000000000000000000000000000000000000000000000000001000) = 3
}

func ExampleLen8() {
	fmt.Printf("Len8(%08b) = %d\n", 8, bits.Len8(8))
	// Output:
	// Len8(00001000) = 4
}

func ExampleLen16() {
	fmt.Printf("Len16(%016b) = %d\n", 8, bits.Len16(8))
	// Output:
	// Len16(0000000000001000) = 4
}

func ExampleLen32() {
	fmt.Printf("Len32(%032b) = %d\n", 8, bits.Len32(8))
	// Output:
	// Len32(00000000000000000000000000001000) = 4
}

func ExampleLen64() {
	fmt.Printf("Len64(%064b) = %d\n", 8, bits.Len64(8))
	// Output:
	// Len64(0000000000000000000000000000000000000000000000000000000000001000) = 4
}

func ExampleReverse16() {
	fmt.Printf("%016b\n", 19)
	fmt.Printf("%016b\n", bits.Reverse16(19))
	// Output:
	// 0000000000010011
	// 1100100000000000
}

func ExampleReverse32() {
	fmt.Printf("%032b\n", 19)
	fmt.Printf("%032b\n", bits.Reverse32(19))
	// Output:
	// 00000000000000000000000000010011
	// 11001000000000000000000000000000
}

func ExampleReverse64() {
	fmt.Printf("%064b\n", 19)
	fmt.Printf("%064b\n", bits.Reverse64(19))
	// Output:
	// 0000000000000000000000000000000000000000000000000000000000010011
	// 1100100000000000000000000000000000000000000000000000000000000000
}

func ExampleReverse8() {
	fmt.Printf("%008b\n", 19)
	fmt.Printf("%008b\n", bits.Reverse8(19))
	// Output:
	// 00010011
	// 11001000
}
