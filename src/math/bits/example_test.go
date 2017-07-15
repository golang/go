// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bits_test

import (
	"fmt"
	"math/bits"
)

func ExampleLeadingZeros16() {
	fmt.Println(bits.LeadingZeros16(0))
	fmt.Println(bits.LeadingZeros16(1))
	fmt.Println(bits.LeadingZeros16(256))
	fmt.Println(bits.LeadingZeros16(65535))
	// Output:
	// 16
	// 15
	// 7
	// 0
}

func ExampleLeadingZeros32() {
	fmt.Println(bits.LeadingZeros32(0))
	fmt.Println(bits.LeadingZeros32(1))
	// Output:
	// 32
	// 31
}

func ExampleLeadingZeros64() {
	fmt.Println(bits.LeadingZeros64(0))
	fmt.Println(bits.LeadingZeros64(1))
	// Output:
	// 64
	// 63
}
