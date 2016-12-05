// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	T "runtime/internal/sys"
)

var A = []uint64{0x0102030405060708, 0x1122334455667788}
var B = []uint64{0x0807060504030201, 0x8877665544332211}

var errors int

func logf(f string, args ...interface{}) {
	errors++
	fmt.Printf(f, args...)
	if errors > 100 { // 100 is enough spewage
		panic("100 errors is plenty is enough")
	}
}

func test(i, x uint64) {
	t := T.Ctz64(x) // ERROR "intrinsic substitution for Ctz64"
	if i != t {
		logf("Ctz64(0x%x) expected %d but got %d\n", x, i, t)
	}
	x = -x
	t = T.Ctz64(x) // ERROR "intrinsic substitution for Ctz64"
	if i != t {
		logf("Ctz64(0x%x) expected %d but got %d\n", x, i, t)
	}

	if i <= 32 {
		x32 := uint32(x)
		t32 := T.Ctz32(x32) // ERROR "intrinsic substitution for Ctz32"
		if uint32(i) != t32 {
			logf("Ctz32(0x%x) expected %d but got %d\n", x32, i, t32)
		}
		x32 = -x32
		t32 = T.Ctz32(x32) // ERROR "intrinsic substitution for Ctz32"
		if uint32(i) != t32 {
			logf("Ctz32(0x%x) expected %d but got %d\n", x32, i, t32)
		}
	}
}

func main() {
	// Test Bswap first because the other test relies on it
	// working correctly (to implement bit reversal).
	for i := range A {
		x := A[i]
		y := B[i]
		X := T.Bswap64(x) // ERROR "intrinsic substitution for Bswap64"
		Y := T.Bswap64(y) // ERROR "intrinsic substitution for Bswap64"
		if y != X {
			logf("Bswap64(0x%08x) expected 0x%08x but got 0x%08x\n", x, y, X)
		}
		if x != Y {
			logf("Bswap64(0x%08x) expected 0x%08x but got 0x%08x\n", y, x, Y)
		}

		x32 := uint32(X)
		y32 := uint32(Y >> 32)

		X32 := T.Bswap32(x32) // ERROR "intrinsic substitution for Bswap32"
		Y32 := T.Bswap32(y32) // ERROR "intrinsic substitution for Bswap32"
		if y32 != X32 {
			logf("Bswap32(0x%08x) expected 0x%08x but got 0x%08x\n", x32, y32, X32)
		}
		if x32 != Y32 {
			logf("Bswap32(0x%08x) expected 0x%08x but got 0x%08x\n", y32, x32, Y32)
		}
	}

	// Zero is a special case, be sure it is done right.
	if T.Ctz32(0) != 32 { // ERROR "intrinsic substitution for Ctz32"
		logf("ctz32(0) != 32")
	}
	if T.Ctz64(0) != 64 { // ERROR "intrinsic substitution for Ctz64"
		logf("ctz64(0) != 64")
	}

	for i := uint64(0); i <= 64; i++ {
		for j := uint64(1); j <= 255; j += 2 {
			for k := uint64(1); k <= 65537; k += 128 {
				x := (j * k) << i
				test(i, x)
			}
		}
	}
}
