// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.13

package poly1305

import "math/bits"

func bitsAdd64(x, y, carry uint64) (sum, carryOut uint64) {
	return bits.Add64(x, y, carry)
}

func bitsSub64(x, y, borrow uint64) (diff, borrowOut uint64) {
	return bits.Sub64(x, y, borrow)
}

func bitsMul64(x, y uint64) (hi, lo uint64) {
	return bits.Mul64(x, y)
}
