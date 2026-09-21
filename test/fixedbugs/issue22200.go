// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1(x *[1<<30 - 1e6]byte) byte {
	sum := byte(0)
	for _, b := range *x {
		sum += b
	}
	return sum
}
func f2(x *[1<<30 + 1e6]byte) byte { // GC_ERROR "stack frame too large"
	sum := byte(0)
	for _, b := range *x {
		sum += b
	}
	return sum
}
