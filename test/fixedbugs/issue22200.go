// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1(x *[1<<30 - 1e6]byte) byte {
	for _, b := range *x {
		return b
	}
	return 0
}
func f2(x *[1<<30 + 1e6]byte) byte { // GC_ERROR "stack frame too large"
	for _, b := range *x {
		return b
	}
	return 0
}
