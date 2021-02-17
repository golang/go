// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !386,!amd64p32,!arm,!mips,!mipsle

package p

func f3(x *[1 << 31]byte) byte { // GC_ERROR "stack frame too large"
	for _, b := range *x {
		return b
	}
	return 0
}
func f4(x *[1 << 32]byte) byte { // GC_ERROR "stack frame too large"
	for _, b := range *x {
		return b
	}
	return 0
}
func f5(x *[1 << 33]byte) byte { // GC_ERROR "stack frame too large"
	for _, b := range *x {
		return b
	}
	return 0
}
