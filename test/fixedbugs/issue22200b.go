// errorcheck

//go:build !386 && !amd64p32 && !arm && !mips && !mipsle

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f3(x *[1 << 31]byte) byte { // GC_ERROR "stack frame too large"
	sum := byte(0)
	for _, b := range *x {
		sum += b
	}
	return sum
}
func f4(x *[1 << 32]byte) byte { // GC_ERROR "stack frame too large"
	sum := byte(0)
	for _, b := range *x {
		sum += b
	}
	return sum
}
func f5(x *[1 << 33]byte) byte { // GC_ERROR "stack frame too large"
	sum := byte(0)
	for _, b := range *x {
		sum += b
	}
	return sum
}
