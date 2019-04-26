// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

var x [2]bool
var x8 [2]uint8
var x16 [2]uint16
var x32 [2]uint32
var x64 [2]uint64

func compMem1() int {
	// amd64:`CMPB\t"".x\+1\(SB\), [$]0`
	if x[1] {
		return 1
	}
	// amd64:`CMPB\t"".x8\+1\(SB\), [$]7`
	if x8[1] == 7 {
		return 1
	}
	// amd64:`CMPW\t"".x16\+2\(SB\), [$]7`
	if x16[1] == 7 {
		return 1
	}
	// amd64:`CMPL\t"".x32\+4\(SB\), [$]7`
	if x32[1] == 7 {
		return 1
	}
	// amd64:`CMPQ\t"".x64\+8\(SB\), [$]7`
	if x64[1] == 7 {
		return 1
	}
	return 0
}

//go:noinline
func f(x int) bool {
	return false
}

//go:noinline
func f8(x int) int8 {
	return 0
}

//go:noinline
func f16(x int) int16 {
	return 0
}

//go:noinline
func f32(x int) int32 {
	return 0
}

//go:noinline
func f64(x int) int64 {
	return 0
}

func compMem2() int {
	// amd64:`CMPB\t8\(SP\), [$]0`
	if f(3) {
		return 1
	}
	// amd64:`CMPB\t8\(SP\), [$]7`
	if f8(3) == 7 {
		return 1
	}
	// amd64:`CMPW\t8\(SP\), [$]7`
	if f16(3) == 7 {
		return 1
	}
	// amd64:`CMPL\t8\(SP\), [$]7`
	if f32(3) == 7 {
		return 1
	}
	// amd64:`CMPQ\t8\(SP\), [$]7`
	if f64(3) == 7 {
		return 1
	}
	return 0
}

func compMem3(x, y *int) (int, bool) {
	// We can do comparisons of a register with memory even if
	// the register is used subsequently.
	r := *x
	// amd64:`CMPQ\t\(`
	// 386:`CMPL\t\(`
	return r, r < *y
}
