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

// The following functions test that indexed load/store operations get generated.

func idxInt8(x, y []int8, i int) {
	var t int8
	// amd64: `MOVBL[SZ]X\t1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVB\t[A-Z]+[0-9]*, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	y[i+1] = t
	// amd64: `MOVB\t[$]77, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	x[i+1] = 77
}

func idxInt16(x, y []int16, i int) {
	var t int16
	// amd64: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	y[i+1] = t
	// amd64: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\), [A-Z]+[0-9]*`
	t = x[16*i+1]
	// amd64: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	y[16*i+1] = t
	// amd64: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	x[i+1] = 77
	// amd64: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	x[16*i+1] = 77
}

func idxInt32(x, y []int32, i int) {
	var t int32
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	y[i+1] = t
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	t = x[2*i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	y[2*i+1] = t
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), [A-Z]+[0-9]*`
	t = x[16*i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	y[16*i+1] = t
	// amd64: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+1] = 77
	// amd64: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	x[16*i+1] = 77
}

func idxInt64(x, y []int64, i int) {
	var t int64
	// amd64: `MOVQ\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVQ\t[A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	y[i+1] = t
	// amd64: `MOVQ\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), [A-Z]+[0-9]*`
	t = x[16*i+1]
	// amd64: `MOVQ\t[A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\)`
	y[16*i+1] = t
	// amd64: `MOVQ\t[$]77, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+1] = 77
	// amd64: `MOVQ\t[$]77, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\)`
	x[16*i+1] = 77
}

func idxFloat32(x, y []float32, i int) {
	var t float32
	// amd64: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	t = x[i+1]
	// amd64: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	y[i+1] = t
	// amd64: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), X[0-9]+`
	t = x[16*i+1]
	// amd64: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	y[16*i+1] = t
}

func idxFloat64(x, y []float64, i int) {
	var t float64
	// amd64: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	t = x[i+1]
	// amd64: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	y[i+1] = t
	// amd64: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), X[0-9]+`
	t = x[16*i+1]
	// amd64: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\)`
	y[16*i+1] = t
}
