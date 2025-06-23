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
	// amd64:`CMPB\tcommand-line-arguments.x\+1\(SB\), [$]0`
	if x[1] {
		return 1
	}
	// amd64:`CMPB\tcommand-line-arguments.x8\+1\(SB\), [$]7`
	if x8[1] == 7 {
		return 1
	}
	// amd64:`CMPW\tcommand-line-arguments.x16\+2\(SB\), [$]7`
	if x16[1] == 7 {
		return 1
	}
	// amd64:`CMPL\tcommand-line-arguments.x32\+4\(SB\), [$]7`
	if x32[1] == 7 {
		return 1
	}
	// amd64:`CMPQ\tcommand-line-arguments.x64\+8\(SB\), [$]7`
	if x64[1] == 7 {
		return 1
	}
	return 0
}

type T struct {
	x   bool
	x8  uint8
	x16 uint16
	x32 uint32
	x64 uint64
	a   [2]int // force it passed in memory
}

func compMem2(t T) int {
	// amd64:`CMPB\t.*\(SP\), [$]0`
	if t.x {
		return 1
	}
	// amd64:`CMPB\t.*\(SP\), [$]7`
	if t.x8 == 7 {
		return 1
	}
	// amd64:`CMPW\t.*\(SP\), [$]7`
	if t.x16 == 7 {
		return 1
	}
	// amd64:`CMPL\t.*\(SP\), [$]7`
	if t.x32 == 7 {
		return 1
	}
	// amd64:`CMPQ\t.*\(SP\), [$]7`
	if t.x64 == 7 {
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
	//   386: `MOVBL[SZ]X\t1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVB\t[A-Z]+[0-9]*, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	//   386: `MOVB\t[A-Z]+[0-9]*, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	y[i+1] = t
	// amd64: `MOVB\t[$]77, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	//   386: `MOVB\t[$]77, 1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	x[i+1] = 77
}

func idxInt16(x, y []int16, i int) {
	var t int16
	// amd64: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\), [A-Z]+[0-9]*`
	//   386: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	//   386: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	y[i+1] = t
	// amd64: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\), [A-Z]+[0-9]*`
	//   386: `MOVWL[SZ]X\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\), [A-Z]+[0-9]*`
	t = x[16*i+1]
	// amd64: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	//   386: `MOVW\t[A-Z]+[0-9]*, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	y[16*i+1] = t
	// amd64: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	//   386: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\)`
	x[i+1] = 77
	// amd64: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	//   386: `MOVW\t[$]77, 2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\)`
	x[16*i+1] = 77
}

func idxInt32(x, y []int32, i int) {
	var t int32
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	//   386: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	t = x[i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	//   386: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	y[i+1] = t
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	t = x[2*i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	y[2*i+1] = t
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), [A-Z]+[0-9]*`
	//   386: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), [A-Z]+[0-9]*`
	t = x[16*i+1]
	// amd64: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	//   386: `MOVL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	y[16*i+1] = t
	// amd64: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	//   386: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+1] = 77
	// amd64: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	//   386: `MOVL\t[$]77, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
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
	//    amd64: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	// 386/sse2: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	//    arm64: `FMOVS\t\(R[0-9]*\)\(R[0-9]*<<2\), F[0-9]+`
	t = x[i+1]
	//    amd64: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	// 386/sse2: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	//    arm64: `FMOVS\tF[0-9]+, \(R[0-9]*\)\(R[0-9]*<<2\)`
	y[i+1] = t
	//    amd64: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), X[0-9]+`
	// 386/sse2: `MOVSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), X[0-9]+`
	t = x[16*i+1]
	//    amd64: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	// 386/sse2: `MOVSS\tX[0-9]+, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\)`
	y[16*i+1] = t
}

func idxFloat64(x, y []float64, i int) {
	var t float64
	//    amd64: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	// 386/sse2: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	//    arm64: `FMOVD\t\(R[0-9]*\)\(R[0-9]*<<3\), F[0-9]+`
	t = x[i+1]
	//    amd64: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	// 386/sse2: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	//    arm64: `FMOVD\tF[0-9]+, \(R[0-9]*\)\(R[0-9]*<<3\)`
	y[i+1] = t
	//    amd64: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), X[0-9]+`
	// 386/sse2: `MOVSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), X[0-9]+`
	t = x[16*i+1]
	//    amd64: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\)`
	// 386/sse2: `MOVSD\tX[0-9]+, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\)`
	y[16*i+1] = t
}

func idxLoadPlusOp32(x []int32, i int) int32 {
	s := x[0]
	// 386: `ADDL\t4\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	// amd64: `ADDL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s += x[i+1]
	// 386: `SUBL\t8\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	// amd64: `SUBL\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s -= x[i+2]
	// 386: `IMULL\t12\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	s *= x[i+3]
	// 386: `ANDL\t16\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	// amd64: `ANDL\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s &= x[i+4]
	// 386: `ORL\t20\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	// amd64: `ORL\t20\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s |= x[i+5]
	// 386: `XORL\t24\([A-Z]+\)\([A-Z]+\*4\), [A-Z]+`
	// amd64: `XORL\t24\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s ^= x[i+6]
	return s
}

func idxLoadPlusOp64(x []int64, i int) int64 {
	s := x[0]
	// amd64: `ADDQ\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s += x[i+1]
	// amd64: `SUBQ\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s -= x[i+2]
	// amd64: `ANDQ\t24\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s &= x[i+3]
	// amd64: `ORQ\t32\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s |= x[i+4]
	// amd64: `XORQ\t40\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s ^= x[i+5]
	return s
}

func idxStorePlusOp32(x []int32, i int, v int32) {
	// 386: `ADDL\t[A-Z]+, 4\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ADDL\t[A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+1] += v
	// 386: `SUBL\t[A-Z]+, 8\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `SUBL\t[A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+2] -= v
	// 386: `ANDL\t[A-Z]+, 12\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ANDL\t[A-Z]+[0-9]*, 12\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+3] &= v
	// 386: `ORL\t[A-Z]+, 16\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ORL\t[A-Z]+[0-9]*, 16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+4] |= v
	// 386: `XORL\t[A-Z]+, 20\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `XORL\t[A-Z]+[0-9]*, 20\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+5] ^= v

	// 386: `ADDL\t[$]77, 24\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ADDL\t[$]77, 24\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+6] += 77
	// 386: `ANDL\t[$]77, 28\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ANDL\t[$]77, 28\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+7] &= 77
	// 386: `ORL\t[$]77, 32\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `ORL\t[$]77, 32\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+8] |= 77
	// 386: `XORL\t[$]77, 36\([A-Z]+\)\([A-Z]+\*4\)`
	// amd64: `XORL\t[$]77, 36\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\)`
	x[i+9] ^= 77
}

func idxStorePlusOp64(x []int64, i int, v int64) {
	// amd64: `ADDQ\t[A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+1] += v
	// amd64: `SUBQ\t[A-Z]+[0-9]*, 16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+2] -= v
	// amd64: `ANDQ\t[A-Z]+[0-9]*, 24\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+3] &= v
	// amd64: `ORQ\t[A-Z]+[0-9]*, 32\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+4] |= v
	// amd64: `XORQ\t[A-Z]+[0-9]*, 40\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+5] ^= v

	// amd64: `ADDQ\t[$]77, 48\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+6] += 77
	// amd64: `ANDQ\t[$]77, 56\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+7] &= 77
	// amd64: `ORQ\t[$]77, 64\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+8] |= 77
	// amd64: `XORQ\t[$]77, 72\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\)`
	x[i+9] ^= 77
}

func idxCompare(i int) int {
	// amd64: `MOVBLZX\t1\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	if x8[i+1] < x8[0] {
		return 0
	}
	// amd64: `MOVWLZX\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\), [A-Z]+[0-9]*`
	if x16[i+1] < x16[0] {
		return 0
	}
	// amd64: `MOVWLZX\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\), [A-Z]+[0-9]*`
	if x16[16*i+1] < x16[0] {
		return 0
	}
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	if x32[i+1] < x32[0] {
		return 0
	}
	// amd64: `MOVL\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), [A-Z]+[0-9]*`
	if x32[16*i+1] < x32[0] {
		return 0
	}
	// amd64: `MOVQ\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	if x64[i+1] < x64[0] {
		return 0
	}
	// amd64: `MOVQ\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), [A-Z]+[0-9]*`
	if x64[16*i+1] < x64[0] {
		return 0
	}
	// amd64: `MOVBLZX\t2\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	if x8[i+2] < 77 {
		return 0
	}
	// amd64: `MOVWLZX\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*2\), [A-Z]+[0-9]*`
	if x16[i+2] < 77 {
		return 0
	}
	// amd64: `MOVWLZX\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[12]\), [A-Z]+[0-9]*`
	if x16[16*i+2] < 77 {
		return 0
	}
	// amd64: `MOVL\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	if x32[i+2] < 77 {
		return 0
	}
	// amd64: `MOVL\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[14]\), [A-Z]+[0-9]*`
	if x32[16*i+2] < 77 {
		return 0
	}
	// amd64: `MOVQ\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	if x64[i+2] < 77 {
		return 0
	}
	// amd64: `MOVQ\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*[18]\), [A-Z]+[0-9]*`
	if x64[16*i+2] < 77 {
		return 0
	}
	return 1
}

func idxFloatOps(a []float64, b []float32, i int) (float64, float32) {
	c := float64(7)
	// amd64: `ADDSD\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	c += a[i+1]
	// amd64: `SUBSD\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	c -= a[i+2]
	// amd64: `MULSD\t24\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	c *= a[i+3]
	// amd64: `DIVSD\t32\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), X[0-9]+`
	c /= a[i+4]

	d := float32(8)
	// amd64: `ADDSS\t4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	d += b[i+1]
	// amd64: `SUBSS\t8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	d -= b[i+2]
	// amd64: `MULSS\t12\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	d *= b[i+3]
	// amd64: `DIVSS\t16\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), X[0-9]+`
	d /= b[i+4]
	return c, d
}

func storeTest(a []bool, v int, i int) {
	// amd64: `BTL\t\$0,`,`SETCS\t4\([A-Z]+[0-9]*\)`
	a[4] = v&1 != 0
	// amd64: `BTL\t\$1,`,`SETCS\t3\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	a[3+i] = v&2 != 0
}

func bitOps(p *[12]uint64) {
	// amd64: `ORQ\t\$8, \(AX\)`
	p[0] |= 8
	// amd64: `ORQ\t\$1073741824, 8\(AX\)`
	p[1] |= 1 << 30
	// amd64: `BTSQ\t\$31, 16\(AX\)`
	p[2] |= 1 << 31
	// amd64: `BTSQ\t\$63, 24\(AX\)`
	p[3] |= 1 << 63

	// amd64: `ANDQ\t\$-9, 32\(AX\)`
	p[4] &^= 8
	// amd64: `ANDQ\t\$-1073741825, 40\(AX\)`
	p[5] &^= 1 << 30
	// amd64: `BTRQ\t\$31, 48\(AX\)`
	p[6] &^= 1 << 31
	// amd64: `BTRQ\t\$63, 56\(AX\)`
	p[7] &^= 1 << 63

	// amd64: `XORQ\t\$8, 64\(AX\)`
	p[8] ^= 8
	// amd64: `XORQ\t\$1073741824, 72\(AX\)`
	p[9] ^= 1 << 30
	// amd64: `BTCQ\t\$31, 80\(AX\)`
	p[10] ^= 1 << 31
	// amd64: `BTCQ\t\$63, 88\(AX\)`
	p[11] ^= 1 << 63
}
