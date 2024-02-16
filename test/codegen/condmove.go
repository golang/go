// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func cmovint(c int) int {
	x := c + 4
	if x < 0 {
		x = 182
	}
	// amd64:"CMOVQLT"
	// arm64:"CSEL\tLT"
	// ppc64x:"ISEL\t[$]0"
	// wasm:"Select"
	return x
}

func cmovchan(x, y chan int) chan int {
	if x != y {
		x = y
	}
	// amd64:"CMOVQNE"
	// arm64:"CSEL\tNE"
	// ppc64x:"ISEL\t[$]2"
	// wasm:"Select"
	return x
}

func cmovuintptr(x, y uintptr) uintptr {
	if x < y {
		x = -y
	}
	// amd64:"CMOVQ(HI|CS)"
	// arm64:"CSNEG\tLS"
	// ppc64x:"ISEL\t[$]1"
	// wasm:"Select"
	return x
}

func cmov32bit(x, y uint32) uint32 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVL(HI|CS)"
	// arm64:"CSNEG\t(LS|HS)"
	// ppc64x:"ISEL\t[$]1"
	// wasm:"Select"
	return x
}

func cmov16bit(x, y uint16) uint16 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVW(HI|CS)"
	// arm64:"CSNEG\t(LS|HS)"
	// ppc64x:"ISEL\t[$][01]"
	// wasm:"Select"
	return x
}

// Floating point comparison. For EQ/NE, we must
// generate special code to handle NaNs.
func cmovfloateq(x, y float64) int {
	a := 128
	if x == y {
		a = 256
	}
	// amd64:"CMOVQNE","CMOVQPC"
	// arm64:"CSEL\tEQ"
	// ppc64x:"ISEL\t[$]2"
	// wasm:"Select"
	return a
}

func cmovfloatne(x, y float64) int {
	a := 128
	if x != y {
		a = 256
	}
	// amd64:"CMOVQNE","CMOVQPS"
	// arm64:"CSEL\tNE"
	// ppc64x:"ISEL\t[$]2"
	// wasm:"Select"
	return a
}

//go:noinline
func frexp(f float64) (frac float64, exp int) {
	return 1.0, 4
}

//go:noinline
func ldexp(frac float64, exp int) float64 {
	return 1.0
}

// Generate a CMOV with a floating comparison and integer move.
func cmovfloatint2(x, y float64) float64 {
	yfr, yexp := 4.0, 5

	r := x
	for r >= y {
		rfr, rexp := frexp(r)
		if rfr < yfr {
			rexp = rexp - 1
		}
		// amd64:"CMOVQHI"
		// arm64:"CSEL\tMI"
		// ppc64x:"ISEL\t[$]0"
		// wasm:"Select"
		r = r - ldexp(y, rexp-yexp)
	}
	return r
}

func cmovloaded(x [4]int, y int) int {
	if x[2] != 0 {
		y = x[2]
	} else {
		y = y >> 2
	}
	// amd64:"CMOVQNE"
	// arm64:"CSEL\tNE"
	// ppc64x:"ISEL\t[$]2"
	// wasm:"Select"
	return y
}

func cmovuintptr2(x, y uintptr) uintptr {
	a := x * 2
	if a == 0 {
		a = 256
	}
	// amd64:"CMOVQEQ"
	// arm64:"CSEL\tEQ"
	// ppc64x:"ISEL\t[$]2"
	// wasm:"Select"
	return a
}

// Floating point CMOVs are not supported by amd64/arm64/ppc64x
func cmovfloatmove(x, y int) float64 {
	a := 1.0
	if x <= y {
		a = 2.0
	}
	// amd64:-"CMOV"
	// arm64:-"CSEL"
	// ppc64x:-"ISEL"
	// wasm:-"Select"
	return a
}

// On amd64, the following patterns trigger comparison inversion.
// Test that we correctly invert the CMOV condition
var gsink int64
var gusink uint64

func cmovinvert1(x, y int64) int64 {
	if x < gsink {
		y = -y
	}
	// amd64:"CMOVQGT"
	return y
}
func cmovinvert2(x, y int64) int64 {
	if x <= gsink {
		y = -y
	}
	// amd64:"CMOVQGE"
	return y
}
func cmovinvert3(x, y int64) int64 {
	if x == gsink {
		y = -y
	}
	// amd64:"CMOVQEQ"
	return y
}
func cmovinvert4(x, y int64) int64 {
	if x != gsink {
		y = -y
	}
	// amd64:"CMOVQNE"
	return y
}
func cmovinvert5(x, y uint64) uint64 {
	if x > gusink {
		y = -y
	}
	// amd64:"CMOVQCS"
	return y
}
func cmovinvert6(x, y uint64) uint64 {
	if x >= gusink {
		y = -y
	}
	// amd64:"CMOVQLS"
	return y
}

func cmovload(a []int, i int, b bool) int {
	if b {
		i++
	}
	// See issue 26306
	// amd64:-"CMOVQNE"
	return a[i]
}

func cmovstore(a []int, i int, b bool) {
	if b {
		i++
	}
	// amd64:"CMOVQNE"
	a[i] = 7
}

var r0, r1, r2, r3, r4, r5 int

func cmovinc(cond bool, a, b, c int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = b + 1
	}
	// arm64:"CSINC\tNE", -"CSEL"
	r0 = x0

	if cond {
		x1 = b + 1
	} else {
		x1 = a
	}
	// arm64:"CSINC\tEQ", -"CSEL"
	r1 = x1

	if cond {
		c++
	}
	// arm64:"CSINC\tEQ", -"CSEL"
	r2 = c
}

func cmovinv(cond bool, a, b int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = ^b
	}
	// arm64:"CSINV\tNE", -"CSEL"
	r0 = x0

	if cond {
		x1 = ^b
	} else {
		x1 = a
	}
	// arm64:"CSINV\tEQ", -"CSEL"
	r1 = x1
}

func cmovneg(cond bool, a, b, c int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = -b
	}
	// arm64:"CSNEG\tNE", -"CSEL"
	r0 = x0

	if cond {
		x1 = -b
	} else {
		x1 = a
	}
	// arm64:"CSNEG\tEQ", -"CSEL"
	r1 = x1
}

func cmovsetm(cond bool, x int) {
	var x0, x1 int

	if cond {
		x0 = -1
	} else {
		x0 = 0
	}
	// arm64:"CSETM\tNE", -"CSEL"
	r0 = x0

	if cond {
		x1 = 0
	} else {
		x1 = -1
	}
	// arm64:"CSETM\tEQ", -"CSEL"
	r1 = x1
}

func cmovFcmp0(s, t float64, a, b int) {
	var x0, x1, x2, x3, x4, x5 int

	if s < t {
		x0 = a
	} else {
		x0 = b + 1
	}
	// arm64:"CSINC\tMI", -"CSEL"
	r0 = x0

	if s <= t {
		x1 = a
	} else {
		x1 = ^b
	}
	// arm64:"CSINV\tLS", -"CSEL"
	r1 = x1

	if s > t {
		x2 = a
	} else {
		x2 = -b
	}
	// arm64:"CSNEG\tMI", -"CSEL"
	r2 = x2

	if s >= t {
		x3 = -1
	} else {
		x3 = 0
	}
	// arm64:"CSETM\tLS", -"CSEL"
	r3 = x3

	if s == t {
		x4 = a
	} else {
		x4 = b + 1
	}
	// arm64:"CSINC\tEQ", -"CSEL"
	r4 = x4

	if s != t {
		x5 = a
	} else {
		x5 = b + 1
	}
	// arm64:"CSINC\tNE", -"CSEL"
	r5 = x5
}

func cmovFcmp1(s, t float64, a, b int) {
	var x0, x1, x2, x3, x4, x5 int

	if s < t {
		x0 = b + 1
	} else {
		x0 = a
	}
	// arm64:"CSINC\tPL", -"CSEL"
	r0 = x0

	if s <= t {
		x1 = ^b
	} else {
		x1 = a
	}
	// arm64:"CSINV\tHI", -"CSEL"
	r1 = x1

	if s > t {
		x2 = -b
	} else {
		x2 = a
	}
	// arm64:"CSNEG\tPL", -"CSEL"
	r2 = x2

	if s >= t {
		x3 = 0
	} else {
		x3 = -1
	}
	// arm64:"CSETM\tHI", -"CSEL"
	r3 = x3

	if s == t {
		x4 = b + 1
	} else {
		x4 = a
	}
	// arm64:"CSINC\tNE", -"CSEL"
	r4 = x4

	if s != t {
		x5 = b + 1
	} else {
		x5 = a
	}
	// arm64:"CSINC\tEQ", -"CSEL"
	r5 = x5
}

func cmovzero1(c bool) int {
	var x int
	if c {
		x = 182
	}
	// loong64:"MASKEQZ", -"MASKNEZ"
	return x
}

func cmovzero2(c bool) int {
	var x int
	if !c {
		x = 182
	}
	// loong64:"MASKNEZ", -"MASKEQZ"
	return x
}

// Conditionally selecting between a value or 0 can be done without
// an extra load of 0 to a register on PPC64 by using R0 (which always
// holds the value $0) instead. Verify both cases where either arg1
// or arg2 is zero.
func cmovzeroreg0(a, b int) int {
	x := 0
	if a == b {
		x = a
	}
	// ppc64x:"ISEL\t[$]2, R[0-9]+, R0, R[0-9]+"
	return x
}

func cmovzeroreg1(a, b int) int {
	x := a
	if a == b {
		x = 0
	}
	// ppc64x:"ISEL\t[$]2, R0, R[0-9]+, R[0-9]+"
	return x
}
