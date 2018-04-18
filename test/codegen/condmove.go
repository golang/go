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
	return x
}

func cmovchan(x, y chan int) chan int {
	if x != y {
		x = y
	}
	// amd64:"CMOVQNE"
	// arm64:"CSEL\tNE"
	return x
}

func cmovuintptr(x, y uintptr) uintptr {
	if x < y {
		x = -y
	}
	// amd64:"CMOVQCS"
	// arm64:"CSEL\tLO"
	return x
}

func cmov32bit(x, y uint32) uint32 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVLCS"
	// arm64:"CSEL\tLO"
	return x
}

func cmov16bit(x, y uint16) uint16 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVWCS"
	// arm64:"CSEL\tLO"
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
	return a
}

func cmovfloatne(x, y float64) int {
	a := 128
	if x != y {
		a = 256
	}
	// amd64:"CMOVQNE","CMOVQPS"
	// arm64:"CSEL\tNE"
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
		// arm64:"CSEL\tGT"
		r = r - ldexp(y, (rexp-yexp))
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
	return y
}

func cmovuintptr2(x, y uintptr) uintptr {
	a := x * 2
	if a == 0 {
		a = 256
	}
	// amd64:"CMOVQEQ"
	// arm64:"CSEL\tEQ"
	return a
}

// Floating point CMOVs are not supported by amd64/arm64
func cmovfloatmove(x, y int) float64 {
	a := 1.0
	if x <= y {
		a = 2.0
	}
	// amd64:-"CMOV"
	// arm64:-"CSEL"
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
