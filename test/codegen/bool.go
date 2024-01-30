// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import (
	"math/bits"
)

// This file contains codegen tests related to boolean simplifications/optimizations.

func convertNeq0B(x uint8, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	// ppc64x:"RLDICL",-"CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0W(x uint16, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	// ppc64x:"RLDICL",-"CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0L(x uint32, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETB"
	// ppc64x:"RLDICL",-"CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0Q(x uint64, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETB"
	// ppc64x:"RLDICL",-"CMP",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeqBool32(x uint32) bool {
	// ppc64x:"RLDICL",-"CMPW",-"ISEL"
	return x&1 != 0
}

func convertEqBool32(x uint32) bool {
	// ppc64x:"RLDICL",-"CMPW","XOR",-"ISEL"
	return x&1 == 0
}

func convertNeqBool64(x uint64) bool {
	// ppc64x:"RLDICL",-"CMP",-"ISEL"
	return x&1 != 0
}

func convertEqBool64(x uint64) bool {
	// ppc64x:"RLDICL","XOR",-"CMP",-"ISEL"
	return x&1 == 0
}

func TestSetEq64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0EQ",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0EQ"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0EQ"
	b := x == y
	return b
}
func TestSetNeq64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0EQ",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0EQ"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0EQ"
	b := x != y
	return b
}
func TestSetLt64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0GT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0GT"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0GT"
	b := x < y
	return b
}
func TestSetLe64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0LT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0LT"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0LT"
	b := x <= y
	return b
}
func TestSetGt64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0LT"
	b := x > y
	return b
}
func TestSetGe64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0GT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0GT"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0GT"
	b := x >= y
	return b
}
func TestSetLtFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBC\tCR0LT"
	b := x < y
	return b
}
func TestSetLeFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT","SETBC\tCR0EQ","OR",-"ISEL",-"ISEL"
	// ppc64x/power9:"ISEL","ISEL",-"SETBC\tCR0LT",-"SETBC\tCR0EQ","OR"
	// ppc64x/power8:"ISEL","ISEL",-"SETBC\tCR0LT",-"SETBC\tCR0EQ","OR"
	b := x <= y
	return b
}
func TestSetGtFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBC\tCR0LT"
	b := x > y
	return b
}
func TestSetGeFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT","SETBC\tCR0EQ","OR",-"ISEL",-"ISEL"
	// ppc64x/power9:"ISEL","ISEL",-"SETBC\tCR0LT",-"SETBC\tCR0EQ","OR"
	// ppc64x/power8:"ISEL","ISEL",-"SETBC\tCR0LT",-"SETBC\tCR0EQ","OR"
	b := x >= y
	return b
}
func TestSetInvEq64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0EQ",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0EQ"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0EQ"
	b := !(x == y)
	return b
}
func TestSetInvNeq64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0EQ",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0EQ"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0EQ"
	b := !(x != y)
	return b
}
func TestSetInvLt64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0GT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0GT"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0GT"
	b := !(x < y)
	return b
}
func TestSetInvLe64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0LT"
	b := !(x <= y)
	return b
}
func TestSetInvGt64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBCR\tCR0LT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBCR\tCR0LT"
	// ppc64x/power8:"CMP","ISEL",-"SETBCR\tCR0LT"
	b := !(x > y)
	return b
}
func TestSetInvGe64(x uint64, y uint64) bool {
	// ppc64x/power10:"SETBC\tCR0GT",-"ISEL"
	// ppc64x/power9:"CMP","ISEL",-"SETBC\tCR0GT"
	// ppc64x/power8:"CMP","ISEL",-"SETBC\tCR0GT"
	b := !(x >= y)
	return b
}

func TestSetInvEqFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBCR\tCR0EQ",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBCR\tCR0EQ"
	// ppc64x/power8:"FCMP","ISEL",-"SETBCR\tCR0EQ"
	b := !(x == y)
	return b
}
func TestSetInvNeqFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0EQ",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBC\tCR0EQ"
	// ppc64x/power8:"FCMP","ISEL",-"SETBC\tCR0EQ"
	b := !(x != y)
	return b
}
func TestSetInvLtFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBCR\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBCR\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBCR\tCR0LT"
	b := !(x < y)
	return b
}
func TestSetInvLeFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBC\tCR0LT"
	b := !(x <= y)
	return b
}
func TestSetInvGtFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBCR\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBCR\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBCR\tCR0LT"
	b := !(x > y)
	return b
}
func TestSetInvGeFp64(x float64, y float64) bool {
	// ppc64x/power10:"SETBC\tCR0LT",-"ISEL"
	// ppc64x/power9:"FCMP","ISEL",-"SETBC\tCR0LT"
	// ppc64x/power8:"FCMP","ISEL",-"SETBC\tCR0LT"
	b := !(x >= y)
	return b
}
func TestLogicalCompareZero(x *[64]uint64) {
	// ppc64x:"ANDCC",^"AND"
	b := x[0]&3
	if b!=0 {
		x[0] = b
	}
	// ppc64x:"ANDCC",^"AND"
	b = x[1]&x[2]
	if b!=0 {
		x[1] = b
	}
	// ppc64x:"ANDNCC",^"ANDN"
	b = x[1]&^x[2]
	if b!=0 {
		x[1] = b
	}
	// ppc64x:"ORCC",^"OR"
	b = x[3]|x[4]
	if b!=0 {
		x[3] = b
	}
	// ppc64x:"SUBCC",^"SUB"
	b = x[5]-x[6]
	if b!=0 {
		x[5] = b
	}
	// ppc64x:"NORCC",^"NOR"
	b = ^(x[5]|x[6])
	if b!=0 {
		x[5] = b
	}
	// ppc64x:"XORCC",^"XOR"
	b = x[7]^x[8]
	if b!=0 {
		x[7] = b
	}
	// ppc64x:"ADDCC",^"ADD"
	b = x[9]+x[10]
	if b!=0 {
		x[9] = b
	}
	// ppc64x:"NEGCC",^"NEG"
	b = -x[11]
	if b!=0 {
		x[11] = b
	}
	// ppc64x:"CNTLZDCC",^"CNTLZD"
	b = uint64(bits.LeadingZeros64(x[12]))
	if b!=0 {
		x[12] = b
	}

	// ppc64x:"ADDCCC\t[$]4,"
	c := int64(x[12]) + 4
	if c <= 0 {
		x[12] = uint64(c)
	}

}
