// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to arithmetic
// simplifications/optimizations.

func Pow2Muls(n1, n2 int) (int, int) {
	// amd64:"SHLQ\t[$]5",-"IMULQ"
	// 386:"SHLL\t[$]5",-"IMULL"
	// arm:"SLL\t[$]5",-"MUL"
	// arm64:"LSL\t[$]5",-"MUL"
	a := n1 * 32

	// amd64:"SHLQ\t[$]6",-"IMULQ"
	// 386:"SHLL\t[$]6",-"IMULL"
	// arm:"SLL\t[$]6",-"MUL"
	// arm64:"LSL\t[$]6",-"MUL"
	b := -64 * n2

	return a, b
}

// ------------------ //
//    MULs merging    //
// ------------------ //

func MergeMuls1(n int) int {
	// amd64:"IMUL3Q\t[$]46"
	// 386:"IMUL3L\t[$]46"
	return 15*n + 31*n // 46n
}

func MergeMuls2(n int) int {
	// amd64:"IMUL3Q\t[$]23","ADDQ\t[$]29"
	// 386:"IMUL3L\t[$]23","ADDL\t[$]29"
	return 5*n + 7*(n+1) + 11*(n+2) // 23n + 29
}

func MergeMuls3(a, n int) int {
	// amd64:"ADDQ\t[$]19",-"IMULQ\t[$]19"
	// 386:"ADDL\t[$]19",-"IMULL\t[$]19"
	return a*n + 19*n // (a+19)n
}

func MergeMuls4(n int) int {
	// amd64:"IMUL3Q\t[$]14"
	// 386:"IMUL3L\t[$]14"
	return 23*n - 9*n // 14n
}

func MergeMuls5(a, n int) int {
	// amd64:"ADDQ\t[$]-19",-"IMULQ\t[$]19"
	// 386:"ADDL\t[$]-19",-"IMULL\t[$]19"
	return a*n - 19*n // (a-19)n
}
