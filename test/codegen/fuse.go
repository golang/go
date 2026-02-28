// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math"

// Notes:
// - these examples use channels to provide a source of
//   unknown values that cannot be optimized away
// - these examples use for loops to force branches
//   backward (predicted taken)

// ---------------------------------- //
// signed integer range (conjunction) //
// ---------------------------------- //

func si1c(c <-chan int64) {
	// amd64:"CMPQ .+, [$]256"
	// s390x:"CLGIJ [$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si2c(c <-chan int32) {
	// amd64:"CMPL .+, [$]256"
	// s390x:"CLIJ [$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si3c(c <-chan int16) {
	// amd64:"CMPW .+, [$]256"
	// s390x:"CLIJ [$]12, R[0-9]+, [$]255"
	for x := <-c; x >= 0 && x < 256; x = <-c {
	}
}

func si4c(c <-chan int8) {
	// amd64:"CMPB .+, [$]10"
	// s390x:"CLIJ [$]4, R[0-9]+, [$]10"
	for x := <-c; x >= 0 && x < 10; x = <-c {
	}
}

func si5c(c <-chan int64) {
	// amd64:"CMPQ .+, [$]251" "ADDQ [$]-5,"
	// s390x:"CLGIJ [$]4, R[0-9]+, [$]251" "ADD [$]-5,"
	for x := <-c; x < 256 && x > 4; x = <-c {
	}
}

func si6c(c <-chan int32) {
	// amd64:"CMPL .+, [$]255" "DECL "
	// s390x:"CLIJ [$]12, R[0-9]+, [$]255" "ADDW [$]-1,"
	for x := <-c; x > 0 && x <= 256; x = <-c {
	}
}

func si7c(c <-chan int16) {
	// amd64:"CMPW .+, [$]60" "ADDL [$]10,"
	// s390x:"CLIJ [$]12, R[0-9]+, [$]60" "ADDW [$]10,"
	for x := <-c; x >= -10 && x <= 50; x = <-c {
	}
}

func si8c(c <-chan int8) {
	// amd64:"CMPB .+, [$]126" "ADDL [$]126,"
	// s390x:"CLIJ [$]4, R[0-9]+, [$]126" "ADDW [$]126,"
	for x := <-c; x >= -126 && x < 0; x = <-c {
	}
}

// ---------------------------------- //
// signed integer range (disjunction) //
// ---------------------------------- //

func si1d(c <-chan int64) {
	// amd64:"CMPQ .+, [$]256"
	// s390x:"CLGIJ [$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si2d(c <-chan int32) {
	// amd64:"CMPL .+, [$]256"
	// s390x:"CLIJ [$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si3d(c <-chan int16) {
	// amd64:"CMPW .+, [$]256"
	// s390x:"CLIJ [$]2, R[0-9]+, [$]255"
	for x := <-c; x < 0 || x >= 256; x = <-c {
	}
}

func si4d(c <-chan int8) {
	// amd64:"CMPB .+, [$]10"
	// s390x:"CLIJ [$]10, R[0-9]+, [$]10"
	for x := <-c; x < 0 || x >= 10; x = <-c {
	}
}

func si5d(c <-chan int64) {
	// amd64:"CMPQ .+, [$]251" "ADDQ [$]-5,"
	// s390x:"CLGIJ [$]10, R[0-9]+, [$]251" "ADD [$]-5,"
	for x := <-c; x >= 256 || x <= 4; x = <-c {
	}
}

func si6d(c <-chan int32) {
	// amd64:"CMPL .+, [$]255" "DECL "
	// s390x:"CLIJ [$]2, R[0-9]+, [$]255" "ADDW [$]-1,"
	for x := <-c; x <= 0 || x > 256; x = <-c {
	}
}

func si7d(c <-chan int16) {
	// amd64:"CMPW .+, [$]60" "ADDL [$]10,"
	// s390x:"CLIJ [$]2, R[0-9]+, [$]60" "ADDW [$]10,"
	for x := <-c; x < -10 || x > 50; x = <-c {
	}
}

func si8d(c <-chan int8) {
	// amd64:"CMPB .+, [$]126" "ADDL [$]126,"
	// s390x:"CLIJ [$]10, R[0-9]+, [$]126" "ADDW [$]126,"
	for x := <-c; x < -126 || x >= 0; x = <-c {
	}
}

// ------------------------------------ //
// unsigned integer range (conjunction) //
// ------------------------------------ //

func ui1c(c <-chan uint64) {
	// amd64:"CMPQ .+, [$]251" "ADDQ [$]-5,"
	// s390x:"CLGIJ [$]4, R[0-9]+, [$]251" "ADD [$]-5,"
	for x := <-c; x < 256 && x > 4; x = <-c {
	}
}

func ui2c(c <-chan uint32) {
	// amd64:"CMPL .+, [$]255" "DECL "
	// s390x:"CLIJ [$]12, R[0-9]+, [$]255" "ADDW [$]-1,"
	for x := <-c; x > 0 && x <= 256; x = <-c {
	}
}

func ui3c(c <-chan uint16) {
	// amd64:"CMPW .+, [$]40" "ADDL [$]-10,"
	// s390x:"CLIJ [$]12, R[0-9]+, [$]40" "ADDW [$]-10,"
	for x := <-c; x >= 10 && x <= 50; x = <-c {
	}
}

func ui4c(c <-chan uint8) {
	// amd64:"CMPB .+, [$]2" "ADDL [$]-126,"
	// s390x:"CLIJ [$]4, R[0-9]+, [$]2" "ADDW [$]-126,"
	for x := <-c; x >= 126 && x < 128; x = <-c {
	}
}

// ------------------------------------ //
// unsigned integer range (disjunction) //
// ------------------------------------ //

func ui1d(c <-chan uint64) {
	// amd64:"CMPQ .+, [$]251" "ADDQ [$]-5,"
	// s390x:"CLGIJ [$]10, R[0-9]+, [$]251" "ADD [$]-5,"
	for x := <-c; x >= 256 || x <= 4; x = <-c {
	}
}

func ui2d(c <-chan uint32) {
	// amd64:"CMPL .+, [$]254" "ADDL [$]-2,"
	// s390x:"CLIJ [$]2, R[0-9]+, [$]254" "ADDW [$]-2,"
	for x := <-c; x <= 1 || x > 256; x = <-c {
	}
}

func ui3d(c <-chan uint16) {
	// amd64:"CMPW .+, [$]40" "ADDL [$]-10,"
	// s390x:"CLIJ [$]2, R[0-9]+, [$]40" "ADDW [$]-10,"
	for x := <-c; x < 10 || x > 50; x = <-c {
	}
}

func ui4d(c <-chan uint8) {
	// amd64:"CMPB .+, [$]2" "ADDL [$]-126,"
	// s390x:"CLIJ [$]10, R[0-9]+, [$]2" "ADDW [$]-126,"
	for x := <-c; x < 126 || x >= 128; x = <-c {
	}
}

// ------------------------------------ //
// single bit difference (conjunction)  //
// ------------------------------------ //

func sisbc64(c <-chan int64) {
	// amd64: "ORQ [$]2,"
	// riscv64: "ORI [$]2,"
	for x := <-c; x != 4 && x != 6; x = <-c {
	}
}

func sisbc32(c <-chan int32) {
	// amd64: "ORL [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x != -1 && x != -5; x = <-c {
	}
}

func sisbc16(c <-chan int16) {
	// amd64: "ORL [$]32,"
	// riscv64: "ORI [$]32,"
	for x := <-c; x != 16 && x != 48; x = <-c {
	}
}

func sisbc8(c <-chan int8) {
	// amd64: "ORL [$]16,"
	// riscv64: "ORI [$]16,"
	for x := <-c; x != -15 && x != -31; x = <-c {
	}
}

func uisbc64(c <-chan uint64) {
	// amd64: "ORQ [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x != 1 && x != 5; x = <-c {
	}
}

func uisbc32(c <-chan uint32) {
	// amd64: "ORL [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x != 2 && x != 6; x = <-c {
	}
}

func uisbc16(c <-chan uint16) {
	// amd64: "ORL [$]32,"
	// riscv64: "ORI [$]32,"
	for x := <-c; x != 16 && x != 48; x = <-c {
	}
}

func uisbc8(c <-chan uint8) {
	// amd64: "ORL [$]64,"
	// riscv64: "ORI [$]64,"
	for x := <-c; x != 64 && x != 0; x = <-c {
	}
}

// ------------------------------------ //
// single bit difference (disjunction)  //
// ------------------------------------ //

func sisbd64(c <-chan int64) {
	// amd64: "ORQ [$]2,"
	// riscv64: "ORI [$]2,"
	for x := <-c; x == 4 || x == 6; x = <-c {
	}
}

func sisbd32(c <-chan int32) {
	// amd64: "ORL [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x == -1 || x == -5; x = <-c {
	}
}

func sisbd16(c <-chan int16) {
	// amd64: "ORL [$]32,"
	// riscv64: "ORI [$]32,"
	for x := <-c; x == 16 || x == 48; x = <-c {
	}
}

func sisbd8(c <-chan int8) {
	// amd64: "ORL [$]16,"
	// riscv64: "ORI [$]16,"
	for x := <-c; x == -15 || x == -31; x = <-c {
	}
}

func uisbd64(c <-chan uint64) {
	// amd64: "ORQ [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x == 1 || x == 5; x = <-c {
	}
}

func uisbd32(c <-chan uint32) {
	// amd64: "ORL [$]4,"
	// riscv64: "ORI [$]4,"
	for x := <-c; x == 2 || x == 6; x = <-c {
	}
}

func uisbd16(c <-chan uint16) {
	// amd64: "ORL [$]32,"
	// riscv64: "ORI [$]32,"
	for x := <-c; x == 16 || x == 48; x = <-c {
	}
}

func uisbd8(c <-chan uint8) {
	// amd64: "ORL [$]64,"
	// riscv64: "ORI [$]64,"
	for x := <-c; x == 64 || x == 0; x = <-c {
	}
}

// -------------------------------------//
// merge NaN checks                     //
// ------------------------------------ //

func f64NaNOrPosInf(c <-chan float64) {
	// This test assumes IsInf(x, 1) is implemented as x > MaxFloat rather than x == Inf(1).

	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FCLASSD" -"FLED" -"FLTD" -"FNED" -"FEQD"
	for x := <-c; math.IsNaN(x) || math.IsInf(x, 1); x = <-c {
	}
}

func f64NaNOrNegInf(c <-chan float64) {
	// This test assumes IsInf(x, -1) is implemented as x < -MaxFloat rather than x == Inf(-1).

	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FCLASSD" -"FLED" -"FLTD" -"FNED" -"FEQD"
	for x := <-c; math.IsNaN(x) || math.IsInf(x, -1); x = <-c {
	}
}

func f64NaNOrLtOne(c <-chan float64) {
	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLED" -"FLTD" -"FNED" -"FEQD"
	for x := <-c; math.IsNaN(x) || x < 1; x = <-c {
	}
}

func f64NaNOrLteOne(c <-chan float64) {
	// amd64:"JLS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLTD" -"FLED" -"FNED" -"FEQD"
	for x := <-c; x <= 1 || math.IsNaN(x); x = <-c {
	}
}

func f64NaNOrGtOne(c <-chan float64) {
	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLED" -"FLTD" -"FNED" -"FEQD"
	for x := <-c; math.IsNaN(x) || x > 1; x = <-c {
	}
}

func f64NaNOrGteOne(c <-chan float64) {
	// amd64:"JLS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLTD" -"FLED" -"FNED" -"FEQD"
	for x := <-c; x >= 1 || math.IsNaN(x); x = <-c {
	}
}

func f32NaNOrLtOne(c <-chan float32) {
	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLES" -"FLTS" -"FNES" -"FEQS"
	for x := <-c; x < 1 || x != x; x = <-c {
	}
}

func f32NaNOrLteOne(c <-chan float32) {
	// amd64:"JLS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLTS" -"FLES" -"FNES" -"FEQS"
	for x := <-c; x != x || x <= 1; x = <-c {
	}
}

func f32NaNOrGtOne(c <-chan float32) {
	// amd64:"JCS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLES" -"FLTS" -"FNES" -"FEQS"
	for x := <-c; x > 1 || x != x; x = <-c {
	}
}

func f32NaNOrGteOne(c <-chan float32) {
	// amd64:"JLS" -"JNE" -"JPS" -"JPC"
	// riscv64:"FLTS" -"FLES" -"FNES" -"FEQS"
	for x := <-c; x != x || x >= 1; x = <-c {
	}
}

// ------------------------------------ //
// regressions                          //
// ------------------------------------ //

func gte4(x uint64) bool {
	return x >= 4
}

func lt20(x uint64) bool {
	return x < 20
}

func issue74915(c <-chan uint64) {
	// Check that the optimization is not blocked by function inlining.

	// amd64:"CMPQ .+, [$]16" "ADDQ [$]-4,"
	// s390x:"CLGIJ [$]4, R[0-9]+, [$]16" "ADD [$]-4,"
	for x := <-c; gte4(x) && lt20(x); x = <-c {
	}
}
