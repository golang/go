// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:noinline
func dummy() {}

// Signed 64-bit compare-and-branch.
func si64(x, y chan int64) {
	// s390x:"CGRJ\t[$](2|4), R[0-9]+, R[0-9]+, "
	for <-x < <-y {
		dummy()
	}

	// s390x:"CL?GRJ\t[$]8, R[0-9]+, R[0-9]+, "
	for <-x == <-y {
		dummy()
	}
}

// Signed 64-bit compare-and-branch with 8-bit immediate.
func si64x8(doNotOptimize int64) {
	// take in doNotOptimize as an argument to avoid the loops being rewritten to count down
	// s390x:"CGIJ\t[$]12, R[0-9]+, [$]127, "
	for i := doNotOptimize; i < 128; i++ {
		dummy()
	}

	// s390x:"CGIJ\t[$]10, R[0-9]+, [$]-128, "
	for i := doNotOptimize; i > -129; i-- {
		dummy()
	}

	// s390x:"CGIJ\t[$]2, R[0-9]+, [$]127, "
	for i := doNotOptimize; i >= 128; i++ {
		dummy()
	}

	// s390x:"CGIJ\t[$]4, R[0-9]+, [$]-128, "
	for i := doNotOptimize; i <= -129; i-- {
		dummy()
	}
}

// Unsigned 64-bit compare-and-branch.
func ui64(x, y chan uint64) {
	// s390x:"CLGRJ\t[$](2|4), R[0-9]+, R[0-9]+, "
	for <-x > <-y {
		dummy()
	}

	// s390x:"CL?GRJ\t[$]6, R[0-9]+, R[0-9]+, "
	for <-x != <-y {
		dummy()
	}
}

// Unsigned 64-bit comparison with 8-bit immediate.
func ui64x8() {
	// s390x:"CLGIJ\t[$]4, R[0-9]+, [$]128, "
	for i := uint64(0); i < 128; i++ {
		dummy()
	}

	// s390x:"CLGIJ\t[$]12, R[0-9]+, [$]255, "
	for i := uint64(0); i < 256; i++ {
		dummy()
	}

	// s390x:"CLGIJ\t[$]2, R[0-9]+, [$]255, "
	for i := uint64(257); i >= 256; i-- {
		dummy()
	}

	// s390x:"CLGIJ\t[$]2, R[0-9]+, [$]0, "
	for i := uint64(1024); i > 0; i-- {
		dummy()
	}
}

// Signed 32-bit compare-and-branch.
func si32(x, y chan int32) {
	// s390x:"CRJ\t[$](2|4), R[0-9]+, R[0-9]+, "
	for <-x < <-y {
		dummy()
	}

	// s390x:"CL?RJ\t[$]8, R[0-9]+, R[0-9]+, "
	for <-x == <-y {
		dummy()
	}
}

// Signed 32-bit compare-and-branch with 8-bit immediate.
func si32x8(doNotOptimize int32) {
	// take in doNotOptimize as an argument to avoid the loops being rewritten to count down
	// s390x:"CIJ\t[$]12, R[0-9]+, [$]127, "
	for i := doNotOptimize; i < 128; i++ {
		dummy()
	}

	// s390x:"CIJ\t[$]10, R[0-9]+, [$]-128, "
	for i := doNotOptimize; i > -129; i-- {
		dummy()
	}

	// s390x:"CIJ\t[$]2, R[0-9]+, [$]127, "
	for i := doNotOptimize; i >= 128; i++ {
		dummy()
	}

	// s390x:"CIJ\t[$]4, R[0-9]+, [$]-128, "
	for i := doNotOptimize; i <= -129; i-- {
		dummy()
	}
}

// Unsigned 32-bit compare-and-branch.
func ui32(x, y chan uint32) {
	// s390x:"CLRJ\t[$](2|4), R[0-9]+, R[0-9]+, "
	for <-x > <-y {
		dummy()
	}

	// s390x:"CL?RJ\t[$]6, R[0-9]+, R[0-9]+, "
	for <-x != <-y {
		dummy()
	}
}

// Unsigned 32-bit comparison with 8-bit immediate.
func ui32x8() {
	// s390x:"CLIJ\t[$]4, R[0-9]+, [$]128, "
	for i := uint32(0); i < 128; i++ {
		dummy()
	}

	// s390x:"CLIJ\t[$]12, R[0-9]+, [$]255, "
	for i := uint32(0); i < 256; i++ {
		dummy()
	}

	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]255, "
	for i := uint32(257); i >= 256; i-- {
		dummy()
	}

	// s390x:"CLIJ\t[$]2, R[0-9]+, [$]0, "
	for i := uint32(1024); i > 0; i-- {
		dummy()
	}
}

// Signed 64-bit comparison with unsigned 8-bit immediate.
func si64xu8(x chan int64) {
	// s390x:"CLGIJ\t[$]8, R[0-9]+, [$]128, "
	for <-x == 128 {
		dummy()
	}

	// s390x:"CLGIJ\t[$]6, R[0-9]+, [$]255, "
	for <-x != 255 {
		dummy()
	}
}

// Signed 32-bit comparison with unsigned 8-bit immediate.
func si32xu8(x chan int32) {
	// s390x:"CLIJ\t[$]8, R[0-9]+, [$]255, "
	for <-x == 255 {
		dummy()
	}

	// s390x:"CLIJ\t[$]6, R[0-9]+, [$]128, "
	for <-x != 128 {
		dummy()
	}
}

// Unsigned 64-bit comparison with signed 8-bit immediate.
func ui64xu8(x chan uint64) {
	// s390x:"CGIJ\t[$]8, R[0-9]+, [$]-1, "
	for <-x == ^uint64(0) {
		dummy()
	}

	// s390x:"CGIJ\t[$]6, R[0-9]+, [$]-128, "
	for <-x != ^uint64(127) {
		dummy()
	}
}

// Unsigned 32-bit comparison with signed 8-bit immediate.
func ui32xu8(x chan uint32) {
	// s390x:"CIJ\t[$]8, R[0-9]+, [$]-128, "
	for <-x == ^uint32(127) {
		dummy()
	}

	// s390x:"CIJ\t[$]6, R[0-9]+, [$]-1, "
	for <-x != ^uint32(0) {
		dummy()
	}
}

// Signed 64-bit comparison with 1/-1 to comparison with 0.
func si64x0(x chan int64) {
	// riscv64:"BGTZ"
	for <-x >= 1 {
		dummy()
	}

	// riscv64:"BLEZ"
	for <-x < 1 {
		dummy()
	}

	// riscv64:"BLTZ"
	for <-x <= -1 {
		dummy()
	}

	// riscv64:"BGEZ"
	for <-x > -1 {
		dummy()
	}
}

// Unsigned 64-bit comparison with 1 to comparison with 0.
func ui64x0(x chan uint64) {
	// riscv64:"BNEZ"
	for <-x >= 1 {
		dummy()
	}

	// riscv64:"BEQZ"
	for <-x < 1 {
		dummy()
	}
}
