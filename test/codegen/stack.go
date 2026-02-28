// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import (
	"runtime"
	"unsafe"
)

// This file contains code generation tests related to the use of the
// stack.

// Check that stack stores are optimized away.

// 386:"TEXT .*, [$]0-"
// amd64:"TEXT .*, [$]0-"
// arm:"TEXT .*, [$]-4-"
// arm64:"TEXT .*, [$]0-"
// mips:"TEXT .*, [$]-4-"
// ppc64x:"TEXT .*, [$]0-"
// s390x:"TEXT .*, [$]0-"
func StackStore() int {
	var x int
	return *(&x)
}

type T struct {
	A, B, C, D int // keep exported fields
	x, y, z    int // reset unexported fields
}

// Check that large structs are cleared directly (issue #24416).

// 386:"TEXT .*, [$]0-"
// amd64:"TEXT .*, [$]0-"
// arm:"TEXT .*, [$]0-" (spills return address)
// arm64:"TEXT .*, [$]0-"
// mips:"TEXT .*, [$]-4-"
// ppc64x:"TEXT .*, [$]0-"
// s390x:"TEXT .*, [$]0-"
func ZeroLargeStruct(x *T) {
	t := T{}
	*x = t
}

// Check that structs are partially initialised directly (issue #24386).

// Notes:
// - 386 fails due to spilling a register
// amd64:"TEXT .*, [$]0-"
// arm:"TEXT .*, [$]0-" (spills return address)
// arm64:"TEXT .*, [$]0-"
// ppc64x:"TEXT .*, [$]0-"
// s390x:"TEXT .*, [$]0-"
// Note: that 386 currently has to spill a register.
func KeepWanted(t *T) {
	*t = T{A: t.A, B: t.B, C: t.C, D: t.D}
}

// Check that small array operations avoid using the stack (issue #15925).

// Notes:
// - 386 fails due to spilling a register
// - arm & mips fail due to softfloat calls
// amd64:"TEXT .*, [$]0-"
// arm64:"TEXT .*, [$]0-"
// ppc64x:"TEXT .*, [$]0-"
// s390x:"TEXT .*, [$]0-"
func ArrayAdd64(a, b [4]float64) [4]float64 {
	return [4]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}
}

// Check that small array initialization avoids using the stack.

// 386:"TEXT .*, [$]0-"
// amd64:"TEXT .*, [$]0-"
// arm:"TEXT .*, [$]0-" (spills return address)
// arm64:"TEXT .*, [$]0-"
// mips:"TEXT .*, [$]-4-"
// ppc64x:"TEXT .*, [$]0-"
// s390x:"TEXT .*, [$]0-"
func ArrayInit(i, j int) [4]int {
	return [4]int{i, 0, j, 0}
}

// Check that assembly output has matching offset and base register
// (issue #21064).

func check_asmout(b [2]int) int {
	runtime.GC() // use some frame
	// amd64:`.*b\+24\(SP\)`
	// arm:`.*b\+4\(FP\)`
	return b[1]
}

// Check that simple functions get promoted to nosplit, even when
// they might panic in various ways. See issue 31219.
// amd64:"TEXT .*NOSPLIT.*"
func MightPanic(a []int, i, j, k, s int) {
	_ = a[i]     // panicIndex
	_ = a[i:j]   // panicSlice
	_ = a[i:j:k] // also panicSlice
	_ = i << s   // panicShift
	_ = i / j    // panicDivide
}

// Put a defer in a loop, so second defer is not open-coded
func Defer() {
	for i := 0; i < 2; i++ {
		defer func() {}()
	}
	// amd64:`CALL runtime\.deferprocStack`
	defer func() {}()
}

// Check that stack slots are shared among values of the same
// type, but not pointer-identical types. See issue 65783.

func spillSlotReuse() {
	// The return values of getp1 and getp2 need to be
	// spilled around the calls to nopInt. Make sure that
	// spill slot gets reused.

	//arm64:`.*autotmp_2-8\(SP\)`
	getp1()[nopInt()] = 0
	//arm64:`.*autotmp_2-8\(SP\)`
	getp2()[nopInt()] = 0
}

// Check that no stack frame space is needed for simple slice initialization with underlying structure.
type mySlice struct {
	array unsafe.Pointer
	len   int
	cap   int
}

// amd64:"TEXT .*, [$]0-"
func sliceInit(base uintptr) []uintptr {
	const ptrSize = 8
	size := uintptr(4096)
	bitmapSize := size / ptrSize / 8
	elements := int(bitmapSize / ptrSize)
	var sl mySlice
	sl = mySlice{
		unsafe.Pointer(base + size - bitmapSize),
		elements,
		elements,
	}
	// amd64:-"POPQ" -"SP"
	return *(*[]uintptr)(unsafe.Pointer(&sl))
}

//go:noinline
func nopInt() int {
	return 0
}

//go:noinline
func getp1() *[4]int {
	return nil
}

//go:noinline
func getp2() *[4]int {
	return nil
}

// Store to an argument without read can be removed.
func storeArg(a [2]int) {
	// amd64:-`MOVQ \$123,.*\.a\+\d+\(SP\)`
	a[1] = 123
}
