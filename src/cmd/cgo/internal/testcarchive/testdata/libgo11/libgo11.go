// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// // faultAfterPacibsp faults right after signing the link register with
// // pacibsp, as C code does in its prologue, before its first call.
// __attribute__((naked)) static void faultAfterPacibsp(void) {
// 	__asm__(
// 		"hint #27\n"     // pacibsp
// 		"mov x9, #0\n"
// 		"ldr x0, [x9]\n"
// 		"hint #31\n"     // autibsp
// 		"ret\n");
// }
import "C"

import (
	"unsafe"

	"testcarchive/libgo11/badfp"
)

// CallBadFuncValue calls a function value whose code address has bits
// set above the virtual address range, as a signed C function pointer
// called without authenticating it would.
//
//export CallBadFuncValue
func CallBadFuncValue() {
	fv := &struct{ fn uintptr }{0x0012345600001000}
	var f func()
	*(*unsafe.Pointer)(unsafe.Pointer(&f)) = unsafe.Pointer(fv)
	f()
}

// FaultWithBadFP dereferences nil with a non-canonical frame pointer.
//
//export FaultWithBadFP
func FaultWithBadFP() {
	badfp.Fault()
}

// FaultAfterPacibsp calls C code that faults with a signed link register.
//
//export FaultAfterPacibsp
func FaultAfterPacibsp() {
	C.faultAfterPacibsp()
}

func main() {}
