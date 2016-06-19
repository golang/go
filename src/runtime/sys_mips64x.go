// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

package runtime

import "unsafe"

// adjust Gobuf as if it executed a call to fn with context ctxt
// and then did an immediate Gosave.
func gostartcall(buf *gobuf, fn, ctxt unsafe.Pointer) {
	if buf.lr != 0 {
		throw("invalid use of gostartcall")
	}
	buf.lr = buf.pc
	buf.pc = uintptr(fn)
	buf.ctxt = ctxt
}

// Called to rewind context saved during morestack back to beginning of function.
// To help us, the linker emits a jmp back to the beginning right after the
// call to morestack. We just have to decode and apply that jump.
func rewindmorestack(buf *gobuf) {
	var inst uint32
	if buf.pc&3 == 0 && buf.pc != 0 {
		inst = *(*uint32)(unsafe.Pointer(buf.pc))
		if inst>>26 == 2 { // JMP addr
			//print("runtime: rewind pc=", hex(buf.pc), " to pc=", hex(buf.pc &^ uintptr(1<<28-1) | uintptr((inst&^0xfc000000)<<2)), "\n");
			buf.pc &^= 1<<28 - 1
			buf.pc |= uintptr((inst &^ 0xfc000000) << 2)
			return
		}
		if inst>>16 == 0x1000 { // BEQ	R0, R0, offset
			//print("runtime: rewind pc=", hex(buf.pc), " to pc=", hex(buf.pc + uintptr(int32(int16(inst&0xffff))<<2 + 4)), "\n");
			buf.pc += uintptr(int32(int16(inst&0xffff))<<2 + 4)
			return
		}
	}
	print("runtime: pc=", hex(buf.pc), " ", hex(inst), "\n")
	throw("runtime: misuse of rewindmorestack")
}
