// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows)

package jit_test

import (
	"encoding/binary"
	"unsafe"
)

func retTrampoline() []byte {
	// ret
	return leU32(0xD65F03C0)
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// ldr x16, .+8; br x16; .quad <addr>
	code := make([]byte, 16)
	copy(code[0:], leU32(0x58000050)) // ldr x16, #+8
	copy(code[4:], leU32(0xD61F0200)) // br x16
	binary.LittleEndian.PutUint64(code[8:], uint64(fnAddr))
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// stp x29, x30, [sp, #-16]!
	// mov x29, sp
	// ldr x16, .+16 (offset to literal at +16 from this instr = 4 words)
	// blr x16
	// ldp x29, x30, [sp], #16
	// ret
	// .quad <addr>
	code := make([]byte, 32)
	copy(code[0:], leU32(0xA9BF7BFD))  // stp x29, x30, [sp, #-16]!
	copy(code[4:], leU32(0x910003FD))  // mov x29, sp
	copy(code[8:], leU32(0x58000090))  // ldr x16, #+16 (literal at offset 24)
	copy(code[12:], leU32(0xD63F0200)) // blr x16
	copy(code[16:], leU32(0xA8C17BFD)) // ldp x29, x30, [sp], #16
	copy(code[20:], leU32(0xD65F03C0)) // ret
	binary.LittleEndian.PutUint64(code[24:], uint64(fnAddr))
	return code
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// frame.fp of callee = JIT's SP (after stp decremented by 16).
		// [sp+0] = saved x29 (caller FP), [sp+8] = saved x30 (caller LR).
		savedFP := *(*uintptr)(unsafe.Pointer(sp))
		savedLR := *(*uintptr)(unsafe.Pointer(sp + 8))
		return savedLR, sp + 16, savedFP, true
	}
}

func leU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}
