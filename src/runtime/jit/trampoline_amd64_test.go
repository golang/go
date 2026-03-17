// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows)

package jit_test

import "unsafe"

func retTrampoline() []byte {
	return []byte{0xc3} // ret
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// movabs rax, imm64; jmp rax
	code := make([]byte, 12)
	code[0] = 0x48
	code[1] = 0xb8
	putU64(code[2:], fnAddr)
	code[10] = 0xff
	code[11] = 0xe0
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// push rbp; mov rbp, rsp; movabs rax, imm64; call rax; pop rbp; ret
	code := make([]byte, 20)
	code[0] = 0x55                   // push rbp
	code[1] = 0x48; code[2] = 0x89; code[3] = 0xe5 // mov rbp, rsp
	code[4] = 0x48; code[5] = 0xb8  // movabs rax, imm64
	putU64(code[6:], fnAddr)
	code[14] = 0xff; code[15] = 0xd0 // call rax
	code[16] = 0x5d                   // pop rbp
	code[17] = 0xc3                   // ret
	return code
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// frame.fp of callee = entrySP - PtrSize (after push rbp).
		// [sp+0] = saved RBP, [sp+8] = Go caller's return address.
		savedBP := *(*uintptr)(unsafe.Pointer(sp))
		goRetAddr := *(*uintptr)(unsafe.Pointer(sp + 8))
		return goRetAddr, sp + 16, savedBP, true
	}
}

func putU64(b []byte, v uintptr) {
	_ = b[7]
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}
