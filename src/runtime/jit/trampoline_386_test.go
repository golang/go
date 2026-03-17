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
	return []byte{0xc3} // ret
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// mov eax, imm32; jmp eax
	code := make([]byte, 7)
	code[0] = 0xb8 // mov eax, imm32
	binary.LittleEndian.PutUint32(code[1:], uint32(fnAddr))
	code[5] = 0xff
	code[6] = 0xe0 // jmp eax
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// push ebp; mov ebp, esp; mov eax, imm32; call eax; pop ebp; ret
	code := make([]byte, 12)
	code[0] = 0x55             // push ebp
	code[1] = 0x89; code[2] = 0xe5 // mov ebp, esp
	code[3] = 0xb8             // mov eax, imm32
	binary.LittleEndian.PutUint32(code[4:], uint32(fnAddr))
	code[8] = 0xff; code[9] = 0xd0 // call eax
	code[10] = 0x5d            // pop ebp
	code[11] = 0xc3            // ret
	return code
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// Same layout as amd64 but 4-byte pointers.
		// [sp+0] = saved EBP, [sp+4] = Go caller's return address.
		savedBP := *(*uintptr)(unsafe.Pointer(sp))
		goRetAddr := *(*uintptr)(unsafe.Pointer(sp + 4))
		return goRetAddr, sp + 8, savedBP, true
	}
}
