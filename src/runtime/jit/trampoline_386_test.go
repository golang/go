// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows)

package jit_test

import (
	"encoding/binary"
	"runtime/jit"
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
	code[0] = 0x55 // push ebp
	code[1] = 0x89
	code[2] = 0xe5 // mov ebp, esp
	code[3] = 0xb8 // mov eax, imm32
	binary.LittleEndian.PutUint32(code[4:], uint32(fnAddr))
	code[8] = 0xff
	code[9] = 0xd0  // call eax
	code[10] = 0x5d // pop ebp
	code[11] = 0xc3 // ret
	return code
}

func callTrampolineStackMaps() []jit.StackMap {
	return []jit.StackMap{{PCOffset: 10, HasUnwind: true, CallerPCOffset: 4, CallerSPOffset: 8}}
}
