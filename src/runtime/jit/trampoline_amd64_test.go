// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows)

package jit_test

import "runtime/jit"

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
	code[0] = 0x55 // push rbp
	code[1] = 0x48
	code[2] = 0x89
	code[3] = 0xe5 // mov rbp, rsp
	code[4] = 0x48
	code[5] = 0xb8 // movabs rax, imm64
	putU64(code[6:], fnAddr)
	code[14] = 0xff
	code[15] = 0xd0 // call rax
	code[16] = 0x5d // pop rbp
	code[17] = 0xc3 // ret
	return code
}

func pointerSlotCallTrampoline(fnAddr, ptr uintptr) []byte {
	// Establish a frame with two local words, store ptr in the first word,
	// call Go, then discard the locals and return.
	code := []byte{
		0x55,             // push rbp
		0x48, 0x89, 0xe5, // mov rbp, rsp
		0x48, 0x83, 0xec, 0x10, // sub rsp, 16
		0x48, 0xb8, // movabs rax, ptr
		0, 0, 0, 0, 0, 0, 0, 0,
		0x48, 0x89, 0x04, 0x24, // mov [rsp], rax
		0x48, 0xb8, // movabs rax, fnAddr
		0, 0, 0, 0, 0, 0, 0, 0,
		0xff, 0xd0, // call rax
		0x48, 0x83, 0xc4, 0x10, // add rsp, 16
		0x5d, // pop rbp
		0xc3, // ret
	}
	putU64(code[10:], ptr)
	putU64(code[24:], fnAddr)
	return code
}

func stackPointerCallTrampoline(fnAddr uintptr) []byte {
	code := pointerSlotCallTrampoline(fnAddr, 0)
	// Replace movabs/store with: lea rax,[rsp+8]; mov [rsp],rax; nops.
	copy(code[8:22], []byte{
		0x48, 0x8d, 0x44, 0x24, 0x08,
		0x48, 0x89, 0x04, 0x24,
		0x90, 0x90, 0x90, 0x90, 0x90,
	})
	// Replace frame teardown with a comparison against the relocated address:
	// mov rax,[rsp]; lea rcx,[rsp+8]; cmp rax,rcx; sete al; movzx eax,al.
	check := []byte{
		0x48, 0x8b, 0x04, 0x24,
		0x48, 0x8d, 0x4c, 0x24, 0x08,
		0x48, 0x39, 0xc8,
		0x0f, 0x94, 0xc0,
		0x0f, 0xb6, 0xc0,
		0x48, 0x83, 0xc4, 0x10,
		0x5d,
		0xc3,
	}
	code = append(code[:34], check...)
	return code
}

func callerFramePointerCallTrampoline(fnAddr uintptr) []byte {
	// Save the distance from this frame pointer to the caller's frame pointer,
	// trigger stack growth in Go, then verify that stack copying relocated both
	// pointers by the same amount. The first local word contains only the signed
	// distance and is deliberately absent from the pointer map.
	code := []byte{
		0x55,             // push rbp
		0x48, 0x89, 0xe5, // mov rbp, rsp
		0x48, 0x83, 0xec, 0x10, // sub rsp, 16
		0x48, 0x8b, 0x45, 0x00, // mov rax, [rbp]
		0x48, 0x29, 0xe8, // sub rax, rbp
		0x48, 0x89, 0x04, 0x24, // mov [rsp], rax
		0x48, 0xb8, // movabs rax, fnAddr
		0, 0, 0, 0, 0, 0, 0, 0,
		0xff, 0xd0, // call rax
		0x48, 0x8b, 0x45, 0x00, // mov rax, [rbp]
		0x48, 0x29, 0xe8, // sub rax, rbp
		0x48, 0x3b, 0x04, 0x24, // cmp rax, [rsp]
		0x0f, 0x94, 0xc0, // sete al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc9, // leave
		0xc3, // ret
	}
	putU64(code[21:], fnAddr)
	return code
}

func callTrampolineStackMaps() []jit.StackMap {
	return []jit.StackMap{{PCOffset: 16, HasUnwind: true, CallerPCOffset: 8, CallerSPOffset: 16}}
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
