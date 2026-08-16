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
	// bx lr
	return leU32(0xE12FFF1E)
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// ldr r12, [pc, #0]; bx r12; .word <addr>
	// At offset 0: ldr r12, [pc, #0] loads from PC+8+0 = offset 8.
	// At offset 4: bx r12.
	// At offset 8: function address.
	code := make([]byte, 12)
	copy(code[0:], leU32(0xE59FC000)) // ldr r12, [pc, #0]
	copy(code[4:], leU32(0xE12FFF1C)) // bx r12
	binary.LittleEndian.PutUint32(code[8:], uint32(fnAddr))
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// push {r11, lr}
	// mov r11, sp
	// ldr r12, [pc, #8]  (loads from PC+8+8 = current+16, literal at offset 24)
	// blx r12
	// pop {r11, lr}
	// bx lr
	// .word <addr>
	code := make([]byte, 28)
	copy(code[0:], leU32(0xE92D4800))  // push {r11, lr}
	copy(code[4:], leU32(0xE1A0B00D))  // mov r11, sp
	copy(code[8:], leU32(0xE59FC008))  // ldr r12, [pc, #8] → loads from offset 24
	copy(code[12:], leU32(0xE12FFF3C)) // blx r12
	copy(code[16:], leU32(0xE8BD4800)) // pop {r11, lr}
	copy(code[20:], leU32(0xE12FFF1E)) // bx lr
	binary.LittleEndian.PutUint32(code[24:], uint32(fnAddr))
	return code
}

func callTrampolineStackMaps() []jit.StackMap {
	return []jit.StackMap{{PCOffset: 16, HasUnwind: true, CallerPCOffset: 4, CallerSPOffset: 8}}
}

func leU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}
