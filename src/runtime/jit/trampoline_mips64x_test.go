// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows) && (mips64 || mips64le)

package jit_test

import (
	"encoding/binary"
	"runtime"
	"unsafe"
)

var mips64BE = runtime.GOARCH == "mips64"

func mU32(v uint32) []byte {
	b := make([]byte, 4)
	if mips64BE {
		binary.BigEndian.PutUint32(b, v)
	} else {
		binary.LittleEndian.PutUint32(b, v)
	}
	return b
}

func retTrampoline() []byte {
	// jr $ra; nop
	code := make([]byte, 8)
	copy(code[0:], mU32(0x03E00008)) // jr $ra
	copy(code[4:], mU32(0x00000000)) // nop (branch delay slot)
	return code
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// Load 64-bit address into $t9 (r25, standard for PIC calls) and jump.
	// lui $t9, addr[63:48]
	// ori $t9, $t9, addr[47:32]
	// dsll $t9, $t9, 16
	// ori $t9, $t9, addr[31:16]
	// dsll $t9, $t9, 16
	// ori $t9, $t9, addr[15:0]
	// jr $t9
	// nop
	code := make([]byte, 32)
	w0 := uint16(fnAddr >> 48)
	w1 := uint16(fnAddr >> 32)
	w2 := uint16(fnAddr >> 16)
	w3 := uint16(fnAddr)
	copy(code[0:], mU32(0x3C190000|uint32(w0)))  // lui $t9, w0
	copy(code[4:], mU32(0x37390000|uint32(w1)))   // ori $t9, $t9, w1
	copy(code[8:], mU32(0x0019CC38))              // dsll $t9, $t9, 16
	copy(code[12:], mU32(0x37390000|uint32(w2)))  // ori $t9, $t9, w2
	copy(code[16:], mU32(0x0019CC38))             // dsll $t9, $t9, 16
	copy(code[20:], mU32(0x37390000|uint32(w3)))  // ori $t9, $t9, w3
	copy(code[24:], mU32(0x03200008))             // jr $t9
	copy(code[28:], mU32(0x00000000))             // nop
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// MIPS64 calling convention: $ra = return address, $sp = stack pointer,
	// $fp ($30) = frame pointer.
	//
	// daddiu $sp, $sp, -16
	// sd $ra, 8($sp)
	// sd $fp, 0($sp)
	// move $fp, $sp
	// <load addr into $t9>
	// jalr $t9
	// nop
	// ld $fp, 0($sp)
	// ld $ra, 8($sp)
	// daddiu $sp, $sp, 16
	// jr $ra
	// nop
	code := make([]byte, 80)
	off := 0
	w0 := uint16(fnAddr >> 48)
	w1 := uint16(fnAddr >> 32)
	w2 := uint16(fnAddr >> 16)
	w3 := uint16(fnAddr)

	copy(code[off:], mU32(0x67BDFFF0)); off += 4 // daddiu $sp, $sp, -16
	copy(code[off:], mU32(0xFFBF0008)); off += 4 // sd $ra, 8($sp)
	copy(code[off:], mU32(0xFFBE0000)); off += 4 // sd $fp, 0($sp)
	copy(code[off:], mU32(0x03A0F025)); off += 4 // move $fp, $sp (or $fp, $sp, $zero)

	// Load address into $t9
	copy(code[off:], mU32(0x3C190000|uint32(w0))); off += 4  // lui $t9, w0
	copy(code[off:], mU32(0x37390000|uint32(w1))); off += 4  // ori $t9, $t9, w1
	copy(code[off:], mU32(0x0019CC38)); off += 4             // dsll $t9, $t9, 16
	copy(code[off:], mU32(0x37390000|uint32(w2))); off += 4  // ori $t9, $t9, w2
	copy(code[off:], mU32(0x0019CC38)); off += 4             // dsll $t9, $t9, 16
	copy(code[off:], mU32(0x37390000|uint32(w3))); off += 4  // ori $t9, $t9, w3

	copy(code[off:], mU32(0x0320F809)); off += 4 // jalr $t9
	copy(code[off:], mU32(0x00000000)); off += 4 // nop (delay slot)

	copy(code[off:], mU32(0xDFBE0000)); off += 4 // ld $fp, 0($sp)
	copy(code[off:], mU32(0xDFBF0008)); off += 4 // ld $ra, 8($sp)
	copy(code[off:], mU32(0x67BD0010)); off += 4 // daddiu $sp, $sp, 16
	copy(code[off:], mU32(0x03E00008)); off += 4 // jr $ra
	copy(code[off:], mU32(0x00000000)); off += 4 // nop
	return code[:off]
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// [sp+0] = saved $fp, [sp+8] = saved $ra.
		savedFP := *(*uintptr)(unsafe.Pointer(sp))
		savedRA := *(*uintptr)(unsafe.Pointer(sp + 8))
		return savedRA, sp + 16, savedFP, true
	}
}
