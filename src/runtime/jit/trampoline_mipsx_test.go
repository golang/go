// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows) && (mips || mipsle)

package jit_test

import (
	"encoding/binary"
	"runtime"
	"unsafe"
)

var mipsBE = runtime.GOARCH == "mips"

func mU32(v uint32) []byte {
	b := make([]byte, 4)
	if mipsBE {
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
	copy(code[4:], mU32(0x00000000)) // nop
	return code
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// lui $t9, addr[31:16]
	// ori $t9, $t9, addr[15:0]
	// jr $t9
	// nop
	code := make([]byte, 16)
	hi := uint16(fnAddr >> 16)
	lo := uint16(fnAddr)
	copy(code[0:], mU32(0x3C190000|uint32(hi)))  // lui $t9, hi
	copy(code[4:], mU32(0x37390000|uint32(lo)))   // ori $t9, $t9, lo
	copy(code[8:], mU32(0x03200008))              // jr $t9
	copy(code[12:], mU32(0x00000000))             // nop
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// addiu $sp, $sp, -8
	// sw $ra, 4($sp)
	// sw $fp, 0($sp)
	// move $fp, $sp
	// <load addr into $t9>
	// jalr $t9
	// nop
	// lw $fp, 0($sp)
	// lw $ra, 4($sp)
	// addiu $sp, $sp, 8
	// jr $ra
	// nop
	code := make([]byte, 48)
	off := 0
	hi := uint16(fnAddr >> 16)
	lo := uint16(fnAddr)

	copy(code[off:], mU32(0x27BDFFF8)); off += 4 // addiu $sp, $sp, -8
	copy(code[off:], mU32(0xAFBF0004)); off += 4 // sw $ra, 4($sp)
	copy(code[off:], mU32(0xAFBE0000)); off += 4 // sw $fp, 0($sp)
	copy(code[off:], mU32(0x03A0F025)); off += 4 // move $fp, $sp

	copy(code[off:], mU32(0x3C190000|uint32(hi))); off += 4 // lui $t9, hi
	copy(code[off:], mU32(0x37390000|uint32(lo))); off += 4 // ori $t9, $t9, lo

	copy(code[off:], mU32(0x0320F809)); off += 4 // jalr $t9
	copy(code[off:], mU32(0x00000000)); off += 4 // nop

	copy(code[off:], mU32(0x8FBE0000)); off += 4 // lw $fp, 0($sp)
	copy(code[off:], mU32(0x8FBF0004)); off += 4 // lw $ra, 4($sp)
	copy(code[off:], mU32(0x27BD0008)); off += 4 // addiu $sp, $sp, 8
	copy(code[off:], mU32(0x03E00008)); off += 4 // jr $ra
	return code[:off]
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// [sp+0] = saved $fp, [sp+4] = saved $ra.
		savedFP := *(*uintptr)(unsafe.Pointer(sp))
		savedRA := *(*uintptr)(unsafe.Pointer(sp + 4))
		return savedRA, sp + 8, savedFP, true
	}
}
