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
	// ret (jalr x0, ra, 0)
	return leU32(0x00008067)
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// Load 64-bit address into t1 (x6) using auipc + ld from literal pool,
	// then jr t1.
	// auipc t1, 0       ; t1 = PC
	// ld t1, 12(t1)     ; load from PC+12 (literal at offset 12)
	// jr t1             ; jump
	// .quad <addr>      ; at offset 12
	code := make([]byte, 20)
	copy(code[0:], leU32(0x00000317))  // auipc t1, 0
	copy(code[4:], leU32(0x00C33303))  // ld t1, 12(t1)
	copy(code[8:], leU32(0x00030067))  // jalr x0, t1, 0 (jr t1)
	binary.LittleEndian.PutUint64(code[12:], uint64(fnAddr))
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// addi sp, sp, -16   ; allocate frame
	// sd ra, 8(sp)       ; save return address
	// sd s0, 0(sp)       ; save frame pointer
	// addi s0, sp, 16    ; s0 = old sp (frame pointer)
	// auipc t1, 0        ; t1 = PC
	// ld t1, 24(t1)      ; load fn addr from literal at offset 20+24-20=offset 40? Let me recalc.
	//
	// Layout:
	// offset 0:  addi sp, sp, -16
	// offset 4:  sd ra, 8(sp)
	// offset 8:  sd s0, 0(sp)
	// offset 12: addi s0, sp, 16
	// offset 16: auipc t1, 0       ; t1 = PC of this instr
	// offset 20: ld t1, 20(t1)     ; load from PC+20 = offset 36... wait
	// Let me use: ld t1, N(t1) where N = literal_offset - auipc_offset
	// auipc is at offset 16. Literal is at offset 36.
	// N = 36 - 16 = 20.
	// offset 24: jalr ra, t1, 0    ; call
	// offset 28: ld s0, 0(sp)
	// offset 32: ld ra, 8(sp)
	// offset 36: addi sp, sp, 16
	// offset 40: ret
	// offset 44: .quad <addr>
	//
	// Wait, the ld after auipc loads from t1+20 = PC_of_auipc + 20 = offset 16+20 = 36.
	// But at offset 36 I have addi sp, sp, 16. The literal needs to go after all code.
	// Let me restructure:
	// offset 0:  addi sp, sp, -16
	// offset 4:  sd ra, 8(sp)
	// offset 8:  sd s0, 0(sp)
	// offset 12: addi s0, sp, 16
	// offset 16: auipc t1, 0
	// offset 20: ld t1, 28(t1)     ; load from offset 16+28 = 44
	// offset 24: jalr ra, t1, 0
	// offset 28: ld s0, 0(sp)
	// offset 32: ld ra, 8(sp)
	// offset 36: addi sp, sp, 16
	// offset 40: ret
	// offset 44: .quad <addr>       ; 8 bytes
	// Total: 52 bytes
	code := make([]byte, 52)

	// addi sp, sp, -16: imm=-16, rs1=sp(x2), rd=sp(x2), funct3=000, opcode=0010011
	// -16 = 0xFF0 in 12-bit signed
	copy(code[0:], leU32(0xFF010113))  // addi sp, sp, -16
	copy(code[4:], leU32(0x00113423))  // sd ra, 8(sp)
	copy(code[8:], leU32(0x00813023))  // sd s0, 0(sp)
	copy(code[12:], leU32(0x01010413)) // addi s0, sp, 16
	copy(code[16:], leU32(0x00000317)) // auipc t1, 0
	// ld t1, 28(t1): imm=28=0x1C, rs1=t1(x6), rd=t1(x6), funct3=011, opcode=0000011
	copy(code[20:], leU32(0x01C33303)) // ld t1, 28(t1)
	copy(code[24:], leU32(0x000300E7)) // jalr ra, t1, 0
	copy(code[28:], leU32(0x00013403)) // ld s0, 0(sp)
	copy(code[32:], leU32(0x00813083)) // ld ra, 8(sp)
	copy(code[36:], leU32(0x01010113)) // addi sp, sp, 16
	copy(code[40:], leU32(0x00008067)) // ret
	binary.LittleEndian.PutUint64(code[44:], uint64(fnAddr))
	return code
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// JIT frame: [sp+0] = saved s0 (FP), [sp+8] = saved ra (LR).
		savedFP := *(*uintptr)(unsafe.Pointer(sp))
		savedRA := *(*uintptr)(unsafe.Pointer(sp + 8))
		return savedRA, sp + 16, savedFP, true
	}
}

func leU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}
