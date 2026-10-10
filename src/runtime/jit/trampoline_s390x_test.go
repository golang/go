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
	// br %r14 (return via link register)
	return []byte{0x07, 0xFE}
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// Load 64-bit address into r1 and branch.
	// lgrl r1, .+8    ; PC-relative load (6 bytes)
	// br r1           ; branch (2 bytes)
	// .quad <addr>    ; 8 bytes
	//
	// lgrl: opcode C4 18, ri2 = (literal_offset - instr_offset) / 2
	// instr at offset 0, literal at offset 8. ri2 = (8-0)/2 = 4.
	code := make([]byte, 16)
	// lgrl r1, .+8: C4 18 0000 0004 (big-endian)
	code[0] = 0xC4
	code[1] = 0x18
	binary.BigEndian.PutUint32(code[2:], 4) // halfword offset
	// br r1: 07 F1
	code[6] = 0x07
	code[7] = 0xF1
	binary.BigEndian.PutUint64(code[8:], uint64(fnAddr))
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// s390x uses r14 as link register, r15 as stack pointer, r11 as frame pointer.
	//
	// stmg r11, r14, 0(r15)   ; save r11 and r14 (frame ptr + LR) — but stmg saves a range
	// Actually simpler: save FP and LR individually.
	//
	// aghi r15, -16           ; allocate frame
	// stg r14, 8(r15)         ; save LR
	// stg r11, 0(r15)         ; save FP
	// lgr r11, r15            ; set frame pointer
	// lgrl r1, .+N            ; load function address
	// basr r14, r1            ; call (branch and save return in r14)
	// lg r11, 0(r15)          ; restore FP
	// lg r14, 8(r15)          ; restore LR
	// aghi r15, 16            ; pop frame
	// br r14                  ; return
	// .quad <addr>
	//
	// Instruction sizes:
	// aghi: 4 bytes, stg: 6 bytes, lgr: 4 bytes, lgrl: 6 bytes,
	// basr: 2 bytes, lg: 6 bytes, br: 2 bytes
	code := make([]byte, 56)
	off := 0

	// aghi r15, -16: A7 F9 FFF0 (4 bytes)
	code[off] = 0xA7
	code[off+1] = 0xF9
	code[off+2] = 0xFF
	code[off+3] = 0xF0
	off += 4

	// stg r14, 8(r15): E3 E0 F008 0024 (6 bytes)
	code[off] = 0xE3
	code[off+1] = 0xE0
	code[off+2] = 0xF0
	code[off+3] = 0x08
	code[off+4] = 0x00
	code[off+5] = 0x24
	off += 6

	// stg r11, 0(r15): E3 B0 F000 0024 (6 bytes)
	code[off] = 0xE3
	code[off+1] = 0xB0
	code[off+2] = 0xF0
	code[off+3] = 0x00
	code[off+4] = 0x00
	code[off+5] = 0x24
	off += 6

	// lgr r11, r15: B9 04 00 BF (4 bytes)
	code[off] = 0xB9
	code[off+1] = 0x04
	code[off+2] = 0x00
	code[off+3] = 0xBF
	off += 4

	// lgrl r1, literal: C4 18 <ri2> (6 bytes)
	// ri2 = (literal_offset - current_offset) / 2
	// current_offset = 20, literal needs to be after all code.
	// Remaining: basr(2) + lg(6) + lg(6) + aghi(4) + br(2) = 20 bytes
	// literal at offset 20 + 6 + 20 = 46... Let me just compute.
	lgrlOff := off // offset 20
	code[off] = 0xC4
	code[off+1] = 0x18
	off += 6 // ri2 will be filled in after we know literal offset

	// basr r14, r1: 0D E1 (2 bytes)
	code[off] = 0x0D
	code[off+1] = 0xE1
	off += 2

	// lg r11, 0(r15): E3 B0 F000 0004 (6 bytes)
	code[off] = 0xE3
	code[off+1] = 0xB0
	code[off+2] = 0xF0
	code[off+3] = 0x00
	code[off+4] = 0x00
	code[off+5] = 0x04
	off += 6

	// lg r14, 8(r15): E3 E0 F008 0004 (6 bytes)
	code[off] = 0xE3
	code[off+1] = 0xE0
	code[off+2] = 0xF0
	code[off+3] = 0x08
	code[off+4] = 0x00
	code[off+5] = 0x04
	off += 6

	// aghi r15, 16: A7 F9 0010 (4 bytes)
	code[off] = 0xA7
	code[off+1] = 0xF9
	code[off+2] = 0x00
	code[off+3] = 0x10
	off += 4

	// br r14: 07 FE (2 bytes)
	code[off] = 0x07
	code[off+1] = 0xFE
	off += 2

	// Literal: 8 bytes (must be 8-byte aligned; pad if needed)
	if off%8 != 0 {
		off += 8 - off%8
	}
	literalOff := off
	binary.BigEndian.PutUint64(code[off:], uint64(fnAddr))
	off += 8

	// Patch lgrl ri2
	ri2 := int32((literalOff - lgrlOff) / 2)
	binary.BigEndian.PutUint32(code[lgrlOff+2:], uint32(ri2))

	return code[:off]
}

func callTrampolineStackMaps() []jit.StackMap {
	return []jit.StackMap{{PCOffset: 28, HasUnwind: true, CallerPCOffset: 8, CallerSPOffset: 16}}
}
