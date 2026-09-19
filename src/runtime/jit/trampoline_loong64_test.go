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
	// jirl $zero, $ra, 0 (return)
	return leU32(0x4C000020)
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// Load address into $t0 (r12) using pcaddi + ld.d from literal pool.
	// pcaddi $t0, 2      ; t0 = PC + 2*4 = PC+8 (literal at offset 8)
	//                     ; Actually pcaddi adds imm*4, so pcaddi $t0, 2 → t0 = PC + 8
	// ld.d $t0, $t0, 0   ; load 8-byte address
	// jirl $zero, $t0, 0 ; jump (no link)
	// <padding to align>
	// .quad <addr>
	//
	// pcaddi at offset 0: t0 = offset 0 + 2*4 = offset 8.
	// But we have the ld.d at offset 4 and jirl at offset 8, so literal goes at offset 12?
	// Wait: pcaddi $t0, 3 → t0 = PC + 3*4 = offset 0 + 12 = offset 12.
	// Then ld.d $t0, $t0, 0 loads from t0 = offset 12.
	// jirl at offset 8.
	// Literal at offset 12.
	code := make([]byte, 20)
	// pcaddi $t0, 3: opcode=0x18000000, rd=12(t0), imm20=3
	copy(code[0:], leU32(0x1800006C)) // pcaddi $t0, 3
	// ld.d $t0, $t0, 0: opcode=0x28C00000, rd=12, rj=12, imm12=0
	copy(code[4:], leU32(0x28C0018C)) // ld.d $t0, $t0, 0
	// jirl $zero, $t0, 0
	copy(code[8:], leU32(0x4C000180)) // jirl $zero, $t0, 0
	binary.LittleEndian.PutUint64(code[12:], uint64(fnAddr))
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// addi.d $sp, $sp, -16
	// st.d $ra, $sp, 8
	// st.d $fp, $sp, 0
	// addi.d $fp, $sp, 16
	// pcaddi $t0, N       ; load fn address from literal pool
	// ld.d $t0, $t0, 0
	// jirl $ra, $t0, 0    ; call
	// ld.d $fp, $sp, 0
	// ld.d $ra, $sp, 8
	// addi.d $sp, $sp, 16
	// jirl $zero, $ra, 0  ; ret
	// .quad <addr>
	//
	// pcaddi at offset 16, literal at offset 44.
	// imm = (44 - 16) / 4 = 7. So pcaddi $t0, 7.
	code := make([]byte, 52)
	copy(code[0:], leU32(0x02FFC063))  // addi.d $sp, $sp, -16
	copy(code[4:], leU32(0x29C02061))  // st.d $ra, $sp, 8
	copy(code[8:], leU32(0x29C00076))  // st.d $fp, $sp, 0
	copy(code[12:], leU32(0x02C04076)) // addi.d $fp, $sp, 16
	copy(code[16:], leU32(0x180000EC)) // pcaddi $t0, 7
	copy(code[20:], leU32(0x28C0018C)) // ld.d $t0, $t0, 0
	copy(code[24:], leU32(0x4C000181)) // jirl $ra, $t0, 0
	copy(code[28:], leU32(0x28C00076)) // ld.d $fp, $sp, 0
	copy(code[32:], leU32(0x28C02061)) // ld.d $ra, $sp, 8
	copy(code[36:], leU32(0x02C04063)) // addi.d $sp, $sp, 16
	copy(code[40:], leU32(0x4C000020)) // jirl $zero, $ra, 0 (ret)
	binary.LittleEndian.PutUint64(code[44:], uint64(fnAddr))
	return code
}

func callTrampolineStackMaps() []jit.StackMap {
	return []jit.StackMap{{PCOffset: 28, HasUnwind: true, CallerPCOffset: 8, CallerSPOffset: 16}}
}

func leU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}
