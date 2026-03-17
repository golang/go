// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows) && (ppc64 || ppc64le)

package jit_test

import (
	"encoding/binary"
	"runtime"
	"unsafe"
)

var ppc64BE = runtime.GOARCH == "ppc64"

func u32(v uint32) []byte {
	b := make([]byte, 4)
	if ppc64BE {
		binary.BigEndian.PutUint32(b, v)
	} else {
		binary.LittleEndian.PutUint32(b, v)
	}
	return b
}

func putAddr(b []byte, addr uintptr) {
	if ppc64BE {
		binary.BigEndian.PutUint64(b, uint64(addr))
	} else {
		binary.LittleEndian.PutUint64(b, uint64(addr))
	}
}

func retTrampoline() []byte {
	// blr (branch to link register)
	return u32(0x4E800020)
}

func tailCallTrampoline(fnAddr uintptr) []byte {
	// PPC64 ELFv2: load address into r12 (function entry point register) and branch.
	// Use a literal pool approach with bcl to get PC.
	//
	// For simplicity, use addis/ori sequence to load 64-bit address:
	// lis r12, addr@highest       ; r12 = (addr >> 48) << 16
	// ori r12, r12, addr@higher   ; r12 |= (addr >> 32) & 0xFFFF
	// sldi r12, r12, 32           ; r12 <<= 32
	// oris r12, r12, addr@h       ; r12 |= ((addr >> 16) & 0xFFFF) << 16
	// ori r12, r12, addr@l        ; r12 |= addr & 0xFFFF
	// mtctr r12
	// bctr                         ; branch via CTR
	code := make([]byte, 28)
	hi48 := uint16(fnAddr >> 48)
	hi32 := uint16(fnAddr >> 32)
	hi16 := uint16(fnAddr >> 16)
	lo16 := uint16(fnAddr)

	copy(code[0:], u32(0x3D800000|uint32(hi48)))  // lis r12, addr@highest
	copy(code[4:], u32(0x618C0000|uint32(hi32)))   // ori r12, r12, addr@higher
	copy(code[8:], u32(0x798C07C6))                // sldi r12, r12, 32 (rldicr r12, r12, 32, 31)
	copy(code[12:], u32(0x658C0000|uint32(hi16)))  // oris r12, r12, addr@h
	copy(code[16:], u32(0x618C0000|uint32(lo16)))  // ori r12, r12, addr@l
	copy(code[20:], u32(0x7D8903A6))               // mtctr r12
	copy(code[24:], u32(0x4E800420))               // bctr
	return code
}

func callTrampoline(fnAddr uintptr) []byte {
	// PPC64 ELFv2 call with minimal frame.
	// PPC64 requires a minimum frame of 32 bytes (with LR save area at caller's frame+16).
	//
	// mflr r0              ; save LR to r0
	// stdu r1, -48(r1)     ; create stack frame (48 bytes: 32 min + 16 for FP/padding)
	// std r0, 64(r1)       ; save LR at caller's LR save area (old_sp + 16 = r1+48+16=r1+64)
	// std r31, 32(r1)      ; save r31 (frame pointer) in local save area
	// mr r31, r1           ; set frame pointer
	// <load address into r12>
	// mtctr r12
	// bctrl                ; call (saves return in LR)
	// ld r31, 32(r1)       ; restore FP
	// ld r0, 64(r1)        ; restore LR
	// mtlr r0
	// addi r1, r1, 48      ; pop frame
	// blr                   ; return
	code := make([]byte, 72)
	hi48 := uint16(fnAddr >> 48)
	hi32 := uint16(fnAddr >> 32)
	hi16 := uint16(fnAddr >> 16)
	lo16 := uint16(fnAddr)
	off := 0

	copy(code[off:], u32(0x7C0802A6)); off += 4  // mflr r0
	copy(code[off:], u32(0xF821FFD1)); off += 4  // stdu r1, -48(r1)
	copy(code[off:], u32(0xF8010040)); off += 4  // std r0, 64(r1)   ; LR save area
	copy(code[off:], u32(0xFBE10020)); off += 4  // std r31, 32(r1)
	copy(code[off:], u32(0x7C3F0B78)); off += 4  // mr r31, r1

	// Load address into r12
	copy(code[off:], u32(0x3D800000|uint32(hi48))); off += 4  // lis r12, hi48
	copy(code[off:], u32(0x618C0000|uint32(hi32))); off += 4  // ori r12, r12, hi32
	copy(code[off:], u32(0x798C07C6)); off += 4               // sldi r12, r12, 32
	copy(code[off:], u32(0x658C0000|uint32(hi16))); off += 4  // oris r12, r12, hi16
	copy(code[off:], u32(0x618C0000|uint32(lo16))); off += 4  // ori r12, r12, lo16
	copy(code[off:], u32(0x7D8903A6)); off += 4               // mtctr r12
	copy(code[off:], u32(0x4E800421)); off += 4               // bctrl

	copy(code[off:], u32(0xEBE10020)); off += 4  // ld r31, 32(r1)
	copy(code[off:], u32(0xE8010040)); off += 4  // ld r0, 64(r1)
	copy(code[off:], u32(0x7C0803A6)); off += 4  // mtlr r0
	copy(code[off:], u32(0x38210030)); off += 4  // addi r1, r1, 48
	copy(code[off:], u32(0x4E800020)); off += 4  // blr
	return code[:off]
}

func nextCallback() func(pc, sp uintptr) (uintptr, uintptr, uintptr, bool) {
	return func(pc, sp uintptr) (callerPC, callerSP, callerBP uintptr, ok bool) {
		// PPC64 frame: sp points to the JIT frame base.
		// [sp+32] = saved r31 (FP)
		// [sp+64] = saved LR (at caller's LR save area = sp + framesize + 16)
		// callerSP = sp + 48 (frame size)
		savedFP := *(*uintptr)(unsafe.Pointer(sp + 32))
		savedLR := *(*uintptr)(unsafe.Pointer(sp + 64))
		return savedLR, sp + 48, savedFP, true
	}
}
