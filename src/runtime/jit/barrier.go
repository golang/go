// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import "runtime"

// BarrierCheckFixup identifies a fixed-size write-barrier check.
type BarrierCheckFixup struct{ offset, length int }

func barrierCheckCode(base, slow int) instructionBlock {
	r := nativeRegisterConfig()
	var b instructionBlock
	b.immediate(r.Scratch, uint64(userFrameWriteBarrier()))
	// The runtime's enabled flag and its zero padding occupy the first word.
	// This representation dependency stays within this toolchain revision.
	b.memory(true, r.Scratch, r.Scratch, 0)
	switch runtime.GOARCH {
	case "amd64":
		n := r.Scratch.Number
		b.byte(0x48|n>>3<<2|n>>3, 0x85, 0xc0|n&7<<3|n&7)
		pos := base + len(b.bytes)
		b.byte(0x0f, 0x85)
		b.u32(uint32(int32(branchDisplacement(slow, pos, 6, 32))))
	case "arm64":
		b.u32(0xb4000040 | uint32(r.Scratch.Number)) // CBZ scratch,+8
		b.jump(base, slow)
	case "riscv64":
		b.u32(uint32(r.Scratch.Number)<<15 | 6<<8 | 0x63) // BEQ scratch,zero,+12
		b.jump(base, slow)
	}
	return b
}

// EmitWriteBarrierCheck branches to slowOffset when GC write barriers are
// enabled. It clobbers Registers.Scratch and condition flags. A caller may use
// plain stores on the disabled path only if no scheduling boundary occurs
// between this check and those stores. The enabled path must use typed Go
// writes, Copy or Clear through a described Go-call safepoint. Pointer-free new
// values do not eliminate deletion barriers for old pointer-bearing contents.
// This is a GC protocol operation, not a general unsynchronized flag query.
func EmitWriteBarrierCheck(dst CodeBuffer, slowOffset int) BarrierCheckFixup {
	b := barrierCheckCode(dst.Offset(), slowOffset)
	f := BarrierCheckFixup{dst.Offset(), len(b.bytes)}
	dst.Write(b.bytes)
	return f
}

// PatchWriteBarrierCheck resolves a forward slow-path offset.
func PatchWriteBarrierCheck(dst CodeBuffer, f BarrierCheckFixup, slowOffset int) {
	b := barrierCheckCode(f.offset, slowOffset)
	if len(b.bytes) != f.length {
		panic("runtime/jit: barrier check patch length changed")
	}
	dst.Patch(f.offset, b.bytes)
}
