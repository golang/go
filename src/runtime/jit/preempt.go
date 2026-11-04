// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"reflect"
	"runtime"
)

// PreemptCheckFixup identifies an optional cooperative preemption check.
type PreemptCheckFixup struct{ offset, length int }

func preemptCheckCode(base, slow int) instructionBlock {
	r := nativeRegisterConfig()
	d, g := uint32(r.Scratch.Number), uint32(r.Goroutine.Number)
	off := userFramePreemptOffset()
	var b instructionBlock
	switch runtime.GOARCH {
	case "amd64":
		b.byte(0x40|byte(d>>3)<<2|byte(g>>3), 0x0f, 0xb6, 0x80|byte(d&7)<<3|byte(g&7))
		b.u32(uint32(off)) // MOVZX scratch,byte [g+preempt]
		b.byte(0x48|byte(d>>3)<<2|byte(d>>3), 0x85, 0xc0|byte(d&7)<<3|byte(d&7))
		pos := base + len(b.bytes)
		b.byte(0x0f, 0x85)
		b.u32(uint32(int32(branchDisplacement(slow, pos, 6, 32))))
	case "arm64":
		if off > 4095 {
			panic("runtime/jit: preempt flag offset out of range")
		}
		b.u32(0x39400000 | uint32(off)<<10 | g<<5 | d) // LDRB
		b.u32(0xb4000040 | d)                          // CBZ scratch,+8
		b.jump(base, slow)
	case "riscv64":
		if off > 2047 {
			panic("runtime/jit: preempt flag offset out of range")
		}
		b.u32(uint32(off)<<20 | g<<15 | 4<<12 | d<<7 | 3) // LBU
		b.u32(d<<15 | 6<<8 | 0x63)                        // BEQ scratch,zero,+12
		b.jump(base, slow)
	}
	return b
}

// EmitPreemptCheck branches to slowOffset when the current goroutine has a
// pending preemption request. It clobbers Registers.Scratch and condition flags.
// It neither creates a safepoint nor yields. The client chooses whether and
// where to emit checks and must reach an actual scheduling boundary promptly
// if it runs unbounded generated code. Bounded kernels may complete their work
// before returning to a scheduling boundary. No checks are inserted implicitly.
func EmitPreemptCheck(dst CodeBuffer, slowOffset int) PreemptCheckFixup {
	b := preemptCheckCode(dst.Offset(), slowOffset)
	f := PreemptCheckFixup{dst.Offset(), len(b.bytes)}
	dst.Write(b.bytes)
	return f
}

// PatchPreemptCheck resolves the client-owned scheduling slow path.
func PatchPreemptCheck(dst CodeBuffer, f PreemptCheckFixup, slowOffset int) {
	b := preemptCheckCode(f.offset, slowOffset)
	if len(b.bytes) != f.length {
		panic("runtime/jit: preempt check patch length changed")
	}
	dst.Patch(f.offset, b.bytes)
}

// EmitYield emits an ordinary Go call to Yield and returns its exact return-PC
// and clobbers. Before this call the client must preserve live clobbered values,
// root all live pointers and provide the call frame and published safepoint map.
// The emitter does not choose polling policy or modify the client's allocator.
func EmitYield(dst CodeBuffer) CallEmission {
	return EmitCall(dst, CallPlan{Entry: reflect.ValueOf(Yield).Pointer()})
}
