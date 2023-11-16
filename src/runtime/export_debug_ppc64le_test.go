// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le && linux

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"math"
	"unsafe"
)

type sigContext struct {
	savedRegs sigcontext
}

func sigctxtSetContextRegister(ctxt *sigctxt, x uint64) {
	ctxt.regs().gpr[11] = x
}

func sigctxtAtTrapInstruction(ctxt *sigctxt) bool {
	return *(*uint32)(unsafe.Pointer(ctxt.sigpc())) == 0x7fe00008 // Trap
}

func sigctxtStatus(ctxt *sigctxt) uint64 {
	return ctxt.r20()
}

func (h *debugCallHandler) saveSigContext(ctxt *sigctxt) {
	sp := ctxt.sp()
	sp -= 4 * goarch.PtrSize
	ctxt.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = ctxt.link() // save the current lr
	ctxt.set_link(ctxt.pc())                              // set new lr to the current pc
	// Write the argument frame size.
	*(*uintptr)(unsafe.Pointer(uintptr(sp - 32))) = h.argSize
	// Save current registers.
	h.sigCtxt.savedRegs = *ctxt.cregs()
}

// case 0
func (h *debugCallHandler) debugCallRun(ctxt *sigctxt) {
	sp := ctxt.sp()
	memmove(unsafe.Pointer(uintptr(sp)+32), h.argp, h.argSize)
	if h.regArgs != nil {
		storeRegArgs(ctxt.cregs(), h.regArgs)
	}
	// Push return PC, which should be the signal PC+4, because
	// the signal PC is the PC of the trap instruction itself.
	ctxt.set_link(ctxt.pc() + 4)
	// Set PC to call and context register.
	ctxt.set_pc(uint64(h.fv.fn))
	sigctxtSetContextRegister(ctxt, uint64(uintptr(unsafe.Pointer(h.fv))))
}

// case 1
func (h *debugCallHandler) debugCallReturn(ctxt *sigctxt) {
	sp := ctxt.sp()
	memmove(h.argp, unsafe.Pointer(uintptr(sp)+32), h.argSize)
	if h.regArgs != nil {
		loadRegArgs(h.regArgs, ctxt.cregs())
	}
	// Restore the old lr from *sp
	olr := *(*uint64)(unsafe.Pointer(uintptr(sp)))
	ctxt.set_link(olr)
	pc := ctxt.pc()
	ctxt.set_pc(pc + 4) // step to next instruction
}

// case 2
func (h *debugCallHandler) debugCallPanicOut(ctxt *sigctxt) {
	sp := ctxt.sp()
	memmove(unsafe.Pointer(&h.panic), unsafe.Pointer(uintptr(sp)+32), 2*goarch.PtrSize)
	ctxt.set_pc(ctxt.pc() + 4)
}

// case 8
func (h *debugCallHandler) debugCallUnsafe(ctxt *sigctxt) {
	sp := ctxt.sp()
	reason := *(*string)(unsafe.Pointer(uintptr(sp) + 40))
	h.err = plainError(reason)
	ctxt.set_pc(ctxt.pc() + 4)
}

// case 16
func (h *debugCallHandler) restoreSigContext(ctxt *sigctxt) {
	// Restore all registers except for pc and sp
	pc, sp := ctxt.pc(), ctxt.sp()
	*ctxt.cregs() = h.sigCtxt.savedRegs
	ctxt.set_pc(pc + 4)
	ctxt.set_sp(sp)
}

// storeRegArgs sets up argument registers in the signal
// context state from an abi.RegArgs.
//
// Both src and dst must be non-nil.
func storeRegArgs(dst *sigcontext, src *abi.RegArgs) {
	// Gprs R3..R10, R14..R17 are used to pass int arguments in registers on PPC64
	for i := 0; i < 12; i++ {
		if i > 7 {
			dst.gp_regs[i+6] = uint64(src.Ints[i])
		} else {
			dst.gp_regs[i+3] = uint64(src.Ints[i])
		}
	}
	// Fprs F1..F13 are used to pass float arguments in registers on PPC64
	for i := 0; i < 12; i++ {
		dst.fp_regs[i+1] = math.Float64frombits(src.Floats[i])
	}

}

func loadRegArgs(dst *abi.RegArgs, src *sigcontext) {
	// Gprs R3..R10, R14..R17 are used to pass int arguments in registers on PPC64
	for i := range [12]int{} {
		if i > 7 {
			dst.Ints[i] = uintptr(src.gp_regs[i+6])
		} else {
			dst.Ints[i] = uintptr(src.gp_regs[i+3])
		}
	}
	// Fprs F1..F13 are used to pass float arguments in registers on PPC64
	for i := range [12]int{} {
		dst.Floats[i] = math.Float64bits(src.fp_regs[i+1])
	}

}
