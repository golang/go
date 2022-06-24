// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && linux

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

type sigContext struct {
	savedRegs sigcontext
	// sigcontext.fpstate is a pointer, so we need to save
	// the its value with a fpstate1 structure.
	savedFP fpstate1
}

func sigctxtSetContextRegister(ctxt *sigctxt, x uint64) {
	ctxt.regs().rdx = x
}

func sigctxtAtTrapInstruction(ctxt *sigctxt) bool {
	return *(*byte)(unsafe.Pointer(uintptr(ctxt.rip() - 1))) == 0xcc // INT 3
}

func sigctxtStatus(ctxt *sigctxt) uint64 {
	return ctxt.r12()
}

func (h *debugCallHandler) saveSigContext(ctxt *sigctxt) {
	// Push current PC on the stack.
	rsp := ctxt.rsp() - goarch.PtrSize
	*(*uint64)(unsafe.Pointer(uintptr(rsp))) = ctxt.rip()
	ctxt.set_rsp(rsp)
	// Write the argument frame size.
	*(*uintptr)(unsafe.Pointer(uintptr(rsp - 16))) = h.argSize
	// Save current registers.
	h.sigCtxt.savedRegs = *ctxt.regs()
	h.sigCtxt.savedFP = *h.sigCtxt.savedRegs.fpstate
	h.sigCtxt.savedRegs.fpstate = nil
}

// case 0
func (h *debugCallHandler) debugCallRun(ctxt *sigctxt) {
	rsp := ctxt.rsp()
	memmove(unsafe.Pointer(uintptr(rsp)), h.argp, h.argSize)
	if h.regArgs != nil {
		storeRegArgs(ctxt.regs(), h.regArgs)
	}
	// Push return PC.
	rsp -= goarch.PtrSize
	ctxt.set_rsp(rsp)
	// The signal PC is the next PC of the trap instruction.
	*(*uint64)(unsafe.Pointer(uintptr(rsp))) = ctxt.rip()
	// Set PC to call and context register.
	ctxt.set_rip(uint64(h.fv.fn))
	sigctxtSetContextRegister(ctxt, uint64(uintptr(unsafe.Pointer(h.fv))))
}

// case 1
func (h *debugCallHandler) debugCallReturn(ctxt *sigctxt) {
	rsp := ctxt.rsp()
	memmove(h.argp, unsafe.Pointer(uintptr(rsp)), h.argSize)
	if h.regArgs != nil {
		loadRegArgs(h.regArgs, ctxt.regs())
	}
}

// case 2
func (h *debugCallHandler) debugCallPanicOut(ctxt *sigctxt) {
	rsp := ctxt.rsp()
	memmove(unsafe.Pointer(&h.panic), unsafe.Pointer(uintptr(rsp)), 2*goarch.PtrSize)
}

// case 8
func (h *debugCallHandler) debugCallUnsafe(ctxt *sigctxt) {
	rsp := ctxt.rsp()
	reason := *(*string)(unsafe.Pointer(uintptr(rsp)))
	h.err = plainError(reason)
}

// case 16
func (h *debugCallHandler) restoreSigContext(ctxt *sigctxt) {
	// Restore all registers except RIP and RSP.
	rip, rsp := ctxt.rip(), ctxt.rsp()
	fp := ctxt.regs().fpstate
	*ctxt.regs() = h.sigCtxt.savedRegs
	ctxt.regs().fpstate = fp
	*fp = h.sigCtxt.savedFP
	ctxt.set_rip(rip)
	ctxt.set_rsp(rsp)
}

// storeRegArgs sets up argument registers in the signal
// context state from an abi.RegArgs.
//
// Both src and dst must be non-nil.
func storeRegArgs(dst *sigcontext, src *abi.RegArgs) {
	dst.rax = uint64(src.Ints[0])
	dst.rbx = uint64(src.Ints[1])
	dst.rcx = uint64(src.Ints[2])
	dst.rdi = uint64(src.Ints[3])
	dst.rsi = uint64(src.Ints[4])
	dst.r8 = uint64(src.Ints[5])
	dst.r9 = uint64(src.Ints[6])
	dst.r10 = uint64(src.Ints[7])
	dst.r11 = uint64(src.Ints[8])
	for i := range src.Floats {
		dst.fpstate._xmm[i].element[0] = uint32(src.Floats[i] >> 0)
		dst.fpstate._xmm[i].element[1] = uint32(src.Floats[i] >> 32)
	}
}

func loadRegArgs(dst *abi.RegArgs, src *sigcontext) {
	dst.Ints[0] = uintptr(src.rax)
	dst.Ints[1] = uintptr(src.rbx)
	dst.Ints[2] = uintptr(src.rcx)
	dst.Ints[3] = uintptr(src.rdi)
	dst.Ints[4] = uintptr(src.rsi)
	dst.Ints[5] = uintptr(src.r8)
	dst.Ints[6] = uintptr(src.r9)
	dst.Ints[7] = uintptr(src.r10)
	dst.Ints[8] = uintptr(src.r11)
	for i := range dst.Floats {
		dst.Floats[i] = uint64(src.fpstate._xmm[i].element[0]) << 0
		dst.Floats[i] |= uint64(src.fpstate._xmm[i].element[1]) << 32
	}
}
