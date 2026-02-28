// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64 && linux

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

type sigContext struct {
	savedRegs sigcontext
}

func sigctxtSetContextRegister(ctxt *sigctxt, x uint64) {
	ctxt.regs().sc_regs[29] = x
}

func sigctxtAtTrapInstruction(ctxt *sigctxt) bool {
	return *(*uint32)(unsafe.Pointer(ctxt.sigpc())) == 0x002a0000 // BREAK 0
}

func sigctxtStatus(ctxt *sigctxt) uint64 {
	return ctxt.r19()
}

func (h *debugCallHandler) saveSigContext(ctxt *sigctxt) {
	sp := ctxt.sp()
	sp -= goarch.PtrSize
	ctxt.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = ctxt.link() // save the current lr
	ctxt.set_link(ctxt.pc())                              // set new lr to the current pc
	// Write the argument frame size.
	*(*uintptr)(unsafe.Pointer(uintptr(sp - 8))) = h.argSize
	// Save current registers.
	h.sigCtxt.savedRegs = *ctxt.regs()
}

// case 0
func (h *debugCallHandler) debugCallRun(ctxt *sigctxt) {
	sp := ctxt.sp()
	memmove(unsafe.Pointer(uintptr(sp)+8), h.argp, h.argSize)
	if h.regArgs != nil {
		storeRegArgs(ctxt.regs(), h.regArgs)
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
	memmove(h.argp, unsafe.Pointer(uintptr(sp)+8), h.argSize)
	if h.regArgs != nil {
		loadRegArgs(h.regArgs, ctxt.regs())
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
	memmove(unsafe.Pointer(&h.panic), unsafe.Pointer(uintptr(sp)+8), 2*goarch.PtrSize)
	ctxt.set_pc(ctxt.pc() + 4)
}

// case 8
func (h *debugCallHandler) debugCallUnsafe(ctxt *sigctxt) {
	sp := ctxt.sp()
	reason := *(*string)(unsafe.Pointer(uintptr(sp) + 8))
	h.err = plainError(reason)
	ctxt.set_pc(ctxt.pc() + 4)
}

// case 16
func (h *debugCallHandler) restoreSigContext(ctxt *sigctxt) {
	// Restore all registers except for pc and sp
	pc, sp := ctxt.pc(), ctxt.sp()
	*ctxt.regs() = h.sigCtxt.savedRegs
	ctxt.set_pc(pc + 4)
	ctxt.set_sp(sp)
}

func getVal32(base uintptr, off uintptr) uint32 {
	return *(*uint32)(unsafe.Pointer(base + off))
}

func getVal64(base uintptr, off uintptr) uint64 {
	return *(*uint64)(unsafe.Pointer(base + off))
}

func setVal64(base uintptr, off uintptr, val uint64) {
	*(*uint64)(unsafe.Pointer(base + off)) = val
}

// Layout for sigcontext on linux/loong64: arch/loongarch/include/uapi/asm/sigcontext.h
//
//  sc_extcontext |  sctx_info
// ------------------------------------------
//                |  {fpu,lsx,lasx}_context
//                ---------------------------
//                |  sctx_info
//                ---------------------------
//                |  lbt_context
//

const (
	INVALID_MAGIC  uint32 = 0
	FPU_CTX_MAGIC         = 0x46505501
	LSX_CTX_MAGIC         = 0x53580001
	LASX_CTX_MAGIC        = 0x41535801
	LBT_CTX_MAGIC         = 0x42540001
)

const (
	SCTX_INFO_SIZE = 4 + 4 + 8
	FPU_CTX_SIZE   = 8*32 + 8 + 4  // fpu context size
	LSX_CTX_SIZE   = 8*64 + 8 + 4  // lsx context size
	LASX_CTX_SIZE  = 8*128 + 8 + 4 // lasx context size
	LBT_CTX_SIZE   = 8*4 + 4 + 4   // lbt context size
)

// storeRegArgs sets up argument registers in the signal context state
// from an abi.RegArgs.
//
// Both src and dst must be non-nil.
func storeRegArgs(dst *sigcontext, src *abi.RegArgs) {
	// R4..R19 are used to pass int arguments in registers on loong64
	for i := 0; i < abi.IntArgRegs; i++ {
		dst.sc_regs[i+4] = (uint64)(src.Ints[i])
	}

	// F0..F15 are used to pass float arguments in registers on loong64
	offset := (uintptr)(0)
	baseAddr := (uintptr)(unsafe.Pointer(&dst.sc_extcontext))

	for {
		magic := getVal32(baseAddr, offset)
		size := getVal32(baseAddr, offset+4)

		switch magic {
		case INVALID_MAGIC:
			return

		case FPU_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				setVal64(baseAddr, ((uintptr)(i*8) + offset), src.Floats[i])
			}
			return

		case LSX_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				setVal64(baseAddr, ((uintptr)(i*16) + offset), src.Floats[i])
			}
			return

		case LASX_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				setVal64(baseAddr, ((uintptr)(i*32) + offset), src.Floats[i])
			}
			return

		case LBT_CTX_MAGIC:
			offset += uintptr(size)
		}
	}
}

func loadRegArgs(dst *abi.RegArgs, src *sigcontext) {
	// R4..R19 are used to pass int arguments in registers on loong64
	for i := 0; i < abi.IntArgRegs; i++ {
		dst.Ints[i] = uintptr(src.sc_regs[i+4])
	}

	// F0..F15 are used to pass float arguments in registers on loong64
	offset := (uintptr)(0)
	baseAddr := (uintptr)(unsafe.Pointer(&src.sc_extcontext))

	for {
		magic := getVal32(baseAddr, offset)
		size := getVal32(baseAddr, (offset + 4))

		switch magic {
		case INVALID_MAGIC:
			return

		case FPU_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				dst.Floats[i] = getVal64(baseAddr, (uintptr(i*8) + offset))
			}
			return

		case LSX_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				dst.Floats[i] = getVal64(baseAddr, (uintptr(i*16) + offset))
			}
			return

		case LASX_CTX_MAGIC:
			offset += SCTX_INFO_SIZE
			for i := 0; i < abi.FloatArgRegs; i++ {
				dst.Floats[i] = getVal64(baseAddr, (uintptr(i*32) + offset))
			}
			return

		case LBT_CTX_MAGIC:
			offset += uintptr(size)
		}
	}
}
