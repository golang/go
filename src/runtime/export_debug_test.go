// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64
// +build linux

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

// InjectDebugCall injects a debugger call to fn into g. args must be
// a pointer to a valid call frame (including arguments and return
// space) for fn, or nil. tkill must be a function that will send
// SIGTRAP to thread ID tid. gp must be locked to its OS thread and
// running.
//
// On success, InjectDebugCall returns the panic value of fn or nil.
// If fn did not panic, its results will be available in args.
func InjectDebugCall(gp *g, fn, args interface{}, tkill func(tid int) error) (interface{}, error) {
	if gp.lockedm == 0 {
		return nil, plainError("goroutine not locked to thread")
	}

	tid := int(gp.lockedm.ptr().procid)
	if tid == 0 {
		return nil, plainError("missing tid")
	}

	f := efaceOf(&fn)
	if f._type == nil || f._type.kind&kindMask != kindFunc {
		return nil, plainError("fn must be a function")
	}
	fv := (*funcval)(f.data)

	a := efaceOf(&args)
	if a._type != nil && a._type.kind&kindMask != kindPtr {
		return nil, plainError("args must be a pointer or nil")
	}
	argp := a.data
	var argSize uintptr
	if argp != nil {
		argSize = (*ptrtype)(unsafe.Pointer(a._type)).elem.size
	}

	h := new(debugCallHandler)
	h.gp = gp
	h.fv, h.argp, h.argSize = fv, argp, argSize
	h.handleF = h.handle // Avoid allocating closure during signal
	noteclear(&h.done)

	defer func() { testSigtrap = nil }()
	testSigtrap = h.inject
	if err := tkill(tid); err != nil {
		return nil, err
	}
	// Wait for completion.
	notetsleepg(&h.done, -1)
	if len(h.err) != 0 {
		return nil, h.err
	}
	return h.panic, nil
}

type debugCallHandler struct {
	gp      *g
	fv      *funcval
	argp    unsafe.Pointer
	argSize uintptr
	panic   interface{}

	handleF func(info *siginfo, ctxt *sigctxt, gp2 *g) bool

	err       plainError
	done      note
	savedRegs sigcontext
	savedFP   fpstate1
}

func (h *debugCallHandler) inject(info *siginfo, ctxt *sigctxt, gp2 *g) bool {
	switch h.gp.atomicstatus {
	case _Grunning:
		if getg().m != h.gp.m {
			println("trap on wrong M", getg().m, h.gp.m)
			return false
		}
		// Push current PC on the stack.
		rsp := ctxt.rsp() - sys.PtrSize
		*(*uint64)(unsafe.Pointer(uintptr(rsp))) = ctxt.rip()
		ctxt.set_rsp(rsp)
		// Write the argument frame size.
		*(*uintptr)(unsafe.Pointer(uintptr(rsp - 16))) = h.argSize
		// Save current registers.
		h.savedRegs = *ctxt.regs()
		h.savedFP = *h.savedRegs.fpstate
		h.savedRegs.fpstate = nil
		// Set PC to debugCallV1.
		ctxt.set_rip(uint64(funcPC(debugCallV1)))
	default:
		h.err = plainError("goroutine in unexpected state at call inject")
		return true
	}
	// Switch to the debugCall protocol and resume execution.
	testSigtrap = h.handleF
	return true
}

func (h *debugCallHandler) handle(info *siginfo, ctxt *sigctxt, gp2 *g) bool {
	// Sanity check.
	if getg().m != h.gp.m {
		println("trap on wrong M", getg().m, h.gp.m)
		return false
	}
	f := findfunc(uintptr(ctxt.rip()))
	if !(hasprefix(funcname(f), "runtime.debugCall") || hasprefix(funcname(f), "debugCall")) {
		println("trap in unknown function", funcname(f))
		return false
	}
	if *(*byte)(unsafe.Pointer(uintptr(ctxt.rip() - 1))) != 0xcc {
		println("trap at non-INT3 instruction pc =", hex(ctxt.rip()))
		return false
	}

	switch status := ctxt.rax(); status {
	case 0:
		// Frame is ready. Copy the arguments to the frame.
		sp := ctxt.rsp()
		memmove(unsafe.Pointer(uintptr(sp)), h.argp, h.argSize)
		// Push return PC.
		sp -= sys.PtrSize
		ctxt.set_rsp(sp)
		*(*uint64)(unsafe.Pointer(uintptr(sp))) = ctxt.rip()
		// Set PC to call and context register.
		ctxt.set_rip(uint64(h.fv.fn))
		ctxt.regs().rcx = uint64(uintptr(unsafe.Pointer(h.fv)))
	case 1:
		// Function returned. Copy frame back out.
		sp := ctxt.rsp()
		memmove(h.argp, unsafe.Pointer(uintptr(sp)), h.argSize)
	case 2:
		// Function panicked. Copy panic out.
		sp := ctxt.rsp()
		memmove(unsafe.Pointer(&h.panic), unsafe.Pointer(uintptr(sp)), 2*sys.PtrSize)
	case 8:
		// Call isn't safe. Get the reason.
		sp := ctxt.rsp()
		reason := *(*string)(unsafe.Pointer(uintptr(sp)))
		h.err = plainError(reason)
	case 16:
		// Restore all registers except RIP and RSP.
		rip, rsp := ctxt.rip(), ctxt.rsp()
		fp := ctxt.regs().fpstate
		*ctxt.regs() = h.savedRegs
		ctxt.regs().fpstate = fp
		*fp = h.savedFP
		ctxt.set_rip(rip)
		ctxt.set_rsp(rsp)
		// Done
		notewakeup(&h.done)
	default:
		h.err = plainError("unexpected debugCallV1 status")
	}
	// Resume execution.
	return true
}
