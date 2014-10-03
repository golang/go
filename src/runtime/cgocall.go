// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo call and callback support.
//
// To call into the C function f from Go, the cgo-generated code calls
// runtime.cgocall(_cgo_Cfunc_f, frame), where _cgo_Cfunc_f is a
// gcc-compiled function written by cgo.
//
// runtime.cgocall (below) locks g to m, calls entersyscall
// so as not to block other goroutines or the garbage collector,
// and then calls runtime.asmcgocall(_cgo_Cfunc_f, frame).
//
// runtime.asmcgocall (in asm_$GOARCH.s) switches to the m->g0 stack
// (assumed to be an operating system-allocated stack, so safe to run
// gcc-compiled code on) and calls _cgo_Cfunc_f(frame).
//
// _cgo_Cfunc_f invokes the actual C function f with arguments
// taken from the frame structure, records the results in the frame,
// and returns to runtime.asmcgocall.
//
// After it regains control, runtime.asmcgocall switches back to the
// original g (m->curg)'s stack and returns to runtime.cgocall.
//
// After it regains control, runtime.cgocall calls exitsyscall, which blocks
// until this m can run Go code without violating the $GOMAXPROCS limit,
// and then unlocks g from m.
//
// The above description skipped over the possibility of the gcc-compiled
// function f calling back into Go.  If that happens, we continue down
// the rabbit hole during the execution of f.
//
// To make it possible for gcc-compiled C code to call a Go function p.GoF,
// cgo writes a gcc-compiled function named GoF (not p.GoF, since gcc doesn't
// know about packages).  The gcc-compiled C function f calls GoF.
//
// GoF calls crosscall2(_cgoexp_GoF, frame, framesize).  Crosscall2
// (in cgo/gcc_$GOARCH.S, a gcc-compiled assembly file) is a two-argument
// adapter from the gcc function call ABI to the 6c function call ABI.
// It is called from gcc to call 6c functions.  In this case it calls
// _cgoexp_GoF(frame, framesize), still running on m->g0's stack
// and outside the $GOMAXPROCS limit.  Thus, this code cannot yet
// call arbitrary Go code directly and must be careful not to allocate
// memory or use up m->g0's stack.
//
// _cgoexp_GoF calls runtime.cgocallback(p.GoF, frame, framesize).
// (The reason for having _cgoexp_GoF instead of writing a crosscall3
// to make this call directly is that _cgoexp_GoF, because it is compiled
// with 6c instead of gcc, can refer to dotted names like
// runtime.cgocallback and p.GoF.)
//
// runtime.cgocallback (in asm_$GOARCH.s) switches from m->g0's
// stack to the original g (m->curg)'s stack, on which it calls
// runtime.cgocallbackg(p.GoF, frame, framesize).
// As part of the stack switch, runtime.cgocallback saves the current
// SP as m->g0->sched.sp, so that any use of m->g0's stack during the
// execution of the callback will be done below the existing stack frames.
// Before overwriting m->g0->sched.sp, it pushes the old value on the
// m->g0 stack, so that it can be restored later.
//
// runtime.cgocallbackg (below) is now running on a real goroutine
// stack (not an m->g0 stack).  First it calls runtime.exitsyscall, which will
// block until the $GOMAXPROCS limit allows running this goroutine.
// Once exitsyscall has returned, it is safe to do things like call the memory
// allocator or invoke the Go callback function p.GoF.  runtime.cgocallbackg
// first defers a function to unwind m->g0.sched.sp, so that if p.GoF
// panics, m->g0.sched.sp will be restored to its old value: the m->g0 stack
// and the m->curg stack will be unwound in lock step.
// Then it calls p.GoF.  Finally it pops but does not execute the deferred
// function, calls runtime.entersyscall, and returns to runtime.cgocallback.
//
// After it regains control, runtime.cgocallback switches back to
// m->g0's stack (the pointer is still in m->g0.sched.sp), restores the old
// m->g0.sched.sp value from the stack, and returns to _cgoexp_GoF.
//
// _cgoexp_GoF immediately returns to crosscall2, which restores the
// callee-save registers for gcc and returns to GoF, which returns to f.

package runtime

import "unsafe"

// Call from Go to C.
//go:nosplit
func cgocall(fn, arg unsafe.Pointer) {
	cgocall_errno(fn, arg)
}

//go:nosplit
func cgocall_errno(fn, arg unsafe.Pointer) int32 {
	if !iscgo && GOOS != "solaris" && GOOS != "windows" {
		gothrow("cgocall unavailable")
	}

	if fn == nil {
		gothrow("cgocall nil")
	}

	if raceenabled {
		racereleasemerge(unsafe.Pointer(&racecgosync))
	}

	// Create an extra M for callbacks on threads not created by Go on first cgo call.
	if needextram == 1 && cas(&needextram, 1, 0) {
		onM(newextram)
	}

	/*
	 * Lock g to m to ensure we stay on the same stack if we do a
	 * cgo callback. Add entry to defer stack in case of panic.
	 */
	lockOSThread()
	mp := getg().m
	mp.ncgocall++
	mp.ncgo++
	defer endcgo(mp)

	/*
	 * Announce we are entering a system call
	 * so that the scheduler knows to create another
	 * M to run goroutines while we are in the
	 * foreign code.
	 *
	 * The call to asmcgocall is guaranteed not to
	 * split the stack and does not allocate memory,
	 * so it is safe to call while "in a system call", outside
	 * the $GOMAXPROCS accounting.
	 */
	entersyscall()
	errno := asmcgocall_errno(fn, arg)
	exitsyscall()

	return errno
}

//go:nosplit
func endcgo(mp *m) {
	mp.ncgo--
	if mp.ncgo == 0 {
		// We are going back to Go and are not in a recursive
		// call.  Let the GC collect any memory allocated via
		// _cgo_allocate that is no longer referenced.
		mp.cgomal = nil
	}

	if raceenabled {
		raceacquire(unsafe.Pointer(&racecgosync))
	}

	unlockOSThread() // invalidates mp
}

// Helper functions for cgo code.

// Filled by schedinit from corresponding C variables,
// which are in turn filled in by dynamic linker when Cgo is available.
var cgoMalloc, cgoFree unsafe.Pointer

func cmalloc(n uintptr) unsafe.Pointer {
	var args struct {
		n   uint64
		ret unsafe.Pointer
	}
	args.n = uint64(n)
	cgocall(cgoMalloc, unsafe.Pointer(&args))
	if args.ret == nil {
		gothrow("C malloc failed")
	}
	return args.ret
}

func cfree(p unsafe.Pointer) {
	cgocall(cgoFree, p)
}

// Call from C back to Go.
//go:nosplit
func cgocallbackg() {
	gp := getg()
	if gp != gp.m.curg {
		println("runtime: bad g in cgocallback")
		exit(2)
	}

	// entersyscall saves the caller's SP to allow the GC to trace the Go
	// stack. However, since we're returning to an earlier stack frame and
	// need to pair with the entersyscall() call made by cgocall, we must
	// save syscall* and let reentersyscall restore them.
	savedsp := unsafe.Pointer(gp.syscallsp)
	savedpc := gp.syscallpc
	exitsyscall() // coming out of cgo call
	cgocallbackg1()
	// going back to cgo call
	reentersyscall(savedpc, savedsp)
}

func cgocallbackg1() {
	gp := getg()
	if gp.m.needextram {
		gp.m.needextram = false
		onM(newextram)
	}

	// Add entry to defer stack in case of panic.
	restore := true
	defer unwindm(&restore)

	if raceenabled {
		raceacquire(unsafe.Pointer(&racecgosync))
	}

	type args struct {
		fn      *funcval
		arg     unsafe.Pointer
		argsize uintptr
	}
	var cb *args

	// Location of callback arguments depends on stack frame layout
	// and size of stack frame of cgocallback_gofunc.
	sp := gp.m.g0.sched.sp
	switch GOARCH {
	default:
		gothrow("cgocallbackg is unimplemented on arch")
	case "arm":
		// On arm, stack frame is two words and there's a saved LR between
		// SP and the stack frame and between the stack frame and the arguments.
		cb = (*args)(unsafe.Pointer(sp + 4*ptrSize))
	case "amd64":
		// On amd64, stack frame is one word, plus caller PC.
		cb = (*args)(unsafe.Pointer(sp + 2*ptrSize))
	case "386":
		// On 386, stack frame is three words, plus caller PC.
		cb = (*args)(unsafe.Pointer(sp + 4*ptrSize))
	}

	// Invoke callback.
	reflectcall(unsafe.Pointer(cb.fn), unsafe.Pointer(cb.arg), uint32(cb.argsize), 0)

	if raceenabled {
		racereleasemerge(unsafe.Pointer(&racecgosync))
	}

	// Do not unwind m->g0->sched.sp.
	// Our caller, cgocallback, will do that.
	restore = false
}

func unwindm(restore *bool) {
	if !*restore {
		return
	}
	// Restore sp saved by cgocallback during
	// unwind of g's stack (see comment at top of file).
	mp := acquirem()
	sched := &mp.g0.sched
	switch GOARCH {
	default:
		gothrow("unwindm not implemented")
	case "386", "amd64":
		sched.sp = *(*uintptr)(unsafe.Pointer(sched.sp))
	case "arm":
		sched.sp = *(*uintptr)(unsafe.Pointer(sched.sp + 4))
	}
	releasem(mp)
}

// called from assembly
func badcgocallback() {
	gothrow("misaligned stack in cgocallback")
}

// called from (incomplete) assembly
func cgounimpl() {
	gothrow("cgo not implemented")
}

var racecgosync uint64 // represents possible synchronization in C code
