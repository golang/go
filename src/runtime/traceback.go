// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/bytealg"
	"internal/goarch"
	"runtime/internal/sys"
	"unsafe"
)

// The code in this file implements stack trace walking for all architectures.
// The most important fact about a given architecture is whether it uses a link register.
// On systems with link registers, the prologue for a non-leaf function stores the
// incoming value of LR at the bottom of the newly allocated stack frame.
// On systems without link registers (x86), the architecture pushes a return PC during
// the call instruction, so the return PC ends up above the stack frame.
// In this file, the return PC is always called LR, no matter how it was found.

const usesLR = sys.MinFrameSize > 0

const (
	// tracebackInnerFrames is the number of innermost frames to print in a
	// stack trace. The total maximum frames is tracebackInnerFrames +
	// tracebackOuterFrames.
	tracebackInnerFrames = 50

	// tracebackOuterFrames is the number of outermost frames to print in a
	// stack trace.
	tracebackOuterFrames = 50
)

// unwindFlags control the behavior of various unwinders.
type unwindFlags uint8

const (
	// unwindPrintErrors indicates that if unwinding encounters an error, it
	// should print a message and stop without throwing. This is used for things
	// like stack printing, where it's better to get incomplete information than
	// to crash. This is also used in situations where everything may not be
	// stopped nicely and the stack walk may not be able to complete, such as
	// during profiling signals or during a crash.
	//
	// If neither unwindPrintErrors or unwindSilentErrors are set, unwinding
	// performs extra consistency checks and throws on any error.
	//
	// Note that there are a small number of fatal situations that will throw
	// regardless of unwindPrintErrors or unwindSilentErrors.
	unwindPrintErrors unwindFlags = 1 << iota

	// unwindSilentErrors silently ignores errors during unwinding.
	unwindSilentErrors

	// unwindTrap indicates that the initial PC and SP are from a trap, not a
	// return PC from a call.
	//
	// The unwindTrap flag is updated during unwinding. If set, frame.pc is the
	// address of a faulting instruction instead of the return address of a
	// call. It also means the liveness at pc may not be known.
	//
	// TODO: Distinguish frame.continpc, which is really the stack map PC, from
	// the actual continuation PC, which is computed differently depending on
	// this flag and a few other things.
	unwindTrap

	// unwindJumpStack indicates that, if the traceback is on a system stack, it
	// should resume tracing at the user stack when the system stack is
	// exhausted.
	unwindJumpStack
)

// An unwinder iterates the physical stack frames of a Go sack.
//
// Typical use of an unwinder looks like:
//
//	var u unwinder
//	for u.init(gp, 0); u.valid(); u.next() {
//		// ... use frame info in u ...
//	}
//
// Implementation note: This is carefully structured to be pointer-free because
// tracebacks happen in places that disallow write barriers (e.g., signals).
// Even if this is stack-allocated, its pointer-receiver methods don't know that
// their receiver is on the stack, so they still emit write barriers. Here we
// address that by carefully avoiding any pointers in this type. Another
// approach would be to split this into a mutable part that's passed by pointer
// but contains no pointers itself and an immutable part that's passed and
// returned by value and can contain pointers. We could potentially hide that
// we're doing that in trivial methods that are inlined into the caller that has
// the stack allocation, but that's fragile.
type unwinder struct {
	// frame is the current physical stack frame, or all 0s if
	// there is no frame.
	frame stkframe

	// g is the G who's stack is being unwound. If the
	// unwindJumpStack flag is set and the unwinder jumps stacks,
	// this will be different from the initial G.
	g guintptr

	// cgoCtxt is the index into g.cgoCtxt of the next frame on the cgo stack.
	// The cgo stack is unwound in tandem with the Go stack as we find marker frames.
	cgoCtxt int

	// calleeFuncID is the function ID of the caller of the current
	// frame.
	calleeFuncID abi.FuncID

	// flags are the flags to this unwind. Some of these are updated as we
	// unwind (see the flags documentation).
	flags unwindFlags
}

// init initializes u to start unwinding gp's stack and positions the
// iterator on gp's innermost frame. gp must not be the current G.
//
// A single unwinder can be reused for multiple unwinds.
func (u *unwinder) init(gp *g, flags unwindFlags) {
	// Implementation note: This starts the iterator on the first frame and we
	// provide a "valid" method. Alternatively, this could start in a "before
	// the first frame" state and "next" could return whether it was able to
	// move to the next frame, but that's both more awkward to use in a "for"
	// loop and is harder to implement because we have to do things differently
	// for the first frame.
	u.initAt(^uintptr(0), ^uintptr(0), ^uintptr(0), gp, flags)
}

func (u *unwinder) initAt(pc0, sp0, lr0 uintptr, gp *g, flags unwindFlags) {
	// Don't call this "g"; it's too easy get "g" and "gp" confused.
	if ourg := getg(); ourg == gp && ourg == ourg.m.curg {
		// The starting sp has been passed in as a uintptr, and the caller may
		// have other uintptr-typed stack references as well.
		// If during one of the calls that got us here or during one of the
		// callbacks below the stack must be grown, all these uintptr references
		// to the stack will not be updated, and traceback will continue
		// to inspect the old stack memory, which may no longer be valid.
		// Even if all the variables were updated correctly, it is not clear that
		// we want to expose a traceback that begins on one stack and ends
		// on another stack. That could confuse callers quite a bit.
		// Instead, we require that initAt and any other function that
		// accepts an sp for the current goroutine (typically obtained by
		// calling getcallersp) must not run on that goroutine's stack but
		// instead on the g0 stack.
		throw("cannot trace user goroutine on its own stack")
	}

	if pc0 == ^uintptr(0) && sp0 == ^uintptr(0) { // Signal to fetch saved values from gp.
		if gp.syscallsp != 0 {
			pc0 = gp.syscallpc
			sp0 = gp.syscallsp
			if usesLR {
				lr0 = 0
			}
		} else {
			pc0 = gp.sched.pc
			sp0 = gp.sched.sp
			if usesLR {
				lr0 = gp.sched.lr
			}
		}
	}

	var frame stkframe
	frame.pc = pc0
	frame.sp = sp0
	if usesLR {
		frame.lr = lr0
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if frame.pc == 0 {
		if usesLR {
			frame.pc = *(*uintptr)(unsafe.Pointer(frame.sp))
			frame.lr = 0
		} else {
			frame.pc = *(*uintptr)(unsafe.Pointer(frame.sp))
			frame.sp += goarch.PtrSize
		}
	}

	// runtime/internal/atomic functions call into kernel helpers on
	// arm < 7. See runtime/internal/atomic/sys_linux_arm.s.
	//
	// Start in the caller's frame.
	if GOARCH == "arm" && goarm < 7 && GOOS == "linux" && frame.pc&0xffff0000 == 0xffff0000 {
		// Note that the calls are simple BL without pushing the return
		// address, so we use LR directly.
		//
		// The kernel helpers are frameless leaf functions, so SP and
		// LR are not touched.
		frame.pc = frame.lr
		frame.lr = 0
	}

	f := findfunc(frame.pc)
	if !f.valid() {
		if flags&unwindSilentErrors == 0 {
			print("runtime: g ", gp.goid, " gp=", gp, ": unknown pc ", hex(frame.pc), "\n")
			tracebackHexdump(gp.stack, &frame, 0)
		}
		if flags&(unwindPrintErrors|unwindSilentErrors) == 0 {
			throw("unknown pc")
		}
		*u = unwinder{}
		return
	}
	frame.fn = f

	// Populate the unwinder.
	*u = unwinder{
		frame:        frame,
		g:            gp.guintptr(),
		cgoCtxt:      len(gp.cgoCtxt) - 1,
		calleeFuncID: abi.FuncIDNormal,
		flags:        flags,
	}

	isSyscall := frame.pc == pc0 && frame.sp == sp0 && pc0 == gp.syscallpc && sp0 == gp.syscallsp
	u.resolveInternal(true, isSyscall)
}

func (u *unwinder) valid() bool {
	return u.frame.pc != 0
}

// resolveInternal fills in u.frame based on u.frame.fn, pc, and sp.
//
// innermost indicates that this is the first resolve on this stack. If
// innermost is set, isSyscall indicates that the PC/SP was retrieved from
// gp.syscall*; this is otherwise ignored.
//
// On entry, u.frame contains:
//   - fn is the running function.
//   - pc is the PC in the running function.
//   - sp is the stack pointer at that program counter.
//   - For the innermost frame on LR machines, lr is the program counter that called fn.
//
// On return, u.frame contains:
//   - fp is the stack pointer of the caller.
//   - lr is the program counter that called fn.
//   - varp, argp, and continpc are populated for the current frame.
//
// If fn is a stack-jumping function, resolveInternal can change the entire
// frame state to follow that stack jump.
//
// This is internal to unwinder.
func (u *unwinder) resolveInternal(innermost, isSyscall bool) {
	frame := &u.frame
	gp := u.g.ptr()

	f := frame.fn
	if f.pcsp == 0 {
		// No frame information, must be external function, like race support.
		// See golang.org/issue/13568.
		u.finishInternal()
		return
	}

	// Compute function info flags.
	flag := f.flag
	if f.funcID == abi.FuncID_cgocallback {
		// cgocallback does write SP to switch from the g0 to the curg stack,
		// but it carefully arranges that during the transition BOTH stacks
		// have cgocallback frame valid for unwinding through.
		// So we don't need to exclude it with the other SP-writing functions.
		flag &^= abi.FuncFlagSPWrite
	}
	if isSyscall {
		// Some Syscall functions write to SP, but they do so only after
		// saving the entry PC/SP using entersyscall.
		// Since we are using the entry PC/SP, the later SP write doesn't matter.
		flag &^= abi.FuncFlagSPWrite
	}

	// Found an actual function.
	// Derive frame pointer.
	if frame.fp == 0 {
		// Jump over system stack transitions. If we're on g0 and there's a user
		// goroutine, try to jump. Otherwise this is a regular call.
		// We also defensively check that this won't switch M's on us,
		// which could happen at critical points in the scheduler.
		// This ensures gp.m doesn't change from a stack jump.
		if u.flags&unwindJumpStack != 0 && gp == gp.m.g0 && gp.m.curg != nil && gp.m.curg.m == gp.m {
			switch f.funcID {
			case abi.FuncID_morestack:
				// morestack does not return normally -- newstack()
				// gogo's to curg.sched. Match that.
				// This keeps morestack() from showing up in the backtrace,
				// but that makes some sense since it'll never be returned
				// to.
				gp = gp.m.curg
				u.g.set(gp)
				frame.pc = gp.sched.pc
				frame.fn = findfunc(frame.pc)
				f = frame.fn
				flag = f.flag
				frame.lr = gp.sched.lr
				frame.sp = gp.sched.sp
				u.cgoCtxt = len(gp.cgoCtxt) - 1
			case abi.FuncID_systemstack:
				// systemstack returns normally, so just follow the
				// stack transition.
				if usesLR && funcspdelta(f, frame.pc) == 0 {
					// We're at the function prologue and the stack
					// switch hasn't happened, or epilogue where we're
					// about to return. Just unwind normally.
					// Do this only on LR machines because on x86
					// systemstack doesn't have an SP delta (the CALL
					// instruction opens the frame), therefore no way
					// to check.
					flag &^= abi.FuncFlagSPWrite
					break
				}
				gp = gp.m.curg
				u.g.set(gp)
				frame.sp = gp.sched.sp
				u.cgoCtxt = len(gp.cgoCtxt) - 1
				flag &^= abi.FuncFlagSPWrite
			}
		}
		frame.fp = frame.sp + uintptr(funcspdelta(f, frame.pc))
		if !usesLR {
			// On x86, call instruction pushes return PC before entering new function.
			frame.fp += goarch.PtrSize
		}
	}

	// Derive link register.
	if flag&abi.FuncFlagTopFrame != 0 {
		// This function marks the top of the stack. Stop the traceback.
		frame.lr = 0
	} else if flag&abi.FuncFlagSPWrite != 0 && (!innermost || u.flags&(unwindPrintErrors|unwindSilentErrors) != 0) {
		// The function we are in does a write to SP that we don't know
		// how to encode in the spdelta table. Examples include context
		// switch routines like runtime.gogo but also any code that switches
		// to the g0 stack to run host C code.
		// We can't reliably unwind the SP (we might not even be on
		// the stack we think we are), so stop the traceback here.
		//
		// The one exception (encoded in the complex condition above) is that
		// we assume if we're doing a precise traceback, and this is the
		// innermost frame, that the SPWRITE function voluntarily preempted itself on entry
		// during the stack growth check. In that case, the function has
		// not yet had a chance to do any writes to SP and is safe to unwind.
		// isAsyncSafePoint does not allow assembly functions to be async preempted,
		// and preemptPark double-checks that SPWRITE functions are not async preempted.
		// So for GC stack traversal, we can safely ignore SPWRITE for the innermost frame,
		// but farther up the stack we'd better not find any.
		// This is somewhat imprecise because we're just guessing that we're in the stack
		// growth check. It would be better if SPWRITE were encoded in the spdelta
		// table so we would know for sure that we were still in safe code.
		//
		// uSE uPE inn | action
		//  T   _   _  | frame.lr = 0
		//  F   T   _  | frame.lr = 0
		//  F   F   F  | print; panic
		//  F   F   T  | ignore SPWrite
		if u.flags&(unwindPrintErrors|unwindSilentErrors) == 0 && !innermost {
			println("traceback: unexpected SPWRITE function", funcname(f))
			throw("traceback")
		}
		frame.lr = 0
	} else {
		var lrPtr uintptr
		if usesLR {
			if innermost && frame.sp < frame.fp || frame.lr == 0 {
				lrPtr = frame.sp
				frame.lr = *(*uintptr)(unsafe.Pointer(lrPtr))
			}
		} else {
			if frame.lr == 0 {
				lrPtr = frame.fp - goarch.PtrSize
				frame.lr = *(*uintptr)(unsafe.Pointer(lrPtr))
			}
		}
	}

	frame.varp = frame.fp
	if !usesLR {
		// On x86, call instruction pushes return PC before entering new function.
		frame.varp -= goarch.PtrSize
	}

	// For architectures with frame pointers, if there's
	// a frame, then there's a saved frame pointer here.
	//
	// NOTE: This code is not as general as it looks.
	// On x86, the ABI is to save the frame pointer word at the
	// top of the stack frame, so we have to back down over it.
	// On arm64, the frame pointer should be at the bottom of
	// the stack (with R29 (aka FP) = RSP), in which case we would
	// not want to do the subtraction here. But we started out without
	// any frame pointer, and when we wanted to add it, we didn't
	// want to break all the assembly doing direct writes to 8(RSP)
	// to set the first parameter to a called function.
	// So we decided to write the FP link *below* the stack pointer
	// (with R29 = RSP - 8 in Go functions).
	// This is technically ABI-compatible but not standard.
	// And it happens to end up mimicking the x86 layout.
	// Other architectures may make different decisions.
	if frame.varp > frame.sp && framepointer_enabled {
		frame.varp -= goarch.PtrSize
	}

	frame.argp = frame.fp + sys.MinFrameSize

	// Determine frame's 'continuation PC', where it can continue.
	// Normally this is the return address on the stack, but if sigpanic
	// is immediately below this function on the stack, then the frame
	// stopped executing due to a trap, and frame.pc is probably not
	// a safe point for looking up liveness information. In this panicking case,
	// the function either doesn't return at all (if it has no defers or if the
	// defers do not recover) or it returns from one of the calls to
	// deferproc a second time (if the corresponding deferred func recovers).
	// In the latter case, use a deferreturn call site as the continuation pc.
	frame.continpc = frame.pc
	if u.calleeFuncID == abi.FuncID_sigpanic {
		if frame.fn.deferreturn != 0 {
			frame.continpc = frame.fn.entry() + uintptr(frame.fn.deferreturn) + 1
			// Note: this may perhaps keep return variables alive longer than
			// strictly necessary, as we are using "function has a defer statement"
			// as a proxy for "function actually deferred something". It seems
			// to be a minor drawback. (We used to actually look through the
			// gp._defer for a defer corresponding to this function, but that
			// is hard to do with defer records on the stack during a stack copy.)
			// Note: the +1 is to offset the -1 that
			// stack.go:getStackMap does to back up a return
			// address make sure the pc is in the CALL instruction.
		} else {
			frame.continpc = 0
		}
	}
}

func (u *unwinder) next() {
	frame := &u.frame
	f := frame.fn
	gp := u.g.ptr()

	// Do not unwind past the bottom of the stack.
	if frame.lr == 0 {
		u.finishInternal()
		return
	}
	flr := findfunc(frame.lr)
	if !flr.valid() {
		// This happens if you get a profiling interrupt at just the wrong time.
		// In that context it is okay to stop early.
		// But if no error flags are set, we're doing a garbage collection and must
		// get everything, so crash loudly.
		fail := u.flags&(unwindPrintErrors|unwindSilentErrors) == 0
		doPrint := u.flags&unwindSilentErrors == 0
		if doPrint && gp.m.incgo && f.funcID == abi.FuncID_sigpanic {
			// We can inject sigpanic
			// calls directly into C code,
			// in which case we'll see a C
			// return PC. Don't complain.
			doPrint = false
		}
		if fail || doPrint {
			print("runtime: g ", gp.goid, ": unexpected return pc for ", funcname(f), " called from ", hex(frame.lr), "\n")
			tracebackHexdump(gp.stack, frame, 0)
		}
		if fail {
			throw("unknown caller pc")
		}
		frame.lr = 0
		u.finishInternal()
		return
	}

	if frame.pc == frame.lr && frame.sp == frame.fp {
		// If the next frame is identical to the current frame, we cannot make progress.
		print("runtime: traceback stuck. pc=", hex(frame.pc), " sp=", hex(frame.sp), "\n")
		tracebackHexdump(gp.stack, frame, frame.sp)
		throw("traceback stuck")
	}

	injectedCall := f.funcID == abi.FuncID_sigpanic || f.funcID == abi.FuncID_asyncPreempt || f.funcID == abi.FuncID_debugCallV2
	if injectedCall {
		u.flags |= unwindTrap
	} else {
		u.flags &^= unwindTrap
	}

	// Unwind to next frame.
	u.calleeFuncID = f.funcID
	frame.fn = flr
	frame.pc = frame.lr
	frame.lr = 0
	frame.sp = frame.fp
	frame.fp = 0

	// On link register architectures, sighandler saves the LR on stack
	// before faking a call.
	if usesLR && injectedCall {
		x := *(*uintptr)(unsafe.Pointer(frame.sp))
		frame.sp += alignUp(sys.MinFrameSize, sys.StackAlign)
		f = findfunc(frame.pc)
		frame.fn = f
		if !f.valid() {
			frame.pc = x
		} else if funcspdelta(f, frame.pc) == 0 {
			frame.lr = x
		}
	}

	u.resolveInternal(false, false)
}

// finishInternal is an unwinder-internal helper called after the stack has been
// exhausted. It sets the unwinder to an invalid state and checks that it
// successfully unwound the entire stack.
func (u *unwinder) finishInternal() {
	u.frame.pc = 0

	// Note that panic != nil is okay here: there can be leftover panics,
	// because the defers on the panic stack do not nest in frame order as
	// they do on the defer stack. If you have:
	//
	//	frame 1 defers d1
	//	frame 2 defers d2
	//	frame 3 defers d3
	//	frame 4 panics
	//	frame 4's panic starts running defers
	//	frame 5, running d3, defers d4
	//	frame 5 panics
	//	frame 5's panic starts running defers
	//	frame 6, running d4, garbage collects
	//	frame 6, running d2, garbage collects
	//
	// During the execution of d4, the panic stack is d4 -> d3, which
	// is nested properly, and we'll treat frame 3 as resumable, because we
	// can find d3. (And in fact frame 3 is resumable. If d4 recovers
	// and frame 5 continues running, d3, d3 can recover and we'll
	// resume execution in (returning from) frame 3.)
	//
	// During the execution of d2, however, the panic stack is d2 -> d3,
	// which is inverted. The scan will match d2 to frame 2 but having
	// d2 on the stack until then means it will not match d3 to frame 3.
	// This is okay: if we're running d2, then all the defers after d2 have
	// completed and their corresponding frames are dead. Not finding d3
	// for frame 3 means we'll set frame 3's continpc == 0, which is correct
	// (frame 3 is dead). At the end of the walk the panic stack can thus
	// contain defers (d3 in this case) for dead frames. The inversion here
	// always indicates a dead frame, and the effect of the inversion on the
	// scan is to hide those dead frames, so the scan is still okay:
	// what's left on the panic stack are exactly (and only) the dead frames.
	//
	// We require callback != nil here because only when callback != nil
	// do we know that gentraceback is being called in a "must be correct"
	// context as opposed to a "best effort" context. The tracebacks with
	// callbacks only happen when everything is stopped nicely.
	// At other times, such as when gathering a stack for a profiling signal
	// or when printing a traceback during a crash, everything may not be
	// stopped nicely, and the stack walk may not be able to complete.
	gp := u.g.ptr()
	if u.flags&(unwindPrintErrors|unwindSilentErrors) == 0 && u.frame.sp != gp.stktopsp {
		print("runtime: g", gp.goid, ": frame.sp=", hex(u.frame.sp), " top=", hex(gp.stktopsp), "\n")
		print("\tstack=[", hex(gp.stack.lo), "-", hex(gp.stack.hi), "\n")
		throw("traceback did not unwind completely")
	}
}

// symPC returns the PC that should be used for symbolizing the current frame.
// Specifically, this is the PC of the last instruction executed in this frame.
//
// If this frame did a normal call, then frame.pc is a return PC, so this will
// return frame.pc-1, which points into the CALL instruction. If the frame was
// interrupted by a signal (e.g., profiler, segv, etc) then frame.pc is for the
// trapped instruction, so this returns frame.pc. See issue #34123. Finally,
// frame.pc can be at function entry when the frame is initialized without
// actually running code, like in runtime.mstart, in which case this returns
// frame.pc because that's the best we can do.
func (u *unwinder) symPC() uintptr {
	if u.flags&unwindTrap == 0 && u.frame.pc > u.frame.fn.entry() {
		// Regular call.
		return u.frame.pc - 1
	}
	// Trapping instruction or we're at the function entry point.
	return u.frame.pc
}

// cgoCallers populates pcBuf with the cgo callers of the current frame using
// the registered cgo unwinder. It returns the number of PCs written to pcBuf.
// If the current frame is not a cgo frame or if there's no registered cgo
// unwinder, it returns 0.
func (u *unwinder) cgoCallers(pcBuf []uintptr) int {
	if cgoTraceback == nil || u.frame.fn.funcID != abi.FuncID_cgocallback || u.cgoCtxt < 0 {
		// We don't have a cgo unwinder (typical case), or we do but we're not
		// in a cgo frame or we're out of cgo context.
		return 0
	}

	ctxt := u.g.ptr().cgoCtxt[u.cgoCtxt]
	u.cgoCtxt--
	cgoContextPCs(ctxt, pcBuf)
	for i, pc := range pcBuf {
		if pc == 0 {
			return i
		}
	}
	return len(pcBuf)
}

// tracebackPCs populates pcBuf with the return addresses for each frame from u
// and returns the number of PCs written to pcBuf. The returned PCs correspond
// to "logical frames" rather than "physical frames"; that is if A is inlined
// into B, this will still return a PCs for both A and B. This also includes PCs
// generated by the cgo unwinder, if one is registered.
//
// If skip != 0, this skips this many logical frames.
//
// Callers should set the unwindSilentErrors flag on u.
func tracebackPCs(u *unwinder, skip int, pcBuf []uintptr) int {
	var cgoBuf [32]uintptr
	n := 0
	for ; n < len(pcBuf) && u.valid(); u.next() {
		f := u.frame.fn
		cgoN := u.cgoCallers(cgoBuf[:])

		// TODO: Why does &u.cache cause u to escape? (Same in traceback2)
		for iu, uf := newInlineUnwinder(f, u.symPC()); n < len(pcBuf) && uf.valid(); uf = iu.next(uf) {
			sf := iu.srcFunc(uf)
			if sf.funcID == abi.FuncIDWrapper && elideWrapperCalling(u.calleeFuncID) {
				// ignore wrappers
			} else if skip > 0 {
				skip--
			} else {
				// Callers expect the pc buffer to contain return addresses
				// and do the -1 themselves, so we add 1 to the call PC to
				// create a return PC.
				pcBuf[n] = uf.pc + 1
				n++
			}
			u.calleeFuncID = sf.funcID
		}
		// Add cgo frames (if we're done skipping over the requested number of
		// Go frames).
		if skip == 0 {
			n += copy(pcBuf[n:], cgoBuf[:cgoN])
		}
	}
	return n
}

// printArgs prints function arguments in traceback.
func printArgs(f funcInfo, argp unsafe.Pointer, pc uintptr) {
	p := (*[abi.TraceArgsMaxLen]uint8)(funcdata(f, abi.FUNCDATA_ArgInfo))
	if p == nil {
		return
	}

	liveInfo := funcdata(f, abi.FUNCDATA_ArgLiveInfo)
	liveIdx := pcdatavalue(f, abi.PCDATA_ArgLiveIndex, pc)
	startOffset := uint8(0xff) // smallest offset that needs liveness info (slots with a lower offset is always live)
	if liveInfo != nil {
		startOffset = *(*uint8)(liveInfo)
	}

	isLive := func(off, slotIdx uint8) bool {
		if liveInfo == nil || liveIdx <= 0 {
			return true // no liveness info, always live
		}
		if off < startOffset {
			return true
		}
		bits := *(*uint8)(add(liveInfo, uintptr(liveIdx)+uintptr(slotIdx/8)))
		return bits&(1<<(slotIdx%8)) != 0
	}

	print1 := func(off, sz, slotIdx uint8) {
		x := readUnaligned64(add(argp, uintptr(off)))
		// mask out irrelevant bits
		if sz < 8 {
			shift := 64 - sz*8
			if goarch.BigEndian {
				x = x >> shift
			} else {
				x = x << shift >> shift
			}
		}
		print(hex(x))
		if !isLive(off, slotIdx) {
			print("?")
		}
	}

	start := true
	printcomma := func() {
		if !start {
			print(", ")
		}
	}
	pi := 0
	slotIdx := uint8(0) // register arg spill slot index
printloop:
	for {
		o := p[pi]
		pi++
		switch o {
		case abi.TraceArgsEndSeq:
			break printloop
		case abi.TraceArgsStartAgg:
			printcomma()
			print("{")
			start = true
			continue
		case abi.TraceArgsEndAgg:
			print("}")
		case abi.TraceArgsDotdotdot:
			printcomma()
			print("...")
		case abi.TraceArgsOffsetTooLarge:
			printcomma()
			print("_")
		default:
			printcomma()
			sz := p[pi]
			pi++
			print1(o, sz, slotIdx)
			if o >= startOffset {
				slotIdx++
			}
		}
		start = false
	}
}

// funcNamePiecesForPrint returns the function name for printing to the user.
// It returns three pieces so it doesn't need an allocation for string
// concatenation.
func funcNamePiecesForPrint(name string) (string, string, string) {
	// Replace the shape name in generic function with "...".
	i := bytealg.IndexByteString(name, '[')
	if i < 0 {
		return name, "", ""
	}
	j := len(name) - 1
	for name[j] != ']' {
		j--
	}
	if j <= i {
		return name, "", ""
	}
	return name[:i], "[...]", name[j+1:]
}

// funcNameForPrint returns the function name for printing to the user.
func funcNameForPrint(name string) string {
	a, b, c := funcNamePiecesForPrint(name)
	return a + b + c
}

// printFuncName prints a function name. name is the function name in
// the binary's func data table.
func printFuncName(name string) {
	if name == "runtime.gopanic" {
		print("panic")
		return
	}
	a, b, c := funcNamePiecesForPrint(name)
	print(a, b, c)
}

func printcreatedby(gp *g) {
	// Show what created goroutine, except main goroutine (goid 1).
	pc := gp.gopc
	f := findfunc(pc)
	if f.valid() && showframe(f.srcFunc(), gp, false, abi.FuncIDNormal) && gp.goid != 1 {
		printcreatedby1(f, pc, gp.parentGoid)
	}
}

func printcreatedby1(f funcInfo, pc uintptr, goid uint64) {
	print("created by ")
	printFuncName(funcname(f))
	if goid != 0 {
		print(" in goroutine ", goid)
	}
	print("\n")
	tracepc := pc // back up to CALL instruction for funcline.
	if pc > f.entry() {
		tracepc -= sys.PCQuantum
	}
	file, line := funcline(f, tracepc)
	print("\t", file, ":", line)
	if pc > f.entry() {
		print(" +", hex(pc-f.entry()))
	}
	print("\n")
}

func traceback(pc, sp, lr uintptr, gp *g) {
	traceback1(pc, sp, lr, gp, 0)
}

// tracebacktrap is like traceback but expects that the PC and SP were obtained
// from a trap, not from gp->sched or gp->syscallpc/gp->syscallsp or getcallerpc/getcallersp.
// Because they are from a trap instead of from a saved pair,
// the initial PC must not be rewound to the previous instruction.
// (All the saved pairs record a PC that is a return address, so we
// rewind it into the CALL instruction.)
// If gp.m.libcall{g,pc,sp} information is available, it uses that information in preference to
// the pc/sp/lr passed in.
func tracebacktrap(pc, sp, lr uintptr, gp *g) {
	if gp.m.libcallsp != 0 {
		// We're in C code somewhere, traceback from the saved position.
		traceback1(gp.m.libcallpc, gp.m.libcallsp, 0, gp.m.libcallg.ptr(), 0)
		return
	}
	traceback1(pc, sp, lr, gp, unwindTrap)
}

func traceback1(pc, sp, lr uintptr, gp *g, flags unwindFlags) {
	// If the goroutine is in cgo, and we have a cgo traceback, print that.
	if iscgo && gp.m != nil && gp.m.ncgo > 0 && gp.syscallsp != 0 && gp.m.cgoCallers != nil && gp.m.cgoCallers[0] != 0 {
		// Lock cgoCallers so that a signal handler won't
		// change it, copy the array, reset it, unlock it.
		// We are locked to the thread and are not running
		// concurrently with a signal handler.
		// We just have to stop a signal handler from interrupting
		// in the middle of our copy.
		gp.m.cgoCallersUse.Store(1)
		cgoCallers := *gp.m.cgoCallers
		gp.m.cgoCallers[0] = 0
		gp.m.cgoCallersUse.Store(0)

		printCgoTraceback(&cgoCallers)
	}

	if readgstatus(gp)&^_Gscan == _Gsyscall {
		// Override registers if blocked in system call.
		pc = gp.syscallpc
		sp = gp.syscallsp
		flags &^= unwindTrap
	}
	if gp.m != nil && gp.m.vdsoSP != 0 {
		// Override registers if running in VDSO. This comes after the
		// _Gsyscall check to cover VDSO calls after entersyscall.
		pc = gp.m.vdsoPC
		sp = gp.m.vdsoSP
		flags &^= unwindTrap
	}

	// Print traceback.
	//
	// We print the first tracebackInnerFrames frames, and the last
	// tracebackOuterFrames frames. There are many possible approaches to this.
	// There are various complications to this:
	//
	// - We'd prefer to walk the stack once because in really bad situations
	//   traceback may crash (and we want as much output as possible) or the stack
	//   may be changing.
	//
	// - Each physical frame can represent several logical frames, so we might
	//   have to pause in the middle of a physical frame and pick up in the middle
	//   of a physical frame.
	//
	// - The cgo symbolizer can expand a cgo PC to more than one logical frame,
	//   and involves juggling state on the C side that we don't manage. Since its
	//   expansion state is managed on the C side, we can't capture the expansion
	//   state part way through, and because the output strings are managed on the
	//   C side, we can't capture the output. Thus, our only choice is to replay a
	//   whole expansion, potentially discarding some of it.
	//
	// Rejected approaches:
	//
	// - Do two passes where the first pass just counts and the second pass does
	//   all the printing. This is undesirable if the stack is corrupted or changing
	//   because we won't see a partial stack if we panic.
	//
	// - Keep a ring buffer of the last N logical frames and use this to print
	//   the bottom frames once we reach the end of the stack. This works, but
	//   requires keeping a surprising amount of state on the stack, and we have
	//   to run the cgo symbolizer twice—once to count frames, and a second to
	//   print them—since we can't retain the strings it returns.
	//
	// Instead, we print the outer frames, and if we reach that limit, we clone
	// the unwinder, count the remaining frames, and then skip forward and
	// finish printing from the clone. This makes two passes over the outer part
	// of the stack, but the single pass over the inner part ensures that's
	// printed immediately and not revisited. It keeps minimal state on the
	// stack. And through a combination of skip counts and limits, we can do all
	// of the steps we need with a single traceback printer implementation.
	//
	// We could be more lax about exactly how many frames we print, for example
	// always stopping and resuming on physical frame boundaries, or at least
	// cgo expansion boundaries. It's not clear that's much simpler.
	flags |= unwindPrintErrors
	var u unwinder
	tracebackWithRuntime := func(showRuntime bool) int {
		const maxInt int = 0x7fffffff
		u.initAt(pc, sp, lr, gp, flags)
		n, lastN := traceback2(&u, showRuntime, 0, tracebackInnerFrames)
		if n < tracebackInnerFrames {
			// We printed the whole stack.
			return n
		}
		// Clone the unwinder and figure out how many frames are left. This
		// count will include any logical frames already printed for u's current
		// physical frame.
		u2 := u
		remaining, _ := traceback2(&u, showRuntime, maxInt, 0)
		elide := remaining - lastN - tracebackOuterFrames
		if elide > 0 {
			print("...", elide, " frames elided...\n")
			traceback2(&u2, showRuntime, lastN+elide, tracebackOuterFrames)
		} else if elide <= 0 {
			// There are tracebackOuterFrames or fewer frames left to print.
			// Just print the rest of the stack.
			traceback2(&u2, showRuntime, lastN, tracebackOuterFrames)
		}
		return n
	}
	// By default, omits runtime frames. If that means we print nothing at all,
	// repeat forcing all frames printed.
	if tracebackWithRuntime(false) == 0 {
		tracebackWithRuntime(true)
	}
	printcreatedby(gp)

	if gp.ancestors == nil {
		return
	}
	for _, ancestor := range *gp.ancestors {
		printAncestorTraceback(ancestor)
	}
}

// traceback2 prints a stack trace starting at u. It skips the first "skip"
// logical frames, after which it prints at most "max" logical frames. It
// returns n, which is the number of logical frames skipped and printed, and
// lastN, which is the number of logical frames skipped or printed just in the
// physical frame that u references.
func traceback2(u *unwinder, showRuntime bool, skip, max int) (n, lastN int) {
	// commitFrame commits to a logical frame and returns whether this frame
	// should be printed and whether iteration should stop.
	commitFrame := func() (pr, stop bool) {
		if skip == 0 && max == 0 {
			// Stop
			return false, true
		}
		n++
		lastN++
		if skip > 0 {
			// Skip
			skip--
			return false, false
		}
		// Print
		max--
		return true, false
	}

	gp := u.g.ptr()
	level, _, _ := gotraceback()
	var cgoBuf [32]uintptr
	for ; u.valid(); u.next() {
		lastN = 0
		f := u.frame.fn
		for iu, uf := newInlineUnwinder(f, u.symPC()); uf.valid(); uf = iu.next(uf) {
			sf := iu.srcFunc(uf)
			callee := u.calleeFuncID
			u.calleeFuncID = sf.funcID
			if !(showRuntime || showframe(sf, gp, n == 0, callee)) {
				continue
			}

			if pr, stop := commitFrame(); stop {
				return
			} else if !pr {
				continue
			}

			name := sf.name()
			file, line := iu.fileLine(uf)
			// Print during crash.
			//	main(0x1, 0x2, 0x3)
			//		/home/rsc/go/src/runtime/x.go:23 +0xf
			//
			printFuncName(name)
			print("(")
			if iu.isInlined(uf) {
				print("...")
			} else {
				argp := unsafe.Pointer(u.frame.argp)
				printArgs(f, argp, u.symPC())
			}
			print(")\n")
			print("\t", file, ":", line)
			// The contract between Callers and CallersFrames uses
			// return addresses, which are +1 relative to the CALL
			// instruction. Follow that convention.
			pc := uf.pc + 1
			if !iu.isInlined(uf) && pc > f.entry() {
				// Func-relative PCs make no sense for inlined
				// frames because there is no actual entry.
				print(" +", hex(pc-f.entry()))
			}
			if gp.m != nil && gp.m.throwing >= throwTypeRuntime && gp == gp.m.curg || level >= 2 {
				if !iu.isInlined(uf) {
					// The stack information makes no sense for inline frames.
					print(" fp=", hex(u.frame.fp), " sp=", hex(u.frame.sp), " pc=", hex(pc))
				} else {
					// The PC for an inlined frame is a special marker NOP,
					// but crash monitoring tools may still parse the PCs
					// and feed them to CallersFrames.
					print(" pc=", hex(pc))
				}
			}
			print("\n")
		}

		// Print cgo frames.
		if cgoN := u.cgoCallers(cgoBuf[:]); cgoN > 0 {
			var arg cgoSymbolizerArg
			anySymbolized := false
			stop := false
			for _, pc := range cgoBuf[:cgoN] {
				if cgoSymbolizer == nil {
					if pr, stop := commitFrame(); stop {
						break
					} else if pr {
						print("non-Go function at pc=", hex(pc), "\n")
					}
				} else {
					stop = printOneCgoTraceback(pc, commitFrame, &arg)
					anySymbolized = true
					if stop {
						break
					}
				}
			}
			if anySymbolized {
				// Free symbolization state.
				arg.pc = 0
				callCgoSymbolizer(&arg)
			}
			if stop {
				return
			}
		}
	}
	return n, 0
}

// printAncestorTraceback prints the traceback of the given ancestor.
// TODO: Unify this with gentraceback and CallersFrames.
func printAncestorTraceback(ancestor ancestorInfo) {
	print("[originating from goroutine ", ancestor.goid, "]:\n")
	for fidx, pc := range ancestor.pcs {
		f := findfunc(pc) // f previously validated
		if showfuncinfo(f.srcFunc(), fidx == 0, abi.FuncIDNormal) {
			printAncestorTracebackFuncInfo(f, pc)
		}
	}
	if len(ancestor.pcs) == tracebackInnerFrames {
		print("...additional frames elided...\n")
	}
	// Show what created goroutine, except main goroutine (goid 1).
	f := findfunc(ancestor.gopc)
	if f.valid() && showfuncinfo(f.srcFunc(), false, abi.FuncIDNormal) && ancestor.goid != 1 {
		// In ancestor mode, we'll already print the goroutine ancestor.
		// Pass 0 for the goid parameter so we don't print it again.
		printcreatedby1(f, ancestor.gopc, 0)
	}
}

// printAncestorTracebackFuncInfo prints the given function info at a given pc
// within an ancestor traceback. The precision of this info is reduced
// due to only have access to the pcs at the time of the caller
// goroutine being created.
func printAncestorTracebackFuncInfo(f funcInfo, pc uintptr) {
	u, uf := newInlineUnwinder(f, pc)
	file, line := u.fileLine(uf)
	printFuncName(u.srcFunc(uf).name())
	print("(...)\n")
	print("\t", file, ":", line)
	if pc > f.entry() {
		print(" +", hex(pc-f.entry()))
	}
	print("\n")
}

func callers(skip int, pcbuf []uintptr) int {
	sp := getcallersp()
	pc := getcallerpc()
	gp := getg()
	var n int
	systemstack(func() {
		var u unwinder
		u.initAt(pc, sp, 0, gp, unwindSilentErrors)
		n = tracebackPCs(&u, skip, pcbuf)
	})
	return n
}

func gcallers(gp *g, skip int, pcbuf []uintptr) int {
	var u unwinder
	u.init(gp, unwindSilentErrors)
	return tracebackPCs(&u, skip, pcbuf)
}

// showframe reports whether the frame with the given characteristics should
// be printed during a traceback.
func showframe(sf srcFunc, gp *g, firstFrame bool, calleeID abi.FuncID) bool {
	mp := getg().m
	if mp.throwing >= throwTypeRuntime && gp != nil && (gp == mp.curg || gp == mp.caughtsig.ptr()) {
		return true
	}
	return showfuncinfo(sf, firstFrame, calleeID)
}

// showfuncinfo reports whether a function with the given characteristics should
// be printed during a traceback.
func showfuncinfo(sf srcFunc, firstFrame bool, calleeID abi.FuncID) bool {
	level, _, _ := gotraceback()
	if level > 1 {
		// Show all frames.
		return true
	}

	if sf.funcID == abi.FuncIDWrapper && elideWrapperCalling(calleeID) {
		return false
	}

	name := sf.name()

	// Special case: always show runtime.gopanic frame
	// in the middle of a stack trace, so that we can
	// see the boundary between ordinary code and
	// panic-induced deferred code.
	// See golang.org/issue/5832.
	if name == "runtime.gopanic" && !firstFrame {
		return true
	}

	return bytealg.IndexByteString(name, '.') >= 0 && (!hasPrefix(name, "runtime.") || isExportedRuntime(name))
}

// isExportedRuntime reports whether name is an exported runtime function.
// It is only for runtime functions, so ASCII A-Z is fine.
func isExportedRuntime(name string) bool {
	// Check and remove package qualifier.
	n := len("runtime.")
	if len(name) <= n || name[:n] != "runtime." {
		return false
	}
	name = name[n:]
	rcvr := ""

	// Extract receiver type, if any.
	// For example, runtime.(*Func).Entry
	i := len(name) - 1
	for i >= 0 && name[i] != '.' {
		i--
	}
	if i >= 0 {
		rcvr = name[:i]
		name = name[i+1:]
		// Remove parentheses and star for pointer receivers.
		if len(rcvr) >= 3 && rcvr[0] == '(' && rcvr[1] == '*' && rcvr[len(rcvr)-1] == ')' {
			rcvr = rcvr[2 : len(rcvr)-1]
		}
	}

	// Exported functions and exported methods on exported types.
	return len(name) > 0 && 'A' <= name[0] && name[0] <= 'Z' && (len(rcvr) == 0 || 'A' <= rcvr[0] && rcvr[0] <= 'Z')
}

// elideWrapperCalling reports whether a wrapper function that called
// function id should be elided from stack traces.
func elideWrapperCalling(id abi.FuncID) bool {
	// If the wrapper called a panic function instead of the
	// wrapped function, we want to include it in stacks.
	return !(id == abi.FuncID_gopanic || id == abi.FuncID_sigpanic || id == abi.FuncID_panicwrap)
}

var gStatusStrings = [...]string{
	_Gidle:      "idle",
	_Grunnable:  "runnable",
	_Grunning:   "running",
	_Gsyscall:   "syscall",
	_Gwaiting:   "waiting",
	_Gdead:      "dead",
	_Gcopystack: "copystack",
	_Gpreempted: "preempted",
}

func goroutineheader(gp *g) {
	level, _, _ := gotraceback()

	gpstatus := readgstatus(gp)

	isScan := gpstatus&_Gscan != 0
	gpstatus &^= _Gscan // drop the scan bit

	// Basic string status
	var status string
	if 0 <= gpstatus && gpstatus < uint32(len(gStatusStrings)) {
		status = gStatusStrings[gpstatus]
	} else {
		status = "???"
	}

	// Override.
	if gpstatus == _Gwaiting && gp.waitreason != waitReasonZero {
		status = gp.waitreason.String()
	}

	// approx time the G is blocked, in minutes
	var waitfor int64
	if (gpstatus == _Gwaiting || gpstatus == _Gsyscall) && gp.waitsince != 0 {
		waitfor = (nanotime() - gp.waitsince) / 60e9
	}
	print("goroutine ", gp.goid)
	if gp.m != nil && gp.m.throwing >= throwTypeRuntime && gp == gp.m.curg || level >= 2 {
		print(" gp=", gp)
		if gp.m != nil {
			print(" m=", gp.m.id, " mp=", gp.m)
		} else {
			print(" m=nil")
		}
	}
	print(" [", status)
	if isScan {
		print(" (scan)")
	}
	if waitfor >= 1 {
		print(", ", waitfor, " minutes")
	}
	if gp.lockedm != 0 {
		print(", locked to thread")
	}
	print("]:\n")
}

func tracebackothers(me *g) {
	level, _, _ := gotraceback()

	// Show the current goroutine first, if we haven't already.
	curgp := getg().m.curg
	if curgp != nil && curgp != me {
		print("\n")
		goroutineheader(curgp)
		traceback(^uintptr(0), ^uintptr(0), 0, curgp)
	}

	// We can't call locking forEachG here because this may be during fatal
	// throw/panic, where locking could be out-of-order or a direct
	// deadlock.
	//
	// Instead, use forEachGRace, which requires no locking. We don't lock
	// against concurrent creation of new Gs, but even with allglock we may
	// miss Gs created after this loop.
	forEachGRace(func(gp *g) {
		if gp == me || gp == curgp || readgstatus(gp) == _Gdead || isSystemGoroutine(gp, false) && level < 2 {
			return
		}
		print("\n")
		goroutineheader(gp)
		// Note: gp.m == getg().m occurs when tracebackothers is called
		// from a signal handler initiated during a systemstack call.
		// The original G is still in the running state, and we want to
		// print its stack.
		if gp.m != getg().m && readgstatus(gp)&^_Gscan == _Grunning {
			print("\tgoroutine running on other thread; stack unavailable\n")
			printcreatedby(gp)
		} else {
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
		}
	})
}

// tracebackHexdump hexdumps part of stk around frame.sp and frame.fp
// for debugging purposes. If the address bad is included in the
// hexdumped range, it will mark it as well.
func tracebackHexdump(stk stack, frame *stkframe, bad uintptr) {
	const expand = 32 * goarch.PtrSize
	const maxExpand = 256 * goarch.PtrSize
	// Start around frame.sp.
	lo, hi := frame.sp, frame.sp
	// Expand to include frame.fp.
	if frame.fp != 0 && frame.fp < lo {
		lo = frame.fp
	}
	if frame.fp != 0 && frame.fp > hi {
		hi = frame.fp
	}
	// Expand a bit more.
	lo, hi = lo-expand, hi+expand
	// But don't go too far from frame.sp.
	if lo < frame.sp-maxExpand {
		lo = frame.sp - maxExpand
	}
	if hi > frame.sp+maxExpand {
		hi = frame.sp + maxExpand
	}
	// And don't go outside the stack bounds.
	if lo < stk.lo {
		lo = stk.lo
	}
	if hi > stk.hi {
		hi = stk.hi
	}

	// Print the hex dump.
	print("stack: frame={sp:", hex(frame.sp), ", fp:", hex(frame.fp), "} stack=[", hex(stk.lo), ",", hex(stk.hi), ")\n")
	hexdumpWords(lo, hi, func(p uintptr) byte {
		switch p {
		case frame.fp:
			return '>'
		case frame.sp:
			return '<'
		case bad:
			return '!'
		}
		return 0
	})
}

// isSystemGoroutine reports whether the goroutine g must be omitted
// in stack dumps and deadlock detector. This is any goroutine that
// starts at a runtime.* entry point, except for runtime.main and
// sometimes runtime.runfinq and runtime.handleAsyncEvent (wasm only).
//
// If fixed is true, any goroutine that can vary between user and
// system (that is, the finalizer goroutine) is considered a user
// goroutine.
func isSystemGoroutine(gp *g, fixed bool) bool {
	// Keep this in sync with internal/trace.IsSystemGoroutine.
	f := findfunc(gp.startpc)
	if !f.valid() {
		return false
	}
	if f.funcID == abi.FuncID_runtime_main || f.funcID == abi.FuncID_corostart {
		return false
	}
	if f.funcID == abi.FuncID_runfinq || f.funcID == abi.FuncID_handleAsyncEvent {
		// We include the finalizer goroutine if it's calling
		// back into user code, same for handleAsyncEvent on wasm.
		if fixed {
			// This goroutine can vary. In fixed mode,
			// always consider it a user goroutine.
			return false
		}
		return fingStatus.Load()&fingRunningFinalizer == 0
	}
	return hasPrefix(funcname(f), "runtime.")
}

// SetCgoTraceback records three C functions to use to gather
// traceback information from C code and to convert that traceback
// information into symbolic information. These are used when printing
// stack traces for a program that uses cgo.
//
// The traceback and context functions may be called from a signal
// handler, and must therefore use only async-signal safe functions.
// The symbolizer function may be called while the program is
// crashing, and so must be cautious about using memory.  None of the
// functions may call back into Go.
//
// The context function will be called with a single argument, a
// pointer to a struct:
//
//	struct {
//		Context uintptr
//	}
//
// In C syntax, this struct will be
//
//	struct {
//		uintptr_t Context;
//	};
//
// If the Context field is 0, the context function is being called to
// record the current traceback context. It should record in the
// Context field whatever information is needed about the current
// point of execution to later produce a stack trace, probably the
// stack pointer and PC. In this case the context function will be
// called from C code.
//
// If the Context field is not 0, then it is a value returned by a
// previous call to the context function. This case is called when the
// context is no longer needed; that is, when the Go code is returning
// to its C code caller. This permits the context function to release
// any associated resources.
//
// While it would be correct for the context function to record a
// complete a stack trace whenever it is called, and simply copy that
// out in the traceback function, in a typical program the context
// function will be called many times without ever recording a
// traceback for that context. Recording a complete stack trace in a
// call to the context function is likely to be inefficient.
//
// The traceback function will be called with a single argument, a
// pointer to a struct:
//
//	struct {
//		Context    uintptr
//		SigContext uintptr
//		Buf        *uintptr
//		Max        uintptr
//	}
//
// In C syntax, this struct will be
//
//	struct {
//		uintptr_t  Context;
//		uintptr_t  SigContext;
//		uintptr_t* Buf;
//		uintptr_t  Max;
//	};
//
// The Context field will be zero to gather a traceback from the
// current program execution point. In this case, the traceback
// function will be called from C code.
//
// Otherwise Context will be a value previously returned by a call to
// the context function. The traceback function should gather a stack
// trace from that saved point in the program execution. The traceback
// function may be called from an execution thread other than the one
// that recorded the context, but only when the context is known to be
// valid and unchanging. The traceback function may also be called
// deeper in the call stack on the same thread that recorded the
// context. The traceback function may be called multiple times with
// the same Context value; it will usually be appropriate to cache the
// result, if possible, the first time this is called for a specific
// context value.
//
// If the traceback function is called from a signal handler on a Unix
// system, SigContext will be the signal context argument passed to
// the signal handler (a C ucontext_t* cast to uintptr_t). This may be
// used to start tracing at the point where the signal occurred. If
// the traceback function is not called from a signal handler,
// SigContext will be zero.
//
// Buf is where the traceback information should be stored. It should
// be PC values, such that Buf[0] is the PC of the caller, Buf[1] is
// the PC of that function's caller, and so on.  Max is the maximum
// number of entries to store.  The function should store a zero to
// indicate the top of the stack, or that the caller is on a different
// stack, presumably a Go stack.
//
// Unlike runtime.Callers, the PC values returned should, when passed
// to the symbolizer function, return the file/line of the call
// instruction.  No additional subtraction is required or appropriate.
//
// On all platforms, the traceback function is invoked when a call from
// Go to C to Go requests a stack trace. On linux/amd64, linux/ppc64le,
// linux/arm64, and freebsd/amd64, the traceback function is also invoked
// when a signal is received by a thread that is executing a cgo call.
// The traceback function should not make assumptions about when it is
// called, as future versions of Go may make additional calls.
//
// The symbolizer function will be called with a single argument, a
// pointer to a struct:
//
//	struct {
//		PC      uintptr // program counter to fetch information for
//		File    *byte   // file name (NUL terminated)
//		Lineno  uintptr // line number
//		Func    *byte   // function name (NUL terminated)
//		Entry   uintptr // function entry point
//		More    uintptr // set non-zero if more info for this PC
//		Data    uintptr // unused by runtime, available for function
//	}
//
// In C syntax, this struct will be
//
//	struct {
//		uintptr_t PC;
//		char*     File;
//		uintptr_t Lineno;
//		char*     Func;
//		uintptr_t Entry;
//		uintptr_t More;
//		uintptr_t Data;
//	};
//
// The PC field will be a value returned by a call to the traceback
// function.
//
// The first time the function is called for a particular traceback,
// all the fields except PC will be 0. The function should fill in the
// other fields if possible, setting them to 0/nil if the information
// is not available. The Data field may be used to store any useful
// information across calls. The More field should be set to non-zero
// if there is more information for this PC, zero otherwise. If More
// is set non-zero, the function will be called again with the same
// PC, and may return different information (this is intended for use
// with inlined functions). If More is zero, the function will be
// called with the next PC value in the traceback. When the traceback
// is complete, the function will be called once more with PC set to
// zero; this may be used to free any information. Each call will
// leave the fields of the struct set to the same values they had upon
// return, except for the PC field when the More field is zero. The
// function must not keep a copy of the struct pointer between calls.
//
// When calling SetCgoTraceback, the version argument is the version
// number of the structs that the functions expect to receive.
// Currently this must be zero.
//
// The symbolizer function may be nil, in which case the results of
// the traceback function will be displayed as numbers. If the
// traceback function is nil, the symbolizer function will never be
// called. The context function may be nil, in which case the
// traceback function will only be called with the context field set
// to zero.  If the context function is nil, then calls from Go to C
// to Go will not show a traceback for the C portion of the call stack.
//
// SetCgoTraceback should be called only once, ideally from an init function.
func SetCgoTraceback(version int, traceback, context, symbolizer unsafe.Pointer) {
	if version != 0 {
		panic("unsupported version")
	}

	if cgoTraceback != nil && cgoTraceback != traceback ||
		cgoContext != nil && cgoContext != context ||
		cgoSymbolizer != nil && cgoSymbolizer != symbolizer {
		panic("call SetCgoTraceback only once")
	}

	cgoTraceback = traceback
	cgoContext = context
	cgoSymbolizer = symbolizer

	// The context function is called when a C function calls a Go
	// function. As such it is only called by C code in runtime/cgo.
	if _cgo_set_context_function != nil {
		cgocall(_cgo_set_context_function, context)
	}
}

var cgoTraceback unsafe.Pointer
var cgoContext unsafe.Pointer
var cgoSymbolizer unsafe.Pointer

// cgoTracebackArg is the type passed to cgoTraceback.
type cgoTracebackArg struct {
	context    uintptr
	sigContext uintptr
	buf        *uintptr
	max        uintptr
}

// cgoContextArg is the type passed to the context function.
type cgoContextArg struct {
	context uintptr
}

// cgoSymbolizerArg is the type passed to cgoSymbolizer.
type cgoSymbolizerArg struct {
	pc       uintptr
	file     *byte
	lineno   uintptr
	funcName *byte
	entry    uintptr
	more     uintptr
	data     uintptr
}

// printCgoTraceback prints a traceback of callers.
func printCgoTraceback(callers *cgoCallers) {
	if cgoSymbolizer == nil {
		for _, c := range callers {
			if c == 0 {
				break
			}
			print("non-Go function at pc=", hex(c), "\n")
		}
		return
	}

	commitFrame := func() (pr, stop bool) { return true, false }
	var arg cgoSymbolizerArg
	for _, c := range callers {
		if c == 0 {
			break
		}
		printOneCgoTraceback(c, commitFrame, &arg)
	}
	arg.pc = 0
	callCgoSymbolizer(&arg)
}

// printOneCgoTraceback prints the traceback of a single cgo caller.
// This can print more than one line because of inlining.
// It returns the "stop" result of commitFrame.
func printOneCgoTraceback(pc uintptr, commitFrame func() (pr, stop bool), arg *cgoSymbolizerArg) bool {
	arg.pc = pc
	for {
		if pr, stop := commitFrame(); stop {
			return true
		} else if !pr {
			continue
		}

		callCgoSymbolizer(arg)
		if arg.funcName != nil {
			// Note that we don't print any argument
			// information here, not even parentheses.
			// The symbolizer must add that if appropriate.
			println(gostringnocopy(arg.funcName))
		} else {
			println("non-Go function")
		}
		print("\t")
		if arg.file != nil {
			print(gostringnocopy(arg.file), ":", arg.lineno, " ")
		}
		print("pc=", hex(pc), "\n")
		if arg.more == 0 {
			return false
		}
	}
}

// callCgoSymbolizer calls the cgoSymbolizer function.
func callCgoSymbolizer(arg *cgoSymbolizerArg) {
	call := cgocall
	if panicking.Load() > 0 || getg().m.curg != getg() {
		// We do not want to call into the scheduler when panicking
		// or when on the system stack.
		call = asmcgocall
	}
	if msanenabled {
		msanwrite(unsafe.Pointer(arg), unsafe.Sizeof(cgoSymbolizerArg{}))
	}
	if asanenabled {
		asanwrite(unsafe.Pointer(arg), unsafe.Sizeof(cgoSymbolizerArg{}))
	}
	call(cgoSymbolizer, noescape(unsafe.Pointer(arg)))
}

// cgoContextPCs gets the PC values from a cgo traceback.
func cgoContextPCs(ctxt uintptr, buf []uintptr) {
	if cgoTraceback == nil {
		return
	}
	call := cgocall
	if panicking.Load() > 0 || getg().m.curg != getg() {
		// We do not want to call into the scheduler when panicking
		// or when on the system stack.
		call = asmcgocall
	}
	arg := cgoTracebackArg{
		context: ctxt,
		buf:     (*uintptr)(noescape(unsafe.Pointer(&buf[0]))),
		max:     uintptr(len(buf)),
	}
	if msanenabled {
		msanwrite(unsafe.Pointer(&arg), unsafe.Sizeof(arg))
	}
	if asanenabled {
		asanwrite(unsafe.Pointer(&arg), unsafe.Sizeof(arg))
	}
	call(cgoTraceback, noescape(unsafe.Pointer(&arg)))
}
