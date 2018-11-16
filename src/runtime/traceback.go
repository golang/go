// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// The code in this file implements stack trace walking for all architectures.
// The most important fact about a given architecture is whether it uses a link register.
// On systems with link registers, the prologue for a non-leaf function stores the
// incoming value of LR at the bottom of the newly allocated stack frame.
// On systems without link registers, the architecture pushes a return PC during
// the call instruction, so the return PC ends up above the stack frame.
// In this file, the return PC is always called LR, no matter how it was found.
//
// To date, the opposite of a link register architecture is an x86 architecture.
// This code may need to change if some other kind of non-link-register
// architecture comes along.
//
// The other important fact is the size of a pointer: on 32-bit systems the LR
// takes up only 4 bytes on the stack, while on 64-bit systems it takes up 8 bytes.
// Typically this is ptrSize.
//
// As an exception, amd64p32 has ptrSize == 4 but the CALL instruction still
// stores an 8-byte return PC onto the stack. To accommodate this, we use regSize
// as the size of the architecture-pushed return PC.
//
// usesLR is defined below in terms of minFrameSize, which is defined in
// arch_$GOARCH.go. ptrSize and regSize are defined in stubs.go.

const usesLR = sys.MinFrameSize > 0

var skipPC uintptr

func tracebackinit() {
	// Go variable initialization happens late during runtime startup.
	// Instead of initializing the variables above in the declarations,
	// schedinit calls this function so that the variables are
	// initialized and available earlier in the startup sequence.
	skipPC = funcPC(skipPleaseUseCallersFrames)
}

// Traceback over the deferred function calls.
// Report them like calls that have been invoked but not started executing yet.
func tracebackdefers(gp *g, callback func(*stkframe, unsafe.Pointer) bool, v unsafe.Pointer) {
	var frame stkframe
	for d := gp._defer; d != nil; d = d.link {
		fn := d.fn
		if fn == nil {
			// Defer of nil function. Args don't matter.
			frame.pc = 0
			frame.fn = funcInfo{}
			frame.argp = 0
			frame.arglen = 0
			frame.argmap = nil
		} else {
			frame.pc = fn.fn
			f := findfunc(frame.pc)
			if !f.valid() {
				print("runtime: unknown pc in defer ", hex(frame.pc), "\n")
				throw("unknown pc")
			}
			frame.fn = f
			frame.argp = uintptr(deferArgs(d))
			var ok bool
			frame.arglen, frame.argmap, ok = getArgInfoFast(f, true)
			if !ok {
				frame.arglen, frame.argmap = getArgInfo(&frame, f, true, fn)
			}
		}
		frame.continpc = frame.pc
		if !callback((*stkframe)(noescape(unsafe.Pointer(&frame))), v) {
			return
		}
	}
}

const sizeofSkipFunction = 256

// This function is defined in asm.s to be sizeofSkipFunction bytes long.
func skipPleaseUseCallersFrames()

// Generic traceback. Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
//
// The skip argument is only valid with pcbuf != nil and counts the number
// of logical frames to skip rather than physical frames (with inlining, a
// PC in pcbuf can represent multiple calls). If a PC is partially skipped
// and max > 1, pcbuf[1] will be runtime.skipPleaseUseCallersFrames+N where
// N indicates the number of logical frames to skip in pcbuf[0].
func gentraceback(pc0, sp0, lr0 uintptr, gp *g, skip int, pcbuf *uintptr, max int, callback func(*stkframe, unsafe.Pointer) bool, v unsafe.Pointer, flags uint) int {
	if skip > 0 && callback != nil {
		throw("gentraceback callback cannot be used with non-zero skip")
	}

	// Don't call this "g"; it's too easy get "g" and "gp" confused.
	if ourg := getg(); ourg == gp && ourg == ourg.m.curg {
		// The starting sp has been passed in as a uintptr, and the caller may
		// have other uintptr-typed stack references as well.
		// If during one of the calls that got us here or during one of the
		// callbacks below the stack must be grown, all these uintptr references
		// to the stack will not be updated, and gentraceback will continue
		// to inspect the old stack memory, which may no longer be valid.
		// Even if all the variables were updated correctly, it is not clear that
		// we want to expose a traceback that begins on one stack and ends
		// on another stack. That could confuse callers quite a bit.
		// Instead, we require that gentraceback and any other function that
		// accepts an sp for the current goroutine (typically obtained by
		// calling getcallersp) must not run on that goroutine's stack but
		// instead on the g0 stack.
		throw("gentraceback cannot trace user goroutine on its own stack")
	}
	level, _, _ := gotraceback()

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

	nprint := 0
	var frame stkframe
	frame.pc = pc0
	frame.sp = sp0
	if usesLR {
		frame.lr = lr0
	}
	waspanic := false
	cgoCtxt := gp.cgoCtxt
	printing := pcbuf == nil && callback == nil
	_defer := gp._defer
	elideWrapper := false

	for _defer != nil && _defer.sp == _NoArgs {
		_defer = _defer.link
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if frame.pc == 0 {
		if usesLR {
			frame.pc = *(*uintptr)(unsafe.Pointer(frame.sp))
			frame.lr = 0
		} else {
			frame.pc = uintptr(*(*sys.Uintreg)(unsafe.Pointer(frame.sp)))
			frame.sp += sys.RegSize
		}
	}

	f := findfunc(frame.pc)
	if !f.valid() {
		if callback != nil || printing {
			print("runtime: unknown pc ", hex(frame.pc), "\n")
			tracebackHexdump(gp.stack, &frame, 0)
		}
		if callback != nil {
			throw("unknown pc")
		}
		return 0
	}
	frame.fn = f

	var cache pcvalueCache

	n := 0
	for n < max {
		// Typically:
		//	pc is the PC of the running function.
		//	sp is the stack pointer at that program counter.
		//	fp is the frame pointer (caller's stack pointer) at that program counter, or nil if unknown.
		//	stk is the stack containing sp.
		//	The caller's program counter is lr, unless lr is zero, in which case it is *(uintptr*)sp.
		f = frame.fn
		if f.pcsp == 0 {
			// No frame information, must be external function, like race support.
			// See golang.org/issue/13568.
			break
		}

		// Found an actual function.
		// Derive frame pointer and link register.
		if frame.fp == 0 {
			// Jump over system stack transitions. If we're on g0 and there's a user
			// goroutine, try to jump. Otherwise this is a regular call.
			if flags&_TraceJumpStack != 0 && gp == gp.m.g0 && gp.m.curg != nil {
				switch f.funcID {
				case funcID_morestack:
					// morestack does not return normally -- newstack()
					// gogo's to curg.sched. Match that.
					// This keeps morestack() from showing up in the backtrace,
					// but that makes some sense since it'll never be returned
					// to.
					frame.pc = gp.m.curg.sched.pc
					frame.fn = findfunc(frame.pc)
					f = frame.fn
					frame.sp = gp.m.curg.sched.sp
					cgoCtxt = gp.m.curg.cgoCtxt
				case funcID_systemstack:
					// systemstack returns normally, so just follow the
					// stack transition.
					frame.sp = gp.m.curg.sched.sp
					cgoCtxt = gp.m.curg.cgoCtxt
				}
			}
			frame.fp = frame.sp + uintptr(funcspdelta(f, frame.pc, &cache))
			if !usesLR {
				// On x86, call instruction pushes return PC before entering new function.
				frame.fp += sys.RegSize
			}
		}
		var flr funcInfo
		if topofstack(f, gp.m != nil && gp == gp.m.g0) {
			frame.lr = 0
			flr = funcInfo{}
		} else if usesLR && f.funcID == funcID_jmpdefer {
			// jmpdefer modifies SP/LR/PC non-atomically.
			// If a profiling interrupt arrives during jmpdefer,
			// the stack unwind may see a mismatched register set
			// and get confused. Stop if we see PC within jmpdefer
			// to avoid that confusion.
			// See golang.org/issue/8153.
			if callback != nil {
				throw("traceback_arm: found jmpdefer when tracing with callback")
			}
			frame.lr = 0
		} else {
			var lrPtr uintptr
			if usesLR {
				if n == 0 && frame.sp < frame.fp || frame.lr == 0 {
					lrPtr = frame.sp
					frame.lr = *(*uintptr)(unsafe.Pointer(lrPtr))
				}
			} else {
				if frame.lr == 0 {
					lrPtr = frame.fp - sys.RegSize
					frame.lr = uintptr(*(*sys.Uintreg)(unsafe.Pointer(lrPtr)))
				}
			}
			flr = findfunc(frame.lr)
			if !flr.valid() {
				// This happens if you get a profiling interrupt at just the wrong time.
				// In that context it is okay to stop early.
				// But if callback is set, we're doing a garbage collection and must
				// get everything, so crash loudly.
				doPrint := printing
				if doPrint && gp.m.incgo && f.funcID == funcID_sigpanic {
					// We can inject sigpanic
					// calls directly into C code,
					// in which case we'll see a C
					// return PC. Don't complain.
					doPrint = false
				}
				if callback != nil || doPrint {
					print("runtime: unexpected return pc for ", funcname(f), " called from ", hex(frame.lr), "\n")
					tracebackHexdump(gp.stack, &frame, lrPtr)
				}
				if callback != nil {
					throw("unknown caller pc")
				}
			}
		}

		frame.varp = frame.fp
		if !usesLR {
			// On x86, call instruction pushes return PC before entering new function.
			frame.varp -= sys.RegSize
		}

		// If framepointer_enabled and there's a frame, then
		// there's a saved bp here.
		if frame.varp > frame.sp && (framepointer_enabled && GOARCH == "amd64" || GOARCH == "arm64") {
			frame.varp -= sys.RegSize
		}

		// Derive size of arguments.
		// Most functions have a fixed-size argument block,
		// so we can use metadata about the function f.
		// Not all, though: there are some variadic functions
		// in package runtime and reflect, and for those we use call-specific
		// metadata recorded by f's caller.
		if callback != nil || printing {
			frame.argp = frame.fp + sys.MinFrameSize
			var ok bool
			frame.arglen, frame.argmap, ok = getArgInfoFast(f, callback != nil)
			if !ok {
				frame.arglen, frame.argmap = getArgInfo(&frame, f, callback != nil, nil)
			}
		}

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
		if waspanic {
			// We match up defers with frames using the SP.
			// However, if the function has an empty stack
			// frame, then it's possible (on LR machines)
			// for multiple call frames to have the same
			// SP. But, since a function with no frame
			// can't push a defer, the defer can't belong
			// to that frame.
			if _defer != nil && _defer.sp == frame.sp && frame.sp != frame.fp {
				frame.continpc = frame.fn.entry + uintptr(frame.fn.deferreturn) + 1
				// Note: the +1 is to offset the -1 that
				// stack.go:getStackMap does to back up a return
				// address make sure the pc is in the CALL instruction.
			} else {
				frame.continpc = 0
			}
		}

		// Unwind our local defer stack past this frame.
		for _defer != nil && ((_defer.sp == frame.sp && frame.sp != frame.fp) || _defer.sp == _NoArgs) {
			_defer = _defer.link
		}

		if callback != nil {
			if !callback((*stkframe)(noescape(unsafe.Pointer(&frame))), v) {
				return n
			}
		}

		if pcbuf != nil {
			if skip == 0 {
				(*[1 << 20]uintptr)(unsafe.Pointer(pcbuf))[n] = frame.pc
			} else {
				// backup to CALL instruction to read inlining info (same logic as below)
				tracepc := frame.pc
				if (n > 0 || flags&_TraceTrap == 0) && frame.pc > f.entry && !waspanic {
					tracepc--
				}
				inldata := funcdata(f, _FUNCDATA_InlTree)

				// no inlining info, skip the physical frame
				if inldata == nil {
					skip--
					goto skipped
				}

				ix := pcdatavalue(f, _PCDATA_InlTreeIndex, tracepc, &cache)
				inltree := (*[1 << 20]inlinedCall)(inldata)
				// skip the logical (inlined) frames
				logicalSkipped := 0
				for ix >= 0 && skip > 0 {
					skip--
					logicalSkipped++
					ix = inltree[ix].parent
				}

				// skip the physical frame if there's more to skip
				if skip > 0 {
					skip--
					goto skipped
				}

				// now we have a partially skipped frame
				(*[1 << 20]uintptr)(unsafe.Pointer(pcbuf))[n] = frame.pc

				// if there's room, pcbuf[1] is a skip PC that encodes the number of skipped frames in pcbuf[0]
				if n+1 < max {
					n++
					pc := skipPC + uintptr(logicalSkipped)
					(*[1 << 20]uintptr)(unsafe.Pointer(pcbuf))[n] = pc
				}
			}
		}

		if printing {
			// assume skip=0 for printing.
			//
			// Never elide wrappers if we haven't printed
			// any frames. And don't elide wrappers that
			// called panic rather than the wrapped
			// function. Otherwise, leave them out.
			name := funcname(f)
			nextElideWrapper := elideWrapperCalling(name)
			if (flags&_TraceRuntimeFrames) != 0 || showframe(f, gp, nprint == 0, elideWrapper && nprint != 0) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//
				tracepc := frame.pc // back up to CALL instruction for funcline.
				if (n > 0 || flags&_TraceTrap == 0) && frame.pc > f.entry && !waspanic {
					tracepc--
				}
				file, line := funcline(f, tracepc)
				inldata := funcdata(f, _FUNCDATA_InlTree)
				if inldata != nil {
					inltree := (*[1 << 20]inlinedCall)(inldata)
					ix := pcdatavalue(f, _PCDATA_InlTreeIndex, tracepc, nil)
					for ix != -1 {
						name := funcnameFromNameoff(f, inltree[ix].func_)
						print(name, "(...)\n")
						print("\t", file, ":", line, "\n")

						file = funcfile(f, inltree[ix].file)
						line = inltree[ix].line
						ix = inltree[ix].parent
					}
				}
				if name == "runtime.gopanic" {
					name = "panic"
				}
				print(name, "(")
				argp := (*[100]uintptr)(unsafe.Pointer(frame.argp))
				for i := uintptr(0); i < frame.arglen/sys.PtrSize; i++ {
					if i >= 10 {
						print(", ...")
						break
					}
					if i != 0 {
						print(", ")
					}
					print(hex(argp[i]))
				}
				print(")\n")
				print("\t", file, ":", line)
				if frame.pc > f.entry {
					print(" +", hex(frame.pc-f.entry))
				}
				if gp.m != nil && gp.m.throwing > 0 && gp == gp.m.curg || level >= 2 {
					print(" fp=", hex(frame.fp), " sp=", hex(frame.sp), " pc=", hex(frame.pc))
				}
				print("\n")
				nprint++
			}
			elideWrapper = nextElideWrapper
		}
		n++

	skipped:
		if f.funcID == funcID_cgocallback_gofunc && len(cgoCtxt) > 0 {
			ctxt := cgoCtxt[len(cgoCtxt)-1]
			cgoCtxt = cgoCtxt[:len(cgoCtxt)-1]

			// skip only applies to Go frames.
			// callback != nil only used when we only care
			// about Go frames.
			if skip == 0 && callback == nil {
				n = tracebackCgoContext(pcbuf, printing, ctxt, n, max)
			}
		}

		waspanic = f.funcID == funcID_sigpanic

		// Do not unwind past the bottom of the stack.
		if !flr.valid() {
			break
		}

		// Unwind to next frame.
		frame.fn = flr
		frame.pc = frame.lr
		frame.lr = 0
		frame.sp = frame.fp
		frame.fp = 0
		frame.argmap = nil

		// On link register architectures, sighandler saves the LR on stack
		// before faking a call to sigpanic.
		if usesLR && waspanic {
			x := *(*uintptr)(unsafe.Pointer(frame.sp))
			frame.sp += sys.MinFrameSize
			if GOARCH == "arm64" {
				// arm64 needs 16-byte aligned SP, always
				frame.sp += sys.PtrSize
			}
			f = findfunc(frame.pc)
			frame.fn = f
			if !f.valid() {
				frame.pc = x
			} else if funcspdelta(f, frame.pc, &cache) == 0 {
				frame.lr = x
			}
		}
	}

	if printing {
		n = nprint
	}

	// If callback != nil, we're being called to gather stack information during
	// garbage collection or stack growth. In that context, require that we used
	// up the entire defer stack. If not, then there is a bug somewhere and the
	// garbage collection or stack growth may not have seen the correct picture
	// of the stack. Crash now instead of silently executing the garbage collection
	// or stack copy incorrectly and setting up for a mysterious crash later.
	//
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
	// It's okay in those situations not to use up the entire defer stack:
	// incomplete information then is still better than nothing.
	if callback != nil && n < max && _defer != nil {
		print("runtime: g", gp.goid, ": leftover defer sp=", hex(_defer.sp), " pc=", hex(_defer.pc), "\n")
		for _defer = gp._defer; _defer != nil; _defer = _defer.link {
			print("\tdefer ", _defer, " sp=", hex(_defer.sp), " pc=", hex(_defer.pc), "\n")
		}
		throw("traceback has leftover defers")
	}

	if callback != nil && n < max && frame.sp != gp.stktopsp {
		print("runtime: g", gp.goid, ": frame.sp=", hex(frame.sp), " top=", hex(gp.stktopsp), "\n")
		print("\tstack=[", hex(gp.stack.lo), "-", hex(gp.stack.hi), "] n=", n, " max=", max, "\n")
		throw("traceback did not unwind completely")
	}

	return n
}

// reflectMethodValue is a partial duplicate of reflect.makeFuncImpl
// and reflect.methodValue.
type reflectMethodValue struct {
	fn     uintptr
	stack  *bitvector // ptrmap for both args and results
	argLen uintptr    // just args
}

// getArgInfoFast returns the argument frame information for a call to f.
// It is short and inlineable. However, it does not handle all functions.
// If ok reports false, you must call getArgInfo instead.
// TODO(josharian): once we do mid-stack inlining,
// call getArgInfo directly from getArgInfoFast and stop returning an ok bool.
func getArgInfoFast(f funcInfo, needArgMap bool) (arglen uintptr, argmap *bitvector, ok bool) {
	return uintptr(f.args), nil, !(needArgMap && f.args == _ArgsSizeUnknown)
}

// getArgInfo returns the argument frame information for a call to f
// with call frame frame.
//
// This is used for both actual calls with active stack frames and for
// deferred calls that are not yet executing. If this is an actual
// call, ctxt must be nil (getArgInfo will retrieve what it needs from
// the active stack frame). If this is a deferred call, ctxt must be
// the function object that was deferred.
func getArgInfo(frame *stkframe, f funcInfo, needArgMap bool, ctxt *funcval) (arglen uintptr, argmap *bitvector) {
	arglen = uintptr(f.args)
	if needArgMap && f.args == _ArgsSizeUnknown {
		// Extract argument bitmaps for reflect stubs from the calls they made to reflect.
		switch funcname(f) {
		case "reflect.makeFuncStub", "reflect.methodValueCall":
			// These take a *reflect.methodValue as their
			// context register.
			var mv *reflectMethodValue
			var retValid bool
			if ctxt != nil {
				// This is not an actual call, but a
				// deferred call. The function value
				// is itself the *reflect.methodValue.
				mv = (*reflectMethodValue)(unsafe.Pointer(ctxt))
			} else {
				// This is a real call that took the
				// *reflect.methodValue as its context
				// register and immediately saved it
				// to 0(SP). Get the methodValue from
				// 0(SP).
				arg0 := frame.sp + sys.MinFrameSize
				mv = *(**reflectMethodValue)(unsafe.Pointer(arg0))
				// Figure out whether the return values are valid.
				// Reflect will update this value after it copies
				// in the return values.
				retValid = *(*bool)(unsafe.Pointer(arg0 + 3*sys.PtrSize))
			}
			if mv.fn != f.entry {
				print("runtime: confused by ", funcname(f), "\n")
				throw("reflect mismatch")
			}
			bv := mv.stack
			arglen = uintptr(bv.n * sys.PtrSize)
			if !retValid {
				arglen = uintptr(mv.argLen) &^ (sys.PtrSize - 1)
			}
			argmap = bv
		}
	}
	return
}

// tracebackCgoContext handles tracing back a cgo context value, from
// the context argument to setCgoTraceback, for the gentraceback
// function. It returns the new value of n.
func tracebackCgoContext(pcbuf *uintptr, printing bool, ctxt uintptr, n, max int) int {
	var cgoPCs [32]uintptr
	cgoContextPCs(ctxt, cgoPCs[:])
	var arg cgoSymbolizerArg
	anySymbolized := false
	for _, pc := range cgoPCs {
		if pc == 0 || n >= max {
			break
		}
		if pcbuf != nil {
			(*[1 << 20]uintptr)(unsafe.Pointer(pcbuf))[n] = pc
		}
		if printing {
			if cgoSymbolizer == nil {
				print("non-Go function at pc=", hex(pc), "\n")
			} else {
				c := printOneCgoTraceback(pc, max-n, &arg)
				n += c - 1 // +1 a few lines down
				anySymbolized = true
			}
		}
		n++
	}
	if anySymbolized {
		arg.pc = 0
		callCgoSymbolizer(&arg)
	}
	return n
}

func printcreatedby(gp *g) {
	// Show what created goroutine, except main goroutine (goid 1).
	pc := gp.gopc
	f := findfunc(pc)
	if f.valid() && showframe(f, gp, false, false) && gp.goid != 1 {
		printcreatedby1(f, pc)
	}
}

func printcreatedby1(f funcInfo, pc uintptr) {
	print("created by ", funcname(f), "\n")
	tracepc := pc // back up to CALL instruction for funcline.
	if pc > f.entry {
		tracepc -= sys.PCQuantum
	}
	file, line := funcline(f, tracepc)
	print("\t", file, ":", line)
	if pc > f.entry {
		print(" +", hex(pc-f.entry))
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
	traceback1(pc, sp, lr, gp, _TraceTrap)
}

func traceback1(pc, sp, lr uintptr, gp *g, flags uint) {
	// If the goroutine is in cgo, and we have a cgo traceback, print that.
	if iscgo && gp.m != nil && gp.m.ncgo > 0 && gp.syscallsp != 0 && gp.m.cgoCallers != nil && gp.m.cgoCallers[0] != 0 {
		// Lock cgoCallers so that a signal handler won't
		// change it, copy the array, reset it, unlock it.
		// We are locked to the thread and are not running
		// concurrently with a signal handler.
		// We just have to stop a signal handler from interrupting
		// in the middle of our copy.
		atomic.Store(&gp.m.cgoCallersUse, 1)
		cgoCallers := *gp.m.cgoCallers
		gp.m.cgoCallers[0] = 0
		atomic.Store(&gp.m.cgoCallersUse, 0)

		printCgoTraceback(&cgoCallers)
	}

	var n int
	if readgstatus(gp)&^_Gscan == _Gsyscall {
		// Override registers if blocked in system call.
		pc = gp.syscallpc
		sp = gp.syscallsp
		flags &^= _TraceTrap
	}
	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	n = gentraceback(pc, sp, lr, gp, 0, nil, _TracebackMaxFrames, nil, nil, flags)
	if n == 0 && (flags&_TraceRuntimeFrames) == 0 {
		n = gentraceback(pc, sp, lr, gp, 0, nil, _TracebackMaxFrames, nil, nil, flags|_TraceRuntimeFrames)
	}
	if n == _TracebackMaxFrames {
		print("...additional frames elided...\n")
	}
	printcreatedby(gp)

	if gp.ancestors == nil {
		return
	}
	for _, ancestor := range *gp.ancestors {
		printAncestorTraceback(ancestor)
	}
}

// printAncestorTraceback prints the traceback of the given ancestor.
// TODO: Unify this with gentraceback and CallersFrames.
func printAncestorTraceback(ancestor ancestorInfo) {
	print("[originating from goroutine ", ancestor.goid, "]:\n")
	elideWrapper := false
	for fidx, pc := range ancestor.pcs {
		f := findfunc(pc) // f previously validated
		if showfuncinfo(f, fidx == 0, elideWrapper && fidx != 0) {
			elideWrapper = printAncestorTracebackFuncInfo(f, pc)
		}
	}
	if len(ancestor.pcs) == _TracebackMaxFrames {
		print("...additional frames elided...\n")
	}
	// Show what created goroutine, except main goroutine (goid 1).
	f := findfunc(ancestor.gopc)
	if f.valid() && showfuncinfo(f, false, false) && ancestor.goid != 1 {
		printcreatedby1(f, ancestor.gopc)
	}
}

// printAncestorTraceback prints the given function info at a given pc
// within an ancestor traceback. The precision of this info is reduced
// due to only have access to the pcs at the time of the caller
// goroutine being created.
func printAncestorTracebackFuncInfo(f funcInfo, pc uintptr) bool {
	tracepc := pc // back up to CALL instruction for funcline.
	if pc > f.entry {
		tracepc -= sys.PCQuantum
	}
	file, line := funcline(f, tracepc)
	inldata := funcdata(f, _FUNCDATA_InlTree)
	if inldata != nil {
		inltree := (*[1 << 20]inlinedCall)(inldata)
		ix := pcdatavalue(f, _PCDATA_InlTreeIndex, tracepc, nil)
		for ix != -1 {
			name := funcnameFromNameoff(f, inltree[ix].func_)
			print(name, "(...)\n")
			print("\t", file, ":", line, "\n")

			file = funcfile(f, inltree[ix].file)
			line = inltree[ix].line
			ix = inltree[ix].parent
		}
	}
	name := funcname(f)
	if name == "runtime.gopanic" {
		name = "panic"
	}
	print(name, "(...)\n")
	print("\t", file, ":", line)
	if pc > f.entry {
		print(" +", hex(pc-f.entry))
	}
	print("\n")
	return elideWrapperCalling(name)
}

func callers(skip int, pcbuf []uintptr) int {
	sp := getcallersp()
	pc := getcallerpc()
	gp := getg()
	var n int
	systemstack(func() {
		n = gentraceback(pc, sp, 0, gp, skip, &pcbuf[0], len(pcbuf), nil, nil, 0)
	})
	return n
}

func gcallers(gp *g, skip int, pcbuf []uintptr) int {
	return gentraceback(^uintptr(0), ^uintptr(0), 0, gp, skip, &pcbuf[0], len(pcbuf), nil, nil, 0)
}

func showframe(f funcInfo, gp *g, firstFrame, elideWrapper bool) bool {
	g := getg()
	if g.m.throwing > 0 && gp != nil && (gp == g.m.curg || gp == g.m.caughtsig.ptr()) {
		return true
	}
	return showfuncinfo(f, firstFrame, elideWrapper)
}

func showfuncinfo(f funcInfo, firstFrame, elideWrapper bool) bool {
	level, _, _ := gotraceback()
	if level > 1 {
		// Show all frames.
		return true
	}

	if !f.valid() {
		return false
	}

	if elideWrapper {
		file, _ := funcline(f, f.entry)
		if file == "<autogenerated>" {
			return false
		}
	}

	name := funcname(f)

	// Special case: always show runtime.gopanic frame
	// in the middle of a stack trace, so that we can
	// see the boundary between ordinary code and
	// panic-induced deferred code.
	// See golang.org/issue/5832.
	if name == "runtime.gopanic" && !firstFrame {
		return true
	}

	return contains(name, ".") && (!hasPrefix(name, "runtime.") || isExportedRuntime(name))
}

// isExportedRuntime reports whether name is an exported runtime function.
// It is only for runtime functions, so ASCII A-Z is fine.
func isExportedRuntime(name string) bool {
	const n = len("runtime.")
	return len(name) > n && name[:n] == "runtime." && 'A' <= name[n] && name[n] <= 'Z'
}

// elideWrapperCalling returns whether a wrapper function that called
// function "name" should be elided from stack traces.
func elideWrapperCalling(name string) bool {
	// If the wrapper called a panic function instead of the
	// wrapped function, we want to include it in stacks.
	return !(name == "runtime.gopanic" || name == "runtime.sigpanic" || name == "runtime.panicwrap")
}

var gStatusStrings = [...]string{
	_Gidle:      "idle",
	_Grunnable:  "runnable",
	_Grunning:   "running",
	_Gsyscall:   "syscall",
	_Gwaiting:   "waiting",
	_Gdead:      "dead",
	_Gcopystack: "copystack",
}

func goroutineheader(gp *g) {
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
	print("goroutine ", gp.goid, " [", status)
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
	g := getg()
	gp := g.m.curg
	if gp != nil && gp != me {
		print("\n")
		goroutineheader(gp)
		traceback(^uintptr(0), ^uintptr(0), 0, gp)
	}

	lock(&allglock)
	for _, gp := range allgs {
		if gp == me || gp == g.m.curg || readgstatus(gp) == _Gdead || isSystemGoroutine(gp, false) && level < 2 {
			continue
		}
		print("\n")
		goroutineheader(gp)
		// Note: gp.m == g.m occurs when tracebackothers is
		// called from a signal handler initiated during a
		// systemstack call. The original G is still in the
		// running state, and we want to print its stack.
		if gp.m != g.m && readgstatus(gp)&^_Gscan == _Grunning {
			print("\tgoroutine running on other thread; stack unavailable\n")
			printcreatedby(gp)
		} else {
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
		}
	}
	unlock(&allglock)
}

// tracebackHexdump hexdumps part of stk around frame.sp and frame.fp
// for debugging purposes. If the address bad is included in the
// hexdumped range, it will mark it as well.
func tracebackHexdump(stk stack, frame *stkframe, bad uintptr) {
	const expand = 32 * sys.PtrSize
	const maxExpand = 256 * sys.PtrSize
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

// Does f mark the top of a goroutine stack?
func topofstack(f funcInfo, g0 bool) bool {
	return f.funcID == funcID_goexit ||
		f.funcID == funcID_mstart ||
		f.funcID == funcID_mcall ||
		f.funcID == funcID_morestack ||
		f.funcID == funcID_rt0_go ||
		f.funcID == funcID_externalthreadhandler ||
		// asmcgocall is TOS on the system stack because it
		// switches to the system stack, but in this case we
		// can come back to the regular stack and still want
		// to be able to unwind through the call that appeared
		// on the regular stack.
		(g0 && f.funcID == funcID_asmcgocall)
}

// isSystemGoroutine reports whether the goroutine g must be omitted
// in stack dumps and deadlock detector. This is any goroutine that
// starts at a runtime.* entry point, except for runtime.main and
// sometimes runtime.runfinq.
//
// If fixed is true, any goroutine that can vary between user and
// system (that is, the finalizer goroutine) is considered a user
// goroutine.
func isSystemGoroutine(gp *g, fixed bool) bool {
	// Keep this in sync with cmd/trace/trace.go:isSystemGoroutine.
	f := findfunc(gp.startpc)
	if !f.valid() {
		return false
	}
	if f.funcID == funcID_runtime_main {
		return false
	}
	if f.funcID == funcID_runfinq {
		// We include the finalizer goroutine if it's calling
		// back into user code.
		if fixed {
			// This goroutine can vary. In fixed mode,
			// always consider it a user goroutine.
			return false
		}
		return !fingRunning
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
// and freebsd/amd64, the traceback function is also invoked when a
// signal is received by a thread that is executing a cgo call. The
// traceback function should not make assumptions about when it is
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

// cgoTraceback prints a traceback of callers.
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

	var arg cgoSymbolizerArg
	for _, c := range callers {
		if c == 0 {
			break
		}
		printOneCgoTraceback(c, 0x7fffffff, &arg)
	}
	arg.pc = 0
	callCgoSymbolizer(&arg)
}

// printOneCgoTraceback prints the traceback of a single cgo caller.
// This can print more than one line because of inlining.
// Returns the number of frames printed.
func printOneCgoTraceback(pc uintptr, max int, arg *cgoSymbolizerArg) int {
	c := 0
	arg.pc = pc
	for {
		if c > max {
			break
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
		c++
		if arg.more == 0 {
			break
		}
	}
	return c
}

// callCgoSymbolizer calls the cgoSymbolizer function.
func callCgoSymbolizer(arg *cgoSymbolizerArg) {
	call := cgocall
	if panicking > 0 || getg().m.curg != getg() {
		// We do not want to call into the scheduler when panicking
		// or when on the system stack.
		call = asmcgocall
	}
	if msanenabled {
		msanwrite(unsafe.Pointer(arg), unsafe.Sizeof(cgoSymbolizerArg{}))
	}
	call(cgoSymbolizer, noescape(unsafe.Pointer(arg)))
}

// cgoContextPCs gets the PC values from a cgo traceback.
func cgoContextPCs(ctxt uintptr, buf []uintptr) {
	if cgoTraceback == nil {
		return
	}
	call := cgocall
	if panicking > 0 || getg().m.curg != getg() {
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
	call(cgoTraceback, noescape(unsafe.Pointer(&arg)))
}
