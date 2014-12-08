// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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
// usesLR is defined below. ptrSize and regSize are defined in stubs.go.

const usesLR = GOARCH != "amd64" && GOARCH != "amd64p32" && GOARCH != "386"

var (
	// initialized in tracebackinit
	goexitPC             uintptr
	jmpdeferPC           uintptr
	mcallPC              uintptr
	morestackPC          uintptr
	mstartPC             uintptr
	rt0_goPC             uintptr
	sigpanicPC           uintptr
	systemstack_switchPC uintptr

	externalthreadhandlerp uintptr // initialized elsewhere
)

func tracebackinit() {
	// Go variable initialization happens late during runtime startup.
	// Instead of initializing the variables above in the declarations,
	// schedinit calls this function so that the variables are
	// initialized and available earlier in the startup sequence.
	goexitPC = funcPC(goexit)
	jmpdeferPC = funcPC(jmpdefer)
	mcallPC = funcPC(mcall)
	morestackPC = funcPC(morestack)
	mstartPC = funcPC(mstart)
	rt0_goPC = funcPC(rt0_go)
	sigpanicPC = funcPC(sigpanic)
	systemstack_switchPC = funcPC(systemstack_switch)
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
			frame.fn = nil
			frame.argp = 0
			frame.arglen = 0
			frame.argmap = nil
		} else {
			frame.pc = uintptr(fn.fn)
			f := findfunc(frame.pc)
			if f == nil {
				print("runtime: unknown pc in defer ", hex(frame.pc), "\n")
				gothrow("unknown pc")
			}
			frame.fn = f
			frame.argp = uintptr(deferArgs(d))
			setArgInfo(&frame, f, true)
		}
		frame.continpc = frame.pc
		if !callback((*stkframe)(noescape(unsafe.Pointer(&frame))), v) {
			return
		}
	}
}

// Generic traceback.  Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
func gentraceback(pc0 uintptr, sp0 uintptr, lr0 uintptr, gp *g, skip int, pcbuf *uintptr, max int, callback func(*stkframe, unsafe.Pointer) bool, v unsafe.Pointer, flags uint) int {
	if goexitPC == 0 {
		gothrow("gentraceback before goexitPC initialization")
	}
	g := getg()
	if g == gp && g == g.m.curg {
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
		gothrow("gentraceback cannot trace user goroutine on its own stack")
	}
	gotraceback := gotraceback(nil)
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
	printing := pcbuf == nil && callback == nil
	_defer := gp._defer

	for _defer != nil && uintptr(_defer.sp) == _NoArgs {
		_defer = _defer.link
	}

	// If the PC is zero, it's likely a nil function call.
	// Start in the caller's frame.
	if frame.pc == 0 {
		if usesLR {
			frame.pc = *(*uintptr)(unsafe.Pointer(frame.sp))
			frame.lr = 0
		} else {
			frame.pc = uintptr(*(*uintreg)(unsafe.Pointer(frame.sp)))
			frame.sp += regSize
		}
	}

	f := findfunc(frame.pc)
	if f == nil {
		if callback != nil {
			print("runtime: unknown pc ", hex(frame.pc), "\n")
			gothrow("unknown pc")
		}
		return 0
	}
	frame.fn = f

	n := 0
	for n < max {
		// Typically:
		//	pc is the PC of the running function.
		//	sp is the stack pointer at that program counter.
		//	fp is the frame pointer (caller's stack pointer) at that program counter, or nil if unknown.
		//	stk is the stack containing sp.
		//	The caller's program counter is lr, unless lr is zero, in which case it is *(uintptr*)sp.
		f = frame.fn

		// Found an actual function.
		// Derive frame pointer and link register.
		if frame.fp == 0 {
			frame.fp = frame.sp + uintptr(funcspdelta(f, frame.pc))
			if !usesLR {
				// On x86, call instruction pushes return PC before entering new function.
				frame.fp += regSize
			}
		}
		var flr *_func
		if topofstack(f) {
			frame.lr = 0
			flr = nil
		} else if usesLR && f.entry == jmpdeferPC {
			// jmpdefer modifies SP/LR/PC non-atomically.
			// If a profiling interrupt arrives during jmpdefer,
			// the stack unwind may see a mismatched register set
			// and get confused. Stop if we see PC within jmpdefer
			// to avoid that confusion.
			// See golang.org/issue/8153.
			if callback != nil {
				gothrow("traceback_arm: found jmpdefer when tracing with callback")
			}
			frame.lr = 0
		} else {
			if usesLR {
				if n == 0 && frame.sp < frame.fp || frame.lr == 0 {
					frame.lr = *(*uintptr)(unsafe.Pointer(frame.sp))
				}
			} else {
				if frame.lr == 0 {
					frame.lr = uintptr(*(*uintreg)(unsafe.Pointer(frame.fp - regSize)))
				}
			}
			flr = findfunc(frame.lr)
			if flr == nil {
				// This happens if you get a profiling interrupt at just the wrong time.
				// In that context it is okay to stop early.
				// But if callback is set, we're doing a garbage collection and must
				// get everything, so crash loudly.
				if callback != nil {
					print("runtime: unexpected return pc for ", gofuncname(f), " called from ", hex(frame.lr), "\n")
					gothrow("unknown caller pc")
				}
			}
		}

		frame.varp = frame.fp
		if !usesLR {
			// On x86, call instruction pushes return PC before entering new function.
			frame.varp -= regSize
		}

		// Derive size of arguments.
		// Most functions have a fixed-size argument block,
		// so we can use metadata about the function f.
		// Not all, though: there are some variadic functions
		// in package runtime and reflect, and for those we use call-specific
		// metadata recorded by f's caller.
		if callback != nil || printing {
			frame.argp = frame.fp
			if usesLR {
				frame.argp += ptrSize
			}
			setArgInfo(&frame, f, callback != nil)
		}

		// Determine frame's 'continuation PC', where it can continue.
		// Normally this is the return address on the stack, but if sigpanic
		// is immediately below this function on the stack, then the frame
		// stopped executing due to a trap, and frame.pc is probably not
		// a safe point for looking up liveness information. In this panicking case,
		// the function either doesn't return at all (if it has no defers or if the
		// defers do not recover) or it returns from one of the calls to
		// deferproc a second time (if the corresponding deferred func recovers).
		// It suffices to assume that the most recent deferproc is the one that
		// returns; everything live at earlier deferprocs is still live at that one.
		frame.continpc = frame.pc
		if waspanic {
			if _defer != nil && _defer.sp == frame.sp {
				frame.continpc = _defer.pc
			} else {
				frame.continpc = 0
			}
		}

		// Unwind our local defer stack past this frame.
		for _defer != nil && (_defer.sp == frame.sp || _defer.sp == _NoArgs) {
			_defer = _defer.link
		}

		if skip > 0 {
			skip--
			goto skipped
		}

		if pcbuf != nil {
			(*[1 << 20]uintptr)(unsafe.Pointer(pcbuf))[n] = frame.pc
		}
		if callback != nil {
			if !callback((*stkframe)(noescape(unsafe.Pointer(&frame))), v) {
				return n
			}
		}
		if printing {
			if (flags&_TraceRuntimeFrames) != 0 || showframe(f, gp) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//
				tracepc := frame.pc // back up to CALL instruction for funcline.
				if (n > 0 || flags&_TraceTrap == 0) && frame.pc > f.entry && !waspanic {
					tracepc--
				}
				print(gofuncname(f), "(")
				argp := (*[100]uintptr)(unsafe.Pointer(frame.argp))
				for i := uintptr(0); i < frame.arglen/ptrSize; i++ {
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
				file, line := funcline(f, tracepc)
				print("\t", file, ":", line)
				if frame.pc > f.entry {
					print(" +", hex(frame.pc-f.entry))
				}
				if g.m.throwing > 0 && gp == g.m.curg || gotraceback >= 2 {
					print(" fp=", hex(frame.fp), " sp=", hex(frame.sp))
				}
				print("\n")
				nprint++
			}
		}
		n++

	skipped:
		waspanic = f.entry == sigpanicPC

		// Do not unwind past the bottom of the stack.
		if flr == nil {
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
			frame.sp += ptrSize
			f = findfunc(frame.pc)
			frame.fn = f
			if f == nil {
				frame.pc = x
			} else if f.frame == 0 {
				frame.lr = x
			}
		}
	}

	if pcbuf == nil && callback == nil {
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
		if _defer != nil {
			print("runtime: g", gp.goid, ": leftover defer sp=", hex(_defer.sp), " pc=", hex(_defer.pc), "\n")
		}
		for _defer = gp._defer; _defer != nil; _defer = _defer.link {
			print("\tdefer ", _defer, " sp=", hex(_defer.sp), " pc=", hex(_defer.pc), "\n")
		}
		gothrow("traceback has leftover defers")
	}

	return n
}

func setArgInfo(frame *stkframe, f *_func, needArgMap bool) {
	frame.arglen = uintptr(f.args)
	if needArgMap && f.args == _ArgsSizeUnknown {
		// Extract argument bitmaps for reflect stubs from the calls they made to reflect.
		switch gofuncname(f) {
		case "reflect.makeFuncStub", "reflect.methodValueCall":
			arg0 := frame.sp
			if usesLR {
				arg0 += ptrSize
			}
			fn := *(**[2]uintptr)(unsafe.Pointer(arg0))
			if fn[0] != f.entry {
				print("runtime: confused by ", gofuncname(f), "\n")
				gothrow("reflect mismatch")
			}
			bv := (*bitvector)(unsafe.Pointer(fn[1]))
			frame.arglen = uintptr(bv.n / 2 * ptrSize)
			frame.argmap = bv
		}
	}
}

func printcreatedby(gp *g) {
	// Show what created goroutine, except main goroutine (goid 1).
	pc := gp.gopc
	f := findfunc(pc)
	if f != nil && showframe(f, gp) && gp.goid != 1 {
		print("created by ", gofuncname(f), "\n")
		tracepc := pc // back up to CALL instruction for funcline.
		if pc > f.entry {
			tracepc -= _PCQuantum
		}
		file, line := funcline(f, tracepc)
		print("\t", file, ":", line)
		if pc > f.entry {
			print(" +", hex(pc-f.entry))
		}
		print("\n")
	}
}

func traceback(pc uintptr, sp uintptr, lr uintptr, gp *g) {
	traceback1(pc, sp, lr, gp, 0)
}

// tracebacktrap is like traceback but expects that the PC and SP were obtained
// from a trap, not from gp->sched or gp->syscallpc/gp->syscallsp or getcallerpc/getcallersp.
// Because they are from a trap instead of from a saved pair,
// the initial PC must not be rewound to the previous instruction.
// (All the saved pairs record a PC that is a return address, so we
// rewind it into the CALL instruction.)
func tracebacktrap(pc uintptr, sp uintptr, lr uintptr, gp *g) {
	traceback1(pc, sp, lr, gp, _TraceTrap)
}

func traceback1(pc uintptr, sp uintptr, lr uintptr, gp *g, flags uint) {
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
}

func callers(skip int, pcbuf *uintptr, m int) int {
	sp := getcallersp(unsafe.Pointer(&skip))
	pc := uintptr(getcallerpc(unsafe.Pointer(&skip)))
	var n int
	systemstack(func() {
		n = gentraceback(pc, sp, 0, getg(), skip, pcbuf, m, nil, nil, 0)
	})
	return n
}

func gcallers(gp *g, skip int, pcbuf *uintptr, m int) int {
	return gentraceback(^uintptr(0), ^uintptr(0), 0, gp, skip, pcbuf, m, nil, nil, 0)
}

func showframe(f *_func, gp *g) bool {
	g := getg()
	if g.m.throwing > 0 && gp != nil && (gp == g.m.curg || gp == g.m.caughtsig) {
		return true
	}
	traceback := gotraceback(nil)
	name := gostringnocopy(funcname(f))

	// Special case: always show runtime.panic frame, so that we can
	// see where a panic started in the middle of a stack trace.
	// See golang.org/issue/5832.
	if name == "runtime.panic" {
		return true
	}

	return traceback > 1 || f != nil && contains(name, ".") && (!hasprefix(name, "runtime.") || isExportedRuntime(name))
}

// isExportedRuntime reports whether name is an exported runtime function.
// It is only for runtime functions, so ASCII A-Z is fine.
func isExportedRuntime(name string) bool {
	const n = len("runtime.")
	return len(name) > n && name[:n] == "runtime." && 'A' <= name[n] && name[n] <= 'Z'
}

var gStatusStrings = [...]string{
	_Gidle:      "idle",
	_Grunnable:  "runnable",
	_Grunning:   "running",
	_Gsyscall:   "syscall",
	_Gwaiting:   "waiting",
	_Gdead:      "dead",
	_Genqueue:   "enqueue",
	_Gcopystack: "copystack",
}

var gScanStatusStrings = [...]string{
	0:          "scan",
	_Grunnable: "scanrunnable",
	_Grunning:  "scanrunning",
	_Gsyscall:  "scansyscall",
	_Gwaiting:  "scanwaiting",
	_Gdead:     "scandead",
	_Genqueue:  "scanenqueue",
}

func goroutineheader(gp *g) {
	gpstatus := readgstatus(gp)

	// Basic string status
	var status string
	if 0 <= gpstatus && gpstatus < uint32(len(gStatusStrings)) {
		status = gStatusStrings[gpstatus]
	} else if gpstatus&_Gscan != 0 && 0 <= gpstatus&^_Gscan && gpstatus&^_Gscan < uint32(len(gStatusStrings)) {
		status = gStatusStrings[gpstatus&^_Gscan]
	} else {
		status = "???"
	}

	// Override.
	if (gpstatus == _Gwaiting || gpstatus == _Gscanwaiting) && gp.waitreason != "" {
		status = gp.waitreason
	}

	// approx time the G is blocked, in minutes
	var waitfor int64
	gpstatus &^= _Gscan // drop the scan bit
	if (gpstatus == _Gwaiting || gpstatus == _Gsyscall) && gp.waitsince != 0 {
		waitfor = (nanotime() - gp.waitsince) / 60e9
	}
	print("goroutine ", gp.goid, " [", status)
	if waitfor >= 1 {
		print(", ", waitfor, " minutes")
	}
	if gp.lockedm != nil {
		print(", locked to thread")
	}
	print("]:\n")
}

func tracebackothers(me *g) {
	level := gotraceback(nil)

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
		if gp == me || gp == g.m.curg || readgstatus(gp) == _Gdead || gp.issystem && level < 2 {
			continue
		}
		print("\n")
		goroutineheader(gp)
		if readgstatus(gp)&^_Gscan == _Grunning {
			print("\tgoroutine running on other thread; stack unavailable\n")
			printcreatedby(gp)
		} else {
			traceback(^uintptr(0), ^uintptr(0), 0, gp)
		}
	}
	unlock(&allglock)
}

// Does f mark the top of a goroutine stack?
func topofstack(f *_func) bool {
	pc := f.entry
	return pc == goexitPC ||
		pc == mstartPC ||
		pc == mcallPC ||
		pc == morestackPC ||
		pc == rt0_goPC ||
		externalthreadhandlerp != 0 && pc == externalthreadhandlerp
}
