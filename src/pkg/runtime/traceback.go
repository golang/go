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

// jmpdeferPC is the PC at the beginning of the jmpdefer assembly function.
// The traceback needs to recognize it on link register architectures.
var jmpdeferPC uintptr

func init() {
	f := jmpdefer
	jmpdeferPC = **(**uintptr)(unsafe.Pointer(&f))
}

// System-specific hook. See traceback_windows.go
var systraceback func(*_func, *stkframe, *g, bool, func(*stkframe, unsafe.Pointer) bool, unsafe.Pointer) (changed, aborted bool)

// Generic traceback.  Handles runtime stack prints (pcbuf == nil),
// the runtime.Callers function (pcbuf != nil), as well as the garbage
// collector (callback != nil).  A little clunky to merge these, but avoids
// duplicating the code and all its subtlety.
func gentraceback(pc0 uintptr, sp0 uintptr, lr0 uintptr, gp *g, skip int, pcbuf *uintptr, max int, callback func(*stkframe, unsafe.Pointer) bool, v unsafe.Pointer, printall bool) int {
	g := getg()
	gotraceback := gotraceback(nil)
	if pc0 == ^uintptr(0) && sp0 == ^uintptr(0) { // Signal to fetch saved values from gp.
		if gp.syscallstack != 0 {
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
	wasnewproc := false
	printing := pcbuf == nil && callback == nil
	panic := gp._panic
	_defer := gp._defer

	for _defer != nil && uintptr(_defer.argp) == _NoArgs {
		_defer = _defer.link
	}
	for panic != nil && panic._defer == nil {
		panic = panic.link
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
	stk := (*stktop)(unsafe.Pointer(gp.stackbase))
	for n < max {
		// Typically:
		//	pc is the PC of the running function.
		//	sp is the stack pointer at that program counter.
		//	fp is the frame pointer (caller's stack pointer) at that program counter, or nil if unknown.
		//	stk is the stack containing sp.
		//	The caller's program counter is lr, unless lr is zero, in which case it is *(uintptr*)sp.
		if frame.pc == uintptr(unsafe.Pointer(&lessstack)) {
			// Hit top of stack segment.  Unwind to next segment.
			frame.pc = stk.gobuf.pc
			frame.sp = stk.gobuf.sp
			frame.lr = 0
			frame.fp = 0
			if printing && showframe(nil, gp) {
				print("----- stack segment boundary -----\n")
			}
			stk = (*stktop)(unsafe.Pointer(stk.stackbase))
			f = findfunc(frame.pc)
			if f == nil {
				print("runtime: unknown pc ", hex(frame.pc), " after stack split\n")
				if callback != nil {
					gothrow("unknown pc")
				}
			}
			frame.fn = f
			continue
		}
		f = frame.fn

		// Hook for handling Windows exception handlers. See traceback_windows.go.
		if systraceback != nil {
			changed, aborted := systraceback(f, (*stkframe)(noescape(unsafe.Pointer(&frame))), gp, printing, callback, v)
			if aborted {
				return n
			}
			if changed {
				continue
			}
		}

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
			if f.args != _ArgsSizeUnknown {
				frame.arglen = uintptr(f.args)
			} else if flr == nil {
				frame.arglen = 0
			} else if frame.lr == uintptr(unsafe.Pointer(&lessstack)) {
				frame.arglen = uintptr(stk.argsize)
			} else {
				i := funcarglen(flr, frame.lr)
				if i >= 0 {
					frame.arglen = uintptr(i)
				} else {
					var tmp string
					if flr != nil {
						tmp = gofuncname(flr)
					} else {
						tmp = "?"
					}
					print("runtime: unknown argument frame size for ", gofuncname(f), " called from ", hex(frame.lr), " [", tmp, "]\n")
					if callback != nil {
						gothrow("invalid stack")
					}
					frame.arglen = 0
				}
			}
		}

		// Determine function SP where deferproc would find its arguments.
		var sparg uintptr
		if usesLR {
			// On link register architectures, that's the standard bottom-of-stack plus 1 word
			// for the saved LR. If the previous frame was a direct call to newproc/deferproc,
			// however, the SP is three words lower than normal.
			// If the function has no frame at all - perhaps it just started, or perhaps
			// it is a leaf with no local variables - then we cannot possibly find its
			// SP in a defer, and we might confuse its SP for its caller's SP, so
			// leave sparg=0 in that case.
			if frame.fp != frame.sp {
				sparg = frame.sp + regSize
				if wasnewproc {
					sparg += 3 * regSize
				}
			}
		} else {
			// On x86 that's the standard bottom-of-stack, so SP exactly.
			// If the previous frame was a direct call to newproc/deferproc, however,
			// the SP is two words lower than normal.
			sparg = frame.sp
			if wasnewproc {
				sparg += 2 * ptrSize
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
		// It suffices to assume that the most recent deferproc is the one that
		// returns; everything live at earlier deferprocs is still live at that one.
		frame.continpc = frame.pc
		if waspanic {
			if panic != nil && panic._defer.argp == sparg {
				frame.continpc = panic._defer.pc
			} else if _defer != nil && _defer.argp == sparg {
				frame.continpc = _defer.pc
			} else {
				frame.continpc = 0
			}
		}

		// Unwind our local panic & defer stacks past this frame.
		for panic != nil && (panic._defer == nil || panic._defer.argp == sparg || panic._defer.argp == _NoArgs) {
			panic = panic.link
		}
		for _defer != nil && (_defer.argp == sparg || _defer.argp == _NoArgs) {
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
			if printall || showframe(f, gp) {
				// Print during crash.
				//	main(0x1, 0x2, 0x3)
				//		/home/rsc/go/src/runtime/x.go:23 +0xf
				//
				tracepc := frame.pc // back up to CALL instruction for funcline.
				if n > 0 && frame.pc > f.entry && !waspanic {
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
				var file string
				line := funcline(f, tracepc, &file)
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
		waspanic = f.entry == uintptr(unsafe.Pointer(&sigpanic))
		wasnewproc = f.entry == uintptr(unsafe.Pointer(&newproc)) || f.entry == uintptr(unsafe.Pointer(&deferproc))

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
			print("runtime: g", gp.goid, ": leftover defer argp=", hex(_defer.argp), " pc=", hex(_defer.pc), "\n")
		}
		if panic != nil {
			print("runtime: g", gp.goid, ": leftover panic argp=", hex(panic._defer.argp), " pc=", hex(panic._defer.pc), "\n")
		}
		for _defer = gp._defer; _defer != nil; _defer = _defer.link {
			print("\tdefer ", _defer, " argp=", hex(_defer.argp), " pc=", hex(_defer.pc), "\n")
		}
		for panic = gp._panic; panic != nil; panic = panic.link {
			print("\tpanic ", panic, " defer ", panic._defer)
			if panic._defer != nil {
				print(" argp=", hex(panic._defer.argp), " pc=", hex(panic._defer.pc))
			}
			print("\n")
		}
		gothrow("traceback has leftover defers or panics")
	}

	return n
}

func showframe(*_func, *g) bool

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
		var file string
		line := funcline(f, tracepc, &file)
		print("\t", file, ":", line)
		if pc > f.entry {
			print(" +", hex(pc-f.entry))
		}
		print("\n")
	}
}

func traceback(pc uintptr, sp uintptr, lr uintptr, gp *g) {
	var n int
	if readgstatus(gp)&^_Gscan == _Gsyscall {
		// Override signal registers if blocked in system call.
		pc = gp.syscallpc
		sp = gp.syscallsp
	}
	// Print traceback. By default, omits runtime frames.
	// If that means we print nothing at all, repeat forcing all frames printed.
	n = gentraceback(pc, sp, 0, gp, 0, nil, _TracebackMaxFrames, nil, nil, false)
	if n == 0 {
		n = gentraceback(pc, sp, 0, gp, 0, nil, _TracebackMaxFrames, nil, nil, true)
	}
	if n == _TracebackMaxFrames {
		print("...additional frames elided...\n")
	}
	printcreatedby(gp)
}

func callers(skip int, pcbuf *uintptr, m int) int {
	sp := getcallersp(unsafe.Pointer(&skip))
	pc := uintptr(getcallerpc(unsafe.Pointer(&skip)))
	return gentraceback(pc, sp, 0, getg(), skip, pcbuf, m, nil, nil, false)
}

func gcallers(gp *g, skip int, pcbuf *uintptr, m int) int {
	return gentraceback(^uintptr(0), ^uintptr(0), 0, gp, skip, pcbuf, m, nil, nil, false)
}
