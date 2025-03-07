// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Goroutine preemption
//
// A goroutine can be preempted at any safe-point. Currently, there
// are a few categories of safe-points:
//
// 1. A blocked safe-point occurs for the duration that a goroutine is
//    descheduled, blocked on synchronization, or in a system call.
//
// 2. Synchronous safe-points occur when a running goroutine checks
//    for a preemption request.
//
// 3. Asynchronous safe-points occur at any instruction in user code
//    where the goroutine can be safely paused and a conservative
//    stack and register scan can find stack roots. The runtime can
//    stop a goroutine at an async safe-point using a signal.
//
// At both blocked and synchronous safe-points, a goroutine's CPU
// state is minimal and the garbage collector has complete information
// about its entire stack. This makes it possible to deschedule a
// goroutine with minimal space, and to precisely scan a goroutine's
// stack.
//
// Synchronous safe-points are implemented by overloading the stack
// bound check in function prologues. To preempt a goroutine at the
// next synchronous safe-point, the runtime poisons the goroutine's
// stack bound to a value that will cause the next stack bound check
// to fail and enter the stack growth implementation, which will
// detect that it was actually a preemption and redirect to preemption
// handling.
//
// Preemption at asynchronous safe-points is implemented by suspending
// the thread using an OS mechanism (e.g., signals) and inspecting its
// state to determine if the goroutine was at an asynchronous
// safe-point. Since the thread suspension itself is generally
// asynchronous, it also checks if the running goroutine wants to be
// preempted, since this could have changed. If all conditions are
// satisfied, it adjusts the signal context to make it look like the
// signaled thread just called asyncPreempt and resumes the thread.
// asyncPreempt spills all registers and enters the scheduler.
//
// (An alternative would be to preempt in the signal handler itself.
// This would let the OS save and restore the register state and the
// runtime would only need to know how to extract potentially
// pointer-containing registers from the signal context. However, this
// would consume an M for every preempted G, and the scheduler itself
// is not designed to run from a signal handler, as it tends to
// allocate memory and start threads in the preemption path.)

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/stringslite"
)

type suspendGState struct {
	g *g

	// dead indicates the goroutine was not suspended because it
	// is dead. This goroutine could be reused after the dead
	// state was observed, so the caller must not assume that it
	// remains dead.
	dead bool

	// stopped indicates that this suspendG transitioned the G to
	// _Gwaiting via g.preemptStop and thus is responsible for
	// readying it when done.
	stopped bool
}

// suspendG suspends goroutine gp at a safe-point and returns the
// state of the suspended goroutine. The caller gets read access to
// the goroutine until it calls resumeG.
//
// It is safe for multiple callers to attempt to suspend the same
// goroutine at the same time. The goroutine may execute between
// subsequent successful suspend operations. The current
// implementation grants exclusive access to the goroutine, and hence
// multiple callers will serialize. However, the intent is to grant
// shared read access, so please don't depend on exclusive access.
//
// This must be called from the system stack and the user goroutine on
// the current M (if any) must be in a preemptible state. This
// prevents deadlocks where two goroutines attempt to suspend each
// other and both are in non-preemptible states. There are other ways
// to resolve this deadlock, but this seems simplest.
//
// TODO(austin): What if we instead required this to be called from a
// user goroutine? Then we could deschedule the goroutine while
// waiting instead of blocking the thread. If two goroutines tried to
// suspend each other, one of them would win and the other wouldn't
// complete the suspend until it was resumed. We would have to be
// careful that they couldn't actually queue up suspend for each other
// and then both be suspended. This would also avoid the need for a
// kernel context switch in the synchronous case because we could just
// directly schedule the waiter. The context switch is unavoidable in
// the signal case.
//
//go:systemstack
func suspendG(gp *g) suspendGState {
	if mp := getg().m; mp.curg != nil && readgstatus(mp.curg) == _Grunning {
		// Since we're on the system stack of this M, the user
		// G is stuck at an unsafe point. If another goroutine
		// were to try to preempt m.curg, it could deadlock.
		throw("suspendG from non-preemptible goroutine")
	}

	// See https://golang.org/cl/21503 for justification of the yield delay.
	const yieldDelay = 10 * 1000
	var nextYield int64

	// Drive the goroutine to a preemption point.
	stopped := false
	var asyncM *m
	var asyncGen uint32
	var nextPreemptM int64
	for i := 0; ; i++ {
		switch s := readgstatus(gp); s {
		default:
			if s&_Gscan != 0 {
				// Someone else is suspending it. Wait
				// for them to finish.
				//
				// TODO: It would be nicer if we could
				// coalesce suspends.
				break
			}

			dumpgstatus(gp)
			throw("invalid g status")

		case _Gdead:
			// Nothing to suspend.
			//
			// preemptStop may need to be cleared, but
			// doing that here could race with goroutine
			// reuse. Instead, goexit0 clears it.
			return suspendGState{dead: true}

		case _Gcopystack:
			// The stack is being copied. We need to wait
			// until this is done.

		case _Gpreempted:
			// We (or someone else) suspended the G. Claim
			// ownership of it by transitioning it to
			// _Gwaiting.
			if !casGFromPreempted(gp, _Gpreempted, _Gwaiting) {
				break
			}

			// We stopped the G, so we have to ready it later.
			stopped = true

			s = _Gwaiting
			fallthrough

		case _Grunnable, _Gsyscall, _Gwaiting:
			// Claim goroutine by setting scan bit.
			// This may race with execution or readying of gp.
			// The scan bit keeps it from transition state.
			if !castogscanstatus(gp, s, s|_Gscan) {
				break
			}

			// Clear the preemption request. It's safe to
			// reset the stack guard because we hold the
			// _Gscan bit and thus own the stack.
			gp.preemptStop = false
			gp.preempt = false
			gp.stackguard0 = gp.stack.lo + stackGuard

			// The goroutine was already at a safe-point
			// and we've now locked that in.
			//
			// TODO: It would be much better if we didn't
			// leave it in _Gscan, but instead gently
			// prevented its scheduling until resumption.
			// Maybe we only use this to bump a suspended
			// count and the scheduler skips suspended
			// goroutines? That wouldn't be enough for
			// {_Gsyscall,_Gwaiting} -> _Grunning. Maybe
			// for all those transitions we need to check
			// suspended and deschedule?
			return suspendGState{g: gp, stopped: stopped}

		case _Grunning:
			// Optimization: if there is already a pending preemption request
			// (from the previous loop iteration), don't bother with the atomics.
			if gp.preemptStop && gp.preempt && gp.stackguard0 == stackPreempt && asyncM == gp.m && asyncM.preemptGen.Load() == asyncGen {
				break
			}

			// Temporarily block state transitions.
			if !castogscanstatus(gp, _Grunning, _Gscanrunning) {
				break
			}

			// Request synchronous preemption.
			gp.preemptStop = true
			gp.preempt = true
			gp.stackguard0 = stackPreempt

			// Prepare for asynchronous preemption.
			asyncM2 := gp.m
			asyncGen2 := asyncM2.preemptGen.Load()
			needAsync := asyncM != asyncM2 || asyncGen != asyncGen2
			asyncM = asyncM2
			asyncGen = asyncGen2

			casfrom_Gscanstatus(gp, _Gscanrunning, _Grunning)

			// Send asynchronous preemption. We do this
			// after CASing the G back to _Grunning
			// because preemptM may be synchronous and we
			// don't want to catch the G just spinning on
			// its status.
			if preemptMSupported && debug.asyncpreemptoff == 0 && needAsync {
				// Rate limit preemptM calls. This is
				// particularly important on Windows
				// where preemptM is actually
				// synchronous and the spin loop here
				// can lead to live-lock.
				now := nanotime()
				if now >= nextPreemptM {
					nextPreemptM = now + yieldDelay/2
					preemptM(asyncM)
				}
			}
		}

		// TODO: Don't busy wait. This loop should really only
		// be a simple read/decide/CAS loop that only fails if
		// there's an active race. Once the CAS succeeds, we
		// should queue up the preemption (which will require
		// it to be reliable in the _Grunning case, not
		// best-effort) and then sleep until we're notified
		// that the goroutine is suspended.
		if i == 0 {
			nextYield = nanotime() + yieldDelay
		}
		if nanotime() < nextYield {
			procyield(10)
		} else {
			osyield()
			nextYield = nanotime() + yieldDelay/2
		}
	}
}

// resumeG undoes the effects of suspendG, allowing the suspended
// goroutine to continue from its current safe-point.
func resumeG(state suspendGState) {
	if state.dead {
		// We didn't actually stop anything.
		return
	}

	gp := state.g
	switch s := readgstatus(gp); s {
	default:
		dumpgstatus(gp)
		throw("unexpected g status")

	case _Grunnable | _Gscan,
		_Gwaiting | _Gscan,
		_Gsyscall | _Gscan:
		casfrom_Gscanstatus(gp, s, s&^_Gscan)
	}

	if state.stopped {
		// We stopped it, so we need to re-schedule it.
		ready(gp, 0, true)
	}
}

// canPreemptM reports whether mp is in a state that is safe to preempt.
//
// It is nosplit because it has nosplit callers.
//
//go:nosplit
func canPreemptM(mp *m) bool {
	return mp.locks == 0 && mp.mallocing == 0 && mp.preemptoff == "" && mp.p.ptr().status == _Prunning
}

//go:generate go run mkpreempt.go

// asyncPreempt saves all user registers and calls asyncPreempt2.
//
// When stack scanning encounters an asyncPreempt frame, it scans that
// frame and its parent frame conservatively.
//
// asyncPreempt is implemented in assembly.
func asyncPreempt()

//go:nosplit
func asyncPreempt2() {
	gp := getg()
	gp.asyncSafePoint = true
	if gp.preemptStop {
		mcall(preemptPark)
	} else {
		mcall(gopreempt_m)
	}
	gp.asyncSafePoint = false
}

// asyncPreemptStack is the bytes of stack space required to inject an
// asyncPreempt call.
var asyncPreemptStack = ^uintptr(0)

func init() {
	f := findfunc(abi.FuncPCABI0(asyncPreempt))
	total := funcMaxSPDelta(f)
	f = findfunc(abi.FuncPCABIInternal(asyncPreempt2))
	total += funcMaxSPDelta(f)
	// Add some overhead for return PCs, etc.
	asyncPreemptStack = uintptr(total) + 8*goarch.PtrSize
	if asyncPreemptStack > stackNosplit {
		// We need more than the nosplit limit. This isn't
		// unsafe, but it may limit asynchronous preemption.
		//
		// This may be a problem if we start using more
		// registers. In that case, we should store registers
		// in a context object. If we pre-allocate one per P,
		// asyncPreempt can spill just a few registers to the
		// stack, then grab its context object and spill into
		// it. When it enters the runtime, it would allocate a
		// new context for the P.
		print("runtime: asyncPreemptStack=", asyncPreemptStack, "\n")
		throw("async stack too large")
	}
}

// wantAsyncPreempt returns whether an asynchronous preemption is
// queued for gp.
func wantAsyncPreempt(gp *g) bool {
	// Check both the G and the P.
	return (gp.preempt || gp.m.p != 0 && gp.m.p.ptr().preempt) && readgstatus(gp)&^_Gscan == _Grunning
}

// isAsyncSafePoint reports whether gp at instruction PC is an
// asynchronous safe point. This indicates that:
//
// 1. It's safe to suspend gp and conservatively scan its stack and
// registers. There are no potentially hidden pointer values and it's
// not in the middle of an atomic sequence like a write barrier.
//
// 2. gp has enough stack space to inject the asyncPreempt call.
//
// 3. It's generally safe to interact with the runtime, even if we're
// in a signal handler stopped here. For example, there are no runtime
// locks held, so acquiring a runtime lock won't self-deadlock.
//
// In some cases the PC is safe for asynchronous preemption but it
// also needs to adjust the resumption PC. The new PC is returned in
// the second result.
func isAsyncSafePoint(gp *g, pc, sp, lr uintptr) (bool, uintptr) {
	mp := gp.m

	// Only user Gs can have safe-points. We check this first
	// because it's extremely common that we'll catch mp in the
	// scheduler processing this G preemption.
	if mp.curg != gp {
		return false, 0
	}

	// Check M state.
	if mp.p == 0 || !canPreemptM(mp) {
		return false, 0
	}

	// Check stack space.
	if sp < gp.stack.lo || sp-gp.stack.lo < asyncPreemptStack {
		return false, 0
	}

	// Check if PC is an unsafe-point.
	f := findfunc(pc)
	if !f.valid() {
		// Not Go code.
		return false, 0
	}
	if (GOARCH == "mips" || GOARCH == "mipsle" || GOARCH == "mips64" || GOARCH == "mips64le") && lr == pc+8 && funcspdelta(f, pc) == 0 {
		// We probably stopped at a half-executed CALL instruction,
		// where the LR is updated but the PC has not. If we preempt
		// here we'll see a seemingly self-recursive call, which is in
		// fact not.
		// This is normally ok, as we use the return address saved on
		// stack for unwinding, not the LR value. But if this is a
		// call to morestack, we haven't created the frame, and we'll
		// use the LR for unwinding, which will be bad.
		return false, 0
	}
	up, startpc := pcdatavalue2(f, abi.PCDATA_UnsafePoint, pc)
	if up == abi.UnsafePointUnsafe {
		// Unsafe-point marked by compiler. This includes
		// atomic sequences (e.g., write barrier) and nosplit
		// functions (except at calls).
		return false, 0
	}
	if fd := funcdata(f, abi.FUNCDATA_LocalsPointerMaps); fd == nil || f.flag&abi.FuncFlagAsm != 0 {
		// This is assembly code. Don't assume it's well-formed.
		// TODO: Empirically we still need the fd == nil check. Why?
		//
		// TODO: Are there cases that are safe but don't have a
		// locals pointer map, like empty frame functions?
		// It might be possible to preempt any assembly functions
		// except the ones that have funcFlag_SPWRITE set in f.flag.
		return false, 0
	}
	// Check the inner-most name
	u, uf := newInlineUnwinder(f, pc)
	name := u.srcFunc(uf).name()
	if stringslite.HasPrefix(name, "runtime.") ||
		stringslite.HasPrefix(name, "internal/runtime/") ||
		stringslite.HasPrefix(name, "reflect.") {
		// For now we never async preempt the runtime or
		// anything closely tied to the runtime. Known issues
		// include: various points in the scheduler ("don't
		// preempt between here and here"), much of the defer
		// implementation (untyped info on stack), bulk write
		// barriers (write barrier check), atomic functions in
		// internal/runtime/atomic, reflect.{makeFuncStub,methodValueCall}.
		//
		// Note that this is a subset of the runtimePkgs in pkgspecial.go
		// and these checks are theoretically redundant because the compiler
		// marks "all points" in runtime functions as unsafe for async preemption.
		// But for some reason, we can't eliminate these checks until https://go.dev/issue/72031
		// is resolved.
		//
		// TODO(austin): We should improve this, or opt things
		// in incrementally.
		return false, 0
	}
	switch up {
	case abi.UnsafePointRestart1, abi.UnsafePointRestart2:
		// Restartable instruction sequence. Back off PC to
		// the start PC.
		if startpc == 0 || startpc > pc || pc-startpc > 20 {
			throw("bad restart PC")
		}
		return true, startpc
	case abi.UnsafePointRestartAtEntry:
		// Restart from the function entry at resumption.
		return true, f.entry()
	}
	return true, pc
}
