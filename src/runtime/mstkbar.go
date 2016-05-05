// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: stack barriers
//
// Stack barriers enable the garbage collector to determine how much
// of a gorountine stack has changed between when a stack is scanned
// during the concurrent scan phase and when it is re-scanned during
// the stop-the-world mark termination phase. Mark termination only
// needs to re-scan the changed part, so for deep stacks this can
// significantly reduce GC pause time compared to the alternative of
// re-scanning whole stacks. The deeper the stacks, the more stack
// barriers help.
//
// When stacks are scanned during the concurrent scan phase, the stack
// scan installs stack barriers by selecting stack frames and
// overwriting the saved return PCs (or link registers) of these
// frames with the PC of a "stack barrier trampoline". Later, when a
// selected frame returns, it "returns" to this trampoline instead of
// returning to its actual caller. The trampoline records that the
// stack has unwound past this frame and jumps to the original return
// PC recorded when the stack barrier was installed. Mark termination
// re-scans only as far as the first frame that hasn't hit a stack
// barrier and then removes and un-hit stack barriers.
//
// This scheme is very lightweight. No special code is required in the
// mutator to record stack unwinding and the trampoline is only a few
// assembly instructions.
//
// Book-keeping
// ------------
//
// The primary cost of stack barriers is book-keeping: the runtime has
// to record the locations of all stack barriers and the original
// return PCs in order to return to the correct caller when a stack
// barrier is hit and so it can remove un-hit stack barriers. In order
// to minimize this cost, the Go runtime places stack barriers in
// exponentially-spaced frames, starting 1K past the current frame.
// The book-keeping structure hence grows logarithmically with the
// size of the stack and mark termination re-scans at most twice as
// much stack as necessary.
//
// The runtime reserves space for this book-keeping structure at the
// top of the stack allocation itself (just above the outermost
// frame). This is necessary because the regular memory allocator can
// itself grow the stack, and hence can't be used when allocating
// stack-related structures.
//
// For debugging, the runtime also supports installing stack barriers
// at every frame. However, this requires significantly more
// book-keeping space.
//
// Correctness
// -----------
//
// The runtime and the compiler cooperate to ensure that all objects
// reachable from the stack as of mark termination are marked.
// Anything unchanged since the concurrent scan phase will be marked
// because it is marked by the concurrent scan. After the concurrent
// scan, there are three possible classes of stack modifications that
// must be tracked:
//
// 1) Mutator writes below the lowest un-hit stack barrier. This
// includes all writes performed by an executing function to its own
// stack frame. This part of the stack will be re-scanned by mark
// termination, which will mark any objects made reachable from
// modifications to this part of the stack.
//
// 2) Mutator writes above the lowest un-hit stack barrier. It's
// possible for a mutator to modify the stack above the lowest un-hit
// stack barrier if a higher frame has passed down a pointer to a
// stack variable in its frame. This is called an "up-pointer". The
// compiler ensures that writes through up-pointers have an
// accompanying write barrier (it simply doesn't distinguish between
// writes through up-pointers and writes through heap pointers). This
// write barrier marks any object made reachable from modifications to
// this part of the stack.
//
// 3) Runtime writes to the stack. Various runtime operations such as
// sends to unbuffered channels can write to arbitrary parts of the
// stack, including above the lowest un-hit stack barrier. We solve
// this in two ways. In many cases, the runtime can perform an
// explicit write barrier operation like in case 2. However, in the
// case of bulk memory move (typedmemmove), the runtime doesn't
// necessary have ready access to a pointer bitmap for the memory
// being copied, so it simply unwinds any stack barriers below the
// destination.
//
// Gotchas
// -------
//
// Anything that inspects or manipulates the stack potentially needs
// to understand stack barriers. The most obvious case is that
// gentraceback needs to use the original return PC when it encounters
// the stack barrier trampoline. Anything that unwinds the stack such
// as panic/recover must unwind stack barriers in tandem with
// unwinding the stack.
//
// Stack barriers require that any goroutine whose stack has been
// scanned must execute write barriers. Go solves this by simply
// enabling write barriers globally during the concurrent scan phase.
// However, traditionally, write barriers are not enabled during this
// phase.
//
// Synchronization
// ---------------
//
// For the most part, accessing and modifying stack barriers is
// synchronized around GC safe points. Installing stack barriers
// forces the G to a safe point, while all other operations that
// modify stack barriers run on the G and prevent it from reaching a
// safe point.
//
// Subtlety arises when a G may be tracebacked when *not* at a safe
// point. This happens during sigprof. For this, each G has a "stack
// barrier lock" (see gcLockStackBarriers, gcUnlockStackBarriers).
// Operations that manipulate stack barriers acquire this lock, while
// sigprof tries to acquire it and simply skips the traceback if it
// can't acquire it. There is one exception for performance and
// complexity reasons: hitting a stack barrier manipulates the stack
// barrier list without acquiring the stack barrier lock. For this,
// gentraceback performs a special fix up if the traceback starts in
// the stack barrier function.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const debugStackBarrier = false

// firstStackBarrierOffset is the approximate byte offset at
// which to place the first stack barrier from the current SP.
// This is a lower bound on how much stack will have to be
// re-scanned during mark termination. Subsequent barriers are
// placed at firstStackBarrierOffset * 2^n offsets.
//
// For debugging, this can be set to 0, which will install a
// stack barrier at every frame. If you do this, you may also
// have to raise _StackMin, since the stack barrier
// bookkeeping will use a large amount of each stack.
var firstStackBarrierOffset = 1024

// gcMaxStackBarriers returns the maximum number of stack barriers
// that can be installed in a stack of stackSize bytes.
func gcMaxStackBarriers(stackSize int) (n int) {
	if firstStackBarrierOffset == 0 {
		// Special debugging case for inserting stack barriers
		// at every frame. Steal half of the stack for the
		// []stkbar. Technically, if the stack were to consist
		// solely of return PCs we would need two thirds of
		// the stack, but stealing that much breaks things and
		// this doesn't happen in practice.
		return stackSize / 2 / int(unsafe.Sizeof(stkbar{}))
	}

	offset := firstStackBarrierOffset
	for offset < stackSize {
		n++
		offset *= 2
	}
	return n + 1
}

// gcInstallStackBarrier installs a stack barrier over the return PC of frame.
//go:nowritebarrier
func gcInstallStackBarrier(gp *g, frame *stkframe) bool {
	if frame.lr == 0 {
		if debugStackBarrier {
			print("not installing stack barrier with no LR, goid=", gp.goid, "\n")
		}
		return false
	}

	if frame.fn.entry == cgocallback_gofuncPC {
		// cgocallback_gofunc doesn't return to its LR;
		// instead, its return path puts LR in g.sched.pc and
		// switches back to the system stack on which
		// cgocallback_gofunc was originally called. We can't
		// have a stack barrier in g.sched.pc, so don't
		// install one in this frame.
		if debugStackBarrier {
			print("not installing stack barrier over LR of cgocallback_gofunc, goid=", gp.goid, "\n")
		}
		return false
	}

	// Save the return PC and overwrite it with stackBarrier.
	var lrUintptr uintptr
	if usesLR {
		lrUintptr = frame.sp
	} else {
		lrUintptr = frame.fp - sys.RegSize
	}
	lrPtr := (*sys.Uintreg)(unsafe.Pointer(lrUintptr))
	if debugStackBarrier {
		print("install stack barrier at ", hex(lrUintptr), " over ", hex(*lrPtr), ", goid=", gp.goid, "\n")
		if uintptr(*lrPtr) != frame.lr {
			print("frame.lr=", hex(frame.lr))
			throw("frame.lr differs from stack LR")
		}
	}

	gp.stkbar = gp.stkbar[:len(gp.stkbar)+1]
	stkbar := &gp.stkbar[len(gp.stkbar)-1]
	stkbar.savedLRPtr = lrUintptr
	stkbar.savedLRVal = uintptr(*lrPtr)
	*lrPtr = sys.Uintreg(stackBarrierPC)
	return true
}

// gcRemoveStackBarriers removes all stack barriers installed in gp's stack.
//
// gp's stack barriers must be locked.
//
//go:nowritebarrier
func gcRemoveStackBarriers(gp *g) {
	if debugStackBarrier && gp.stkbarPos != 0 {
		print("hit ", gp.stkbarPos, " stack barriers, goid=", gp.goid, "\n")
	}

	// Remove stack barriers that we didn't hit.
	for _, stkbar := range gp.stkbar[gp.stkbarPos:] {
		gcRemoveStackBarrier(gp, stkbar)
	}

	// Clear recorded stack barriers so copystack doesn't try to
	// adjust them.
	gp.stkbarPos = 0
	gp.stkbar = gp.stkbar[:0]
}

// gcRemoveStackBarrier removes a single stack barrier. It is the
// inverse operation of gcInstallStackBarrier.
//
// This is nosplit to ensure gp's stack does not move.
//
//go:nowritebarrier
//go:nosplit
func gcRemoveStackBarrier(gp *g, stkbar stkbar) {
	if debugStackBarrier {
		print("remove stack barrier at ", hex(stkbar.savedLRPtr), " with ", hex(stkbar.savedLRVal), ", goid=", gp.goid, "\n")
	}
	lrPtr := (*sys.Uintreg)(unsafe.Pointer(stkbar.savedLRPtr))
	if val := *lrPtr; val != sys.Uintreg(stackBarrierPC) {
		printlock()
		print("at *", hex(stkbar.savedLRPtr), " expected stack barrier PC ", hex(stackBarrierPC), ", found ", hex(val), ", goid=", gp.goid, "\n")
		print("gp.stkbar=")
		gcPrintStkbars(gp, -1)
		print(", gp.stack=[", hex(gp.stack.lo), ",", hex(gp.stack.hi), ")\n")
		throw("stack barrier lost")
	}
	*lrPtr = sys.Uintreg(stkbar.savedLRVal)
}

// gcTryRemoveAllStackBarriers tries to remove stack barriers from all
// Gs in gps. It is best-effort and efficient. If it can't remove
// barriers from a G immediately, it will simply skip it.
func gcTryRemoveAllStackBarriers(gps []*g) {
	for _, gp := range gps {
	retry:
		for {
			switch s := readgstatus(gp); s {
			default:
				break retry

			case _Grunnable, _Gsyscall, _Gwaiting:
				if !castogscanstatus(gp, s, s|_Gscan) {
					continue
				}
				gcLockStackBarriers(gp)
				gcRemoveStackBarriers(gp)
				gcUnlockStackBarriers(gp)
				restartg(gp)
				break retry
			}
		}
	}
}

// gcPrintStkbars prints the stack barriers of gp for debugging. It
// places a "@@@" marker at gp.stkbarPos. If marker >= 0, it will also
// place a "==>" marker before the marker'th entry.
func gcPrintStkbars(gp *g, marker int) {
	print("[")
	for i, s := range gp.stkbar {
		if i > 0 {
			print(" ")
		}
		if i == int(gp.stkbarPos) {
			print("@@@ ")
		}
		if i == marker {
			print("==> ")
		}
		print("*", hex(s.savedLRPtr), "=", hex(s.savedLRVal))
	}
	if int(gp.stkbarPos) == len(gp.stkbar) {
		print(" @@@")
	}
	if marker == len(gp.stkbar) {
		print(" ==>")
	}
	print("]")
}

// gcUnwindBarriers marks all stack barriers up the frame containing
// sp as hit and removes them. This is used during stack unwinding for
// panic/recover and by heapBitsBulkBarrier to force stack re-scanning
// when its destination is on the stack.
//
// This is nosplit to ensure gp's stack does not move.
//
//go:nosplit
func gcUnwindBarriers(gp *g, sp uintptr) {
	gcLockStackBarriers(gp)
	// On LR machines, if there is a stack barrier on the return
	// from the frame containing sp, this will mark it as hit even
	// though it isn't, but it's okay to be conservative.
	before := gp.stkbarPos
	for int(gp.stkbarPos) < len(gp.stkbar) && gp.stkbar[gp.stkbarPos].savedLRPtr < sp {
		gcRemoveStackBarrier(gp, gp.stkbar[gp.stkbarPos])
		gp.stkbarPos++
	}
	gcUnlockStackBarriers(gp)
	if debugStackBarrier && gp.stkbarPos != before {
		print("skip barriers below ", hex(sp), " in goid=", gp.goid, ": ")
		// We skipped barriers between the "==>" marker
		// (before) and the "@@@" marker (gp.stkbarPos).
		gcPrintStkbars(gp, int(before))
		print("\n")
	}
}

// nextBarrierPC returns the original return PC of the next stack barrier.
// Used by getcallerpc, so it must be nosplit.
//go:nosplit
func nextBarrierPC() uintptr {
	gp := getg()
	return gp.stkbar[gp.stkbarPos].savedLRVal
}

// setNextBarrierPC sets the return PC of the next stack barrier.
// Used by setcallerpc, so it must be nosplit.
//go:nosplit
func setNextBarrierPC(pc uintptr) {
	gp := getg()
	gcLockStackBarriers(gp)
	gp.stkbar[gp.stkbarPos].savedLRVal = pc
	gcUnlockStackBarriers(gp)
}

// gcLockStackBarriers synchronizes with tracebacks of gp's stack
// during sigprof for installation or removal of stack barriers. It
// blocks until any current sigprof is done tracebacking gp's stack
// and then disallows profiling tracebacks of gp's stack.
//
// This is necessary because a sigprof during barrier installation or
// removal could observe inconsistencies between the stkbar array and
// the stack itself and crash.
//
//go:nosplit
func gcLockStackBarriers(gp *g) {
	// Disable preemption so scanstack cannot run while the caller
	// is manipulating the stack barriers.
	acquirem()
	for !atomic.Cas(&gp.stackLock, 0, 1) {
		osyield()
	}
}

//go:nosplit
func gcTryLockStackBarriers(gp *g) bool {
	mp := acquirem()
	result := atomic.Cas(&gp.stackLock, 0, 1)
	if !result {
		releasem(mp)
	}
	return result
}

func gcUnlockStackBarriers(gp *g) {
	atomic.Store(&gp.stackLock, 0)
	releasem(getg().m)
}
