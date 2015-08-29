// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: stack barriers

package runtime

import "unsafe"

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
		lrUintptr = frame.fp - regSize
	}
	lrPtr := (*uintreg)(unsafe.Pointer(lrUintptr))
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
	*lrPtr = uintreg(stackBarrierPC)
	return true
}

// gcRemoveStackBarriers removes all stack barriers installed in gp's stack.
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
	lrPtr := (*uintreg)(unsafe.Pointer(stkbar.savedLRPtr))
	if val := *lrPtr; val != uintreg(stackBarrierPC) {
		printlock()
		print("at *", hex(stkbar.savedLRPtr), " expected stack barrier PC ", hex(stackBarrierPC), ", found ", hex(val), ", goid=", gp.goid, "\n")
		print("gp.stkbar=")
		gcPrintStkbars(gp.stkbar)
		print(", gp.stkbarPos=", gp.stkbarPos, ", gp.stack=[", hex(gp.stack.lo), ",", hex(gp.stack.hi), ")\n")
		throw("stack barrier lost")
	}
	*lrPtr = uintreg(stkbar.savedLRVal)
}

// gcPrintStkbars prints a []stkbar for debugging.
func gcPrintStkbars(stkbar []stkbar) {
	print("[")
	for i, s := range stkbar {
		if i > 0 {
			print(" ")
		}
		print("*", hex(s.savedLRPtr), "=", hex(s.savedLRVal))
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
	// On LR machines, if there is a stack barrier on the return
	// from the frame containing sp, this will mark it as hit even
	// though it isn't, but it's okay to be conservative.
	before := gp.stkbarPos
	for int(gp.stkbarPos) < len(gp.stkbar) && gp.stkbar[gp.stkbarPos].savedLRPtr < sp {
		gcRemoveStackBarrier(gp, gp.stkbar[gp.stkbarPos])
		gp.stkbarPos++
	}
	if debugStackBarrier && gp.stkbarPos != before {
		print("skip barriers below ", hex(sp), " in goid=", gp.goid, ": ")
		gcPrintStkbars(gp.stkbar[before:gp.stkbarPos])
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
	gp.stkbar[gp.stkbarPos].savedLRVal = pc
}
