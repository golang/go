// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

/*
Stack layout parameters.
Included both by runtime (compiled via 6c) and linkers (compiled via gcc).

The per-goroutine g->stackguard is set to point stackGuard bytes
above the bottom of the stack.  Each function compares its stack
pointer against g->stackguard to check for overflow.  To cut one
instruction from the check sequence for functions with tiny frames,
the stack is allowed to protrude stackSmall bytes below the stack
guard.  Functions with large frames don't bother with the check and
always call morestack.  The sequences are (for amd64, others are
similar):

	guard = g->stackguard
	frame = function's stack frame size
	argsize = size of function arguments (call + return)

	stack frame size <= stackSmall:
		CMPQ guard, SP
		JHI 3(PC)
		MOVQ m->morearg, $(argsize << 32)
		CALL morestack(SB)

	stack frame size > stackSmall but < stackBig
		LEAQ (frame-stackSmall)(SP), R0
		CMPQ guard, R0
		JHI 3(PC)
		MOVQ m->morearg, $(argsize << 32)
		CALL morestack(SB)

	stack frame size >= stackBig:
		MOVQ m->morearg, $((argsize << 32) | frame)
		CALL morestack(SB)

The bottom stackGuard - stackSmall bytes are important: there has
to be enough room to execute functions that refuse to check for
stack overflow, either because they need to be adjacent to the
actual caller's frame (deferproc) or because they handle the imminent
stack overflow (morestack).

For example, deferproc might call malloc, which does one of the
above checks (without allocating a full frame), which might trigger
a call to morestack.  This sequence needs to fit in the bottom
section of the stack.  On amd64, morestack's frame is 40 bytes, and
deferproc's frame is 56 bytes.  That fits well within the
stackGuard - stackSmall = 128 bytes at the bottom.
The linkers explore all possible call traces involving non-splitting
functions to make sure that this limit cannot be violated.
*/

const (
	// stackSystem is a number of additional bytes to add
	// to each stack below the usual guard area for OS-specific
	// purposes like signal handling. Used on Windows and on
	// Plan 9 because they do not use a separate stack.
	// Defined in os_*.go.

	// The amount of extra stack to allocate beyond the size
	// needed for the single frame that triggered the split.
	stackExtra = 2048

	// The minimum stack segment size to allocate.
	// If the amount needed for the splitting frame + stackExtra
	// is less than this number, the stack will have this size instead.
	stackMin           = 8192
	stackSystemRounded = stackSystem + (-stackSystem & (stackMin - 1))
	Fixedstack         = stackMin + stackSystemRounded

	// Functions that need frames bigger than this use an extra
	// instruction to do the stack split check, to avoid overflow
	// in case SP - framesize wraps below zero.
	// This value can be no bigger than the size of the unmapped
	// space at zero.
	stackBig = 4096

	// The stack guard is a pointer this many bytes above the
	// bottom of the stack.
	stackGuard = 256 + stackSystem

	// After a stack split check the SP is allowed to be this
	// many bytes below the stack guard.  This saves an instruction
	// in the checking sequence for tiny frames.
	stackSmall = 64

	// The maximum number of bytes that a chain of NOSPLIT
	// functions can use.
	stackLimit = stackGuard - stackSystem - stackSmall

	// The assumed size of the top-of-stack data block.
	// The actual size can be smaller than this but cannot be larger.
	// Checked in proc.c's runtime.malg.
	stackTop = 88

	// Goroutine preemption request.
	// Stored into g->stackguard0 to cause split stack check failure.
	// Must be greater than any real sp.
	// 0xfffffade in hex.
	stackPreempt = ^uintptr(1313)
)
