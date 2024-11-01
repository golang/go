// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime -> tracer API.

package runtime

import (
	"internal/runtime/atomic"
	_ "unsafe" // for go:linkname
)

// gTraceState is per-G state for the tracer.
type gTraceState struct {
	traceSchedResourceState
}

// reset resets the gTraceState for a new goroutine.
func (s *gTraceState) reset() {
	s.seq = [2]uint64{}
	// N.B. s.statusTraced is managed and cleared separately.
}

// mTraceState is per-M state for the tracer.
type mTraceState struct {
	seqlock atomic.Uintptr // seqlock indicating that this M is writing to a trace buffer.
	buf     [2]*traceBuf   // Per-M traceBuf for writing. Indexed by trace.gen%2.
	link    *m             // Snapshot of alllink or freelink.
}

// pTraceState is per-P state for the tracer.
type pTraceState struct {
	traceSchedResourceState

	// mSyscallID is the ID of the M this was bound to before entering a syscall.
	mSyscallID int64

	// maySweep indicates the sweep events should be traced.
	// This is used to defer the sweep start event until a span
	// has actually been swept.
	maySweep bool

	// inSweep indicates that at least one sweep event has been traced.
	inSweep bool

	// swept and reclaimed track the number of bytes swept and reclaimed
	// by sweeping in the current sweep loop (while maySweep was true).
	swept, reclaimed uintptr
}

// traceLockInit initializes global trace locks.
func traceLockInit() {
	// Sharing a lock rank here is fine because they should never be accessed
	// together. If they are, we want to find out immediately.
	lockInit(&trace.stringTab[0].lock, lockRankTraceStrings)
	lockInit(&trace.stringTab[0].tab.mem.lock, lockRankTraceStrings)
	lockInit(&trace.stringTab[1].lock, lockRankTraceStrings)
	lockInit(&trace.stringTab[1].tab.mem.lock, lockRankTraceStrings)
	lockInit(&trace.stackTab[0].tab.mem.lock, lockRankTraceStackTab)
	lockInit(&trace.stackTab[1].tab.mem.lock, lockRankTraceStackTab)
	lockInit(&trace.typeTab[0].tab.mem.lock, lockRankTraceTypeTab)
	lockInit(&trace.typeTab[1].tab.mem.lock, lockRankTraceTypeTab)
	lockInit(&trace.lock, lockRankTrace)
}

// lockRankMayTraceFlush records the lock ranking effects of a
// potential call to traceFlush.
//
// nosplit because traceAcquire is nosplit.
//
//go:nosplit
func lockRankMayTraceFlush() {
	lockWithRankMayAcquire(&trace.lock, getLockRank(&trace.lock))
}

// traceBlockReason is an enumeration of reasons a goroutine might block.
// This is the interface the rest of the runtime uses to tell the
// tracer why a goroutine blocked. The tracer then propagates this information
// into the trace however it sees fit.
//
// Note that traceBlockReasons should not be compared, since reasons that are
// distinct by name may *not* be distinct by value.
type traceBlockReason uint8

const (
	traceBlockGeneric traceBlockReason = iota
	traceBlockForever
	traceBlockNet
	traceBlockSelect
	traceBlockCondWait
	traceBlockSync
	traceBlockChanSend
	traceBlockChanRecv
	traceBlockGCMarkAssist
	traceBlockGCSweep
	traceBlockSystemGoroutine
	traceBlockPreempted
	traceBlockDebugCall
	traceBlockUntilGCEnds
	traceBlockSleep
	traceBlockGCWeakToStrongWait
)

var traceBlockReasonStrings = [...]string{
	traceBlockGeneric:            "unspecified",
	traceBlockForever:            "forever",
	traceBlockNet:                "network",
	traceBlockSelect:             "select",
	traceBlockCondWait:           "sync.(*Cond).Wait",
	traceBlockSync:               "sync",
	traceBlockChanSend:           "chan send",
	traceBlockChanRecv:           "chan receive",
	traceBlockGCMarkAssist:       "GC mark assist wait for work",
	traceBlockGCSweep:            "GC background sweeper wait",
	traceBlockSystemGoroutine:    "system goroutine wait",
	traceBlockPreempted:          "preempted",
	traceBlockDebugCall:          "wait for debug call",
	traceBlockUntilGCEnds:        "wait until GC ends",
	traceBlockSleep:              "sleep",
	traceBlockGCWeakToStrongWait: "GC weak to strong wait",
}

// traceGoStopReason is an enumeration of reasons a goroutine might yield.
//
// Note that traceGoStopReasons should not be compared, since reasons that are
// distinct by name may *not* be distinct by value.
type traceGoStopReason uint8

const (
	traceGoStopGeneric traceGoStopReason = iota
	traceGoStopGoSched
	traceGoStopPreempted
)

var traceGoStopReasonStrings = [...]string{
	traceGoStopGeneric:   "unspecified",
	traceGoStopGoSched:   "runtime.Gosched",
	traceGoStopPreempted: "preempted",
}

// traceEnabled returns true if the trace is currently enabled.
//
//go:nosplit
func traceEnabled() bool {
	return trace.enabled
}

// traceAllocFreeEnabled returns true if the trace is currently enabled
// and alloc/free events are also enabled.
//
//go:nosplit
func traceAllocFreeEnabled() bool {
	return trace.enabledWithAllocFree
}

// traceShuttingDown returns true if the trace is currently shutting down.
func traceShuttingDown() bool {
	return trace.shutdown.Load()
}

// traceLocker represents an M writing trace events. While a traceLocker value
// is valid, the tracer observes all operations on the G/M/P or trace events being
// written as happening atomically.
type traceLocker struct {
	mp  *m
	gen uintptr
}

// debugTraceReentrancy checks if the trace is reentrant.
//
// This is optional because throwing in a function makes it instantly
// not inlineable, and we want traceAcquire to be inlineable for
// low overhead when the trace is disabled.
const debugTraceReentrancy = false

// traceAcquire prepares this M for writing one or more trace events.
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func traceAcquire() traceLocker {
	if !traceEnabled() {
		return traceLocker{}
	}
	return traceAcquireEnabled()
}

// traceTryAcquire is like traceAcquire, but may return an invalid traceLocker even
// if tracing is enabled. For example, it will return !ok if traceAcquire is being
// called with an active traceAcquire on the M (reentrant locking). This exists for
// optimistically emitting events in the few contexts where tracing is now allowed.
//
// nosplit for alignment with traceTryAcquire, so it can be used in the
// same contexts.
//
//go:nosplit
func traceTryAcquire() traceLocker {
	if !traceEnabled() {
		return traceLocker{}
	}
	return traceTryAcquireEnabled()
}

// traceAcquireEnabled is the traceEnabled path for traceAcquire. It's explicitly
// broken out to make traceAcquire inlineable to keep the overhead of the tracer
// when it's disabled low.
//
// nosplit because it's called by traceAcquire, which is nosplit.
//
//go:nosplit
func traceAcquireEnabled() traceLocker {
	// Any time we acquire a traceLocker, we may flush a trace buffer. But
	// buffer flushes are rare. Record the lock edge even if it doesn't happen
	// this time.
	lockRankMayTraceFlush()

	// Prevent preemption.
	mp := acquirem()

	// Acquire the trace seqlock. This prevents traceAdvance from moving forward
	// until all Ms are observed to be outside of their seqlock critical section.
	//
	// Note: The seqlock is mutated here and also in traceCPUSample. If you update
	// usage of the seqlock here, make sure to also look at what traceCPUSample is
	// doing.
	seq := mp.trace.seqlock.Add(1)
	if debugTraceReentrancy && seq%2 != 1 {
		throw("bad use of trace.seqlock or tracer is reentrant")
	}

	// N.B. This load of gen appears redundant with the one in traceEnabled.
	// However, it's very important that the gen we use for writing to the trace
	// is acquired under a traceLocker so traceAdvance can make sure no stale
	// gen values are being used.
	//
	// Because we're doing this load again, it also means that the trace
	// might end up being disabled when we load it. In that case we need to undo
	// what we did and bail.
	gen := trace.gen.Load()
	if gen == 0 {
		mp.trace.seqlock.Add(1)
		releasem(mp)
		return traceLocker{}
	}
	return traceLocker{mp, gen}
}

// traceTryAcquireEnabled is like traceAcquireEnabled but may return an invalid
// traceLocker under some conditions. See traceTryAcquire for more details.
//
// nosplit for alignment with traceAcquireEnabled, so it can be used in the
// same contexts.
//
//go:nosplit
func traceTryAcquireEnabled() traceLocker {
	// Any time we acquire a traceLocker, we may flush a trace buffer. But
	// buffer flushes are rare. Record the lock edge even if it doesn't happen
	// this time.
	lockRankMayTraceFlush()

	// Check if we're already locked. If so, return an invalid traceLocker.
	if getg().m.trace.seqlock.Load()%2 == 1 {
		return traceLocker{}
	}
	return traceAcquireEnabled()
}

// ok returns true if the traceLocker is valid (i.e. tracing is enabled).
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func (tl traceLocker) ok() bool {
	return tl.gen != 0
}

// traceRelease indicates that this M is done writing trace events.
//
// nosplit because it's called on the syscall path when stack movement is forbidden.
//
//go:nosplit
func traceRelease(tl traceLocker) {
	seq := tl.mp.trace.seqlock.Add(1)
	if debugTraceReentrancy && seq%2 != 0 {
		print("runtime: seq=", seq, "\n")
		throw("bad use of trace.seqlock")
	}
	releasem(tl.mp)
}

// traceExitingSyscall marks a goroutine as exiting the syscall slow path.
//
// Must be paired with a traceExitedSyscall call.
func traceExitingSyscall() {
	trace.exitingSyscall.Add(1)
}

// traceExitedSyscall marks a goroutine as having exited the syscall slow path.
func traceExitedSyscall() {
	trace.exitingSyscall.Add(-1)
}

// Gomaxprocs emits a ProcsChange event.
func (tl traceLocker) Gomaxprocs(procs int32) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvProcsChange, traceArg(procs), tl.stack(1))
}

// ProcStart traces a ProcStart event.
//
// Must be called with a valid P.
func (tl traceLocker) ProcStart() {
	pp := tl.mp.p.ptr()
	// Procs are typically started within the scheduler when there is no user goroutine. If there is a user goroutine,
	// it must be in _Gsyscall because the only time a goroutine is allowed to have its Proc moved around from under it
	// is during a syscall.
	tl.eventWriter(traceGoSyscall, traceProcIdle).commit(traceEvProcStart, traceArg(pp.id), pp.trace.nextSeq(tl.gen))
}

// ProcStop traces a ProcStop event.
func (tl traceLocker) ProcStop(pp *p) {
	// The only time a goroutine is allowed to have its Proc moved around
	// from under it is during a syscall.
	tl.eventWriter(traceGoSyscall, traceProcRunning).commit(traceEvProcStop)
}

// GCActive traces a GCActive event.
//
// Must be emitted by an actively running goroutine on an active P. This restriction can be changed
// easily and only depends on where it's currently called.
func (tl traceLocker) GCActive() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCActive, traceArg(trace.seqGC))
	// N.B. Only one GC can be running at a time, so this is naturally
	// serialized by the caller.
	trace.seqGC++
}

// GCStart traces a GCBegin event.
//
// Must be emitted by an actively running goroutine on an active P. This restriction can be changed
// easily and only depends on where it's currently called.
func (tl traceLocker) GCStart() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCBegin, traceArg(trace.seqGC), tl.stack(3))
	// N.B. Only one GC can be running at a time, so this is naturally
	// serialized by the caller.
	trace.seqGC++
}

// GCDone traces a GCEnd event.
//
// Must be emitted by an actively running goroutine on an active P. This restriction can be changed
// easily and only depends on where it's currently called.
func (tl traceLocker) GCDone() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCEnd, traceArg(trace.seqGC))
	// N.B. Only one GC can be running at a time, so this is naturally
	// serialized by the caller.
	trace.seqGC++
}

// STWStart traces a STWBegin event.
func (tl traceLocker) STWStart(reason stwReason) {
	// Although the current P may be in _Pgcstop here, we model the P as running during the STW. This deviates from the
	// runtime's state tracking, but it's more accurate and doesn't result in any loss of information.
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvSTWBegin, tl.string(reason.String()), tl.stack(2))
}

// STWDone traces a STWEnd event.
func (tl traceLocker) STWDone() {
	// Although the current P may be in _Pgcstop here, we model the P as running during the STW. This deviates from the
	// runtime's state tracking, but it's more accurate and doesn't result in any loss of information.
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvSTWEnd)
}

// GCSweepStart prepares to trace a sweep loop. This does not
// emit any events until traceGCSweepSpan is called.
//
// GCSweepStart must be paired with traceGCSweepDone and there
// must be no preemption points between these two calls.
//
// Must be called with a valid P.
func (tl traceLocker) GCSweepStart() {
	// Delay the actual GCSweepBegin event until the first span
	// sweep. If we don't sweep anything, don't emit any events.
	pp := tl.mp.p.ptr()
	if pp.trace.maySweep {
		throw("double traceGCSweepStart")
	}
	pp.trace.maySweep, pp.trace.swept, pp.trace.reclaimed = true, 0, 0
}

// GCSweepSpan traces the sweep of a single span. If this is
// the first span swept since traceGCSweepStart was called, this
// will emit a GCSweepBegin event.
//
// This may be called outside a traceGCSweepStart/traceGCSweepDone
// pair; however, it will not emit any trace events in this case.
//
// Must be called with a valid P.
func (tl traceLocker) GCSweepSpan(bytesSwept uintptr) {
	pp := tl.mp.p.ptr()
	if pp.trace.maySweep {
		if pp.trace.swept == 0 {
			tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCSweepBegin, tl.stack(1))
			pp.trace.inSweep = true
		}
		pp.trace.swept += bytesSwept
	}
}

// GCSweepDone finishes tracing a sweep loop. If any memory was
// swept (i.e. traceGCSweepSpan emitted an event) then this will emit
// a GCSweepEnd event.
//
// Must be called with a valid P.
func (tl traceLocker) GCSweepDone() {
	pp := tl.mp.p.ptr()
	if !pp.trace.maySweep {
		throw("missing traceGCSweepStart")
	}
	if pp.trace.inSweep {
		tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCSweepEnd, traceArg(pp.trace.swept), traceArg(pp.trace.reclaimed))
		pp.trace.inSweep = false
	}
	pp.trace.maySweep = false
}

// GCMarkAssistStart emits a MarkAssistBegin event.
func (tl traceLocker) GCMarkAssistStart() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCMarkAssistBegin, tl.stack(1))
}

// GCMarkAssistDone emits a MarkAssistEnd event.
func (tl traceLocker) GCMarkAssistDone() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGCMarkAssistEnd)
}

// GoCreate emits a GoCreate event.
func (tl traceLocker) GoCreate(newg *g, pc uintptr, blocked bool) {
	newg.trace.setStatusTraced(tl.gen)
	ev := traceEvGoCreate
	if blocked {
		ev = traceEvGoCreateBlocked
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(ev, traceArg(newg.goid), tl.startPC(pc), tl.stack(2))
}

// GoStart emits a GoStart event.
//
// Must be called with a valid P.
func (tl traceLocker) GoStart() {
	gp := getg().m.curg
	pp := gp.m.p
	w := tl.eventWriter(traceGoRunnable, traceProcRunning)
	w = w.write(traceEvGoStart, traceArg(gp.goid), gp.trace.nextSeq(tl.gen))
	if pp.ptr().gcMarkWorkerMode != gcMarkWorkerNotWorker {
		w = w.write(traceEvGoLabel, trace.markWorkerLabels[tl.gen%2][pp.ptr().gcMarkWorkerMode])
	}
	w.end()
}

// GoEnd emits a GoDestroy event.
//
// TODO(mknyszek): Rename this to GoDestroy.
func (tl traceLocker) GoEnd() {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoDestroy)
}

// GoSched emits a GoStop event with a GoSched reason.
func (tl traceLocker) GoSched() {
	tl.GoStop(traceGoStopGoSched)
}

// GoPreempt emits a GoStop event with a GoPreempted reason.
func (tl traceLocker) GoPreempt() {
	tl.GoStop(traceGoStopPreempted)
}

// GoStop emits a GoStop event with the provided reason.
func (tl traceLocker) GoStop(reason traceGoStopReason) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoStop, traceArg(trace.goStopReasons[tl.gen%2][reason]), tl.stack(1))
}

// GoPark emits a GoBlock event with the provided reason.
//
// TODO(mknyszek): Replace traceBlockReason with waitReason. It's silly
// that we have both, and waitReason is way more descriptive.
func (tl traceLocker) GoPark(reason traceBlockReason, skip int) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoBlock, traceArg(trace.goBlockReasons[tl.gen%2][reason]), tl.stack(skip))
}

// GoUnpark emits a GoUnblock event.
func (tl traceLocker) GoUnpark(gp *g, skip int) {
	// Emit a GoWaiting status if necessary for the unblocked goroutine.
	w := tl.eventWriter(traceGoRunning, traceProcRunning)
	// Careful: don't use the event writer. We never want status or in-progress events
	// to trigger more in-progress events.
	w.w = emitUnblockStatus(w.w, gp, tl.gen)
	w.commit(traceEvGoUnblock, traceArg(gp.goid), gp.trace.nextSeq(tl.gen), tl.stack(skip))
}

// GoCoroswitch emits a GoSwitch event. If destroy is true, the calling goroutine
// is simultaneously being destroyed.
func (tl traceLocker) GoSwitch(nextg *g, destroy bool) {
	// Emit a GoWaiting status if necessary for the unblocked goroutine.
	w := tl.eventWriter(traceGoRunning, traceProcRunning)
	// Careful: don't use the event writer. We never want status or in-progress events
	// to trigger more in-progress events.
	w.w = emitUnblockStatus(w.w, nextg, tl.gen)
	ev := traceEvGoSwitch
	if destroy {
		ev = traceEvGoSwitchDestroy
	}
	w.commit(ev, traceArg(nextg.goid), nextg.trace.nextSeq(tl.gen))
}

// emitUnblockStatus emits a GoStatus GoWaiting event for a goroutine about to be
// unblocked to the trace writer.
func emitUnblockStatus(w traceWriter, gp *g, gen uintptr) traceWriter {
	if !gp.trace.statusWasTraced(gen) && gp.trace.acquireStatus(gen) {
		// TODO(go.dev/issue/65634): Although it would be nice to add a stack trace here of gp,
		// we cannot safely do so. gp is in _Gwaiting and so we don't have ownership of its stack.
		// We can fix this by acquiring the goroutine's scan bit.
		w = w.writeGoStatus(gp.goid, -1, traceGoWaiting, gp.inMarkAssist, 0)
	}
	return w
}

// GoSysCall emits a GoSyscallBegin event.
//
// Must be called with a valid P.
func (tl traceLocker) GoSysCall() {
	// Scribble down the M that the P is currently attached to.
	pp := tl.mp.p.ptr()
	pp.trace.mSyscallID = int64(tl.mp.procid)
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvGoSyscallBegin, pp.trace.nextSeq(tl.gen), tl.stack(1))
}

// GoSysExit emits a GoSyscallEnd event, possibly along with a GoSyscallBlocked event
// if lostP is true.
//
// lostP must be true in all cases that a goroutine loses its P during a syscall.
// This means it's not sufficient to check if it has no P. In particular, it needs to be
// true in the following cases:
// - The goroutine lost its P, it ran some other code, and then got it back. It's now running with that P.
// - The goroutine lost its P and was unable to reacquire it, and is now running without a P.
// - The goroutine lost its P and acquired a different one, and is now running with that P.
func (tl traceLocker) GoSysExit(lostP bool) {
	ev := traceEvGoSyscallEnd
	procStatus := traceProcSyscall // Procs implicitly enter traceProcSyscall on GoSyscallBegin.
	if lostP {
		ev = traceEvGoSyscallEndBlocked
		procStatus = traceProcRunning // If a G has a P when emitting this event, it reacquired a P and is indeed running.
	} else {
		tl.mp.p.ptr().trace.mSyscallID = -1
	}
	tl.eventWriter(traceGoSyscall, procStatus).commit(ev)
}

// ProcSteal indicates that our current M stole a P from another M.
//
// inSyscall indicates that we're stealing the P from a syscall context.
//
// The caller must have ownership of pp.
func (tl traceLocker) ProcSteal(pp *p, inSyscall bool) {
	// Grab the M ID we stole from.
	mStolenFrom := pp.trace.mSyscallID
	pp.trace.mSyscallID = -1

	// The status of the proc and goroutine, if we need to emit one here, is not evident from the
	// context of just emitting this event alone. There are two cases. Either we're trying to steal
	// the P just to get its attention (e.g. STW or sysmon retake) or we're trying to steal a P for
	// ourselves specifically to keep running. The two contexts look different, but can be summarized
	// fairly succinctly. In the former, we're a regular running goroutine and proc, if we have either.
	// In the latter, we're a goroutine in a syscall.
	goStatus := traceGoRunning
	procStatus := traceProcRunning
	if inSyscall {
		goStatus = traceGoSyscall
		procStatus = traceProcSyscallAbandoned
	}
	w := tl.eventWriter(goStatus, procStatus)

	// Emit the status of the P we're stealing. We may have *just* done this when creating the event
	// writer but it's not guaranteed, even if inSyscall is true. Although it might seem like from a
	// syscall context we're always stealing a P for ourselves, we may have not wired it up yet (so
	// it wouldn't be visible to eventWriter) or we may not even intend to wire it up to ourselves
	// at all (e.g. entersyscall_gcwait).
	if !pp.trace.statusWasTraced(tl.gen) && pp.trace.acquireStatus(tl.gen) {
		// Careful: don't use the event writer. We never want status or in-progress events
		// to trigger more in-progress events.
		w.w = w.w.writeProcStatus(uint64(pp.id), traceProcSyscallAbandoned, pp.trace.inSweep)
	}
	w.commit(traceEvProcSteal, traceArg(pp.id), pp.trace.nextSeq(tl.gen), traceArg(mStolenFrom))
}

// HeapAlloc emits a HeapAlloc event.
func (tl traceLocker) HeapAlloc(live uint64) {
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvHeapAlloc, traceArg(live))
}

// HeapGoal reads the current heap goal and emits a HeapGoal event.
func (tl traceLocker) HeapGoal() {
	heapGoal := gcController.heapGoal()
	if heapGoal == ^uint64(0) {
		// Heap-based triggering is disabled.
		heapGoal = 0
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvHeapGoal, traceArg(heapGoal))
}

// GoCreateSyscall indicates that a goroutine has transitioned from dead to GoSyscall.
//
// Unlike GoCreate, the caller must be running on gp.
//
// This occurs when C code calls into Go. On pthread platforms it occurs only when
// a C thread calls into Go code for the first time.
func (tl traceLocker) GoCreateSyscall(gp *g) {
	// N.B. We should never trace a status for this goroutine (which we're currently running on),
	// since we want this to appear like goroutine creation.
	gp.trace.setStatusTraced(tl.gen)
	tl.eventWriter(traceGoBad, traceProcBad).commit(traceEvGoCreateSyscall, traceArg(gp.goid))
}

// GoDestroySyscall indicates that a goroutine has transitioned from GoSyscall to dead.
//
// Must not have a P.
//
// This occurs when Go code returns back to C. On pthread platforms it occurs only when
// the C thread is destroyed.
func (tl traceLocker) GoDestroySyscall() {
	// N.B. If we trace a status here, we must never have a P, and we must be on a goroutine
	// that is in the syscall state.
	tl.eventWriter(traceGoSyscall, traceProcBad).commit(traceEvGoDestroySyscall)
}

// To access runtime functions from runtime/trace.
// See runtime/trace/annotation.go

// trace_userTaskCreate emits a UserTaskCreate event.
//
//go:linkname trace_userTaskCreate runtime/trace.userTaskCreate
func trace_userTaskCreate(id, parentID uint64, taskType string) {
	tl := traceAcquire()
	if !tl.ok() {
		// Need to do this check because the caller won't have it.
		return
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvUserTaskBegin, traceArg(id), traceArg(parentID), tl.string(taskType), tl.stack(3))
	traceRelease(tl)
}

// trace_userTaskEnd emits a UserTaskEnd event.
//
//go:linkname trace_userTaskEnd runtime/trace.userTaskEnd
func trace_userTaskEnd(id uint64) {
	tl := traceAcquire()
	if !tl.ok() {
		// Need to do this check because the caller won't have it.
		return
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvUserTaskEnd, traceArg(id), tl.stack(2))
	traceRelease(tl)
}

// trace_userTaskEnd emits a UserRegionBegin or UserRegionEnd event,
// depending on mode (0 == Begin, 1 == End).
//
// TODO(mknyszek): Just make this two functions.
//
//go:linkname trace_userRegion runtime/trace.userRegion
func trace_userRegion(id, mode uint64, name string) {
	tl := traceAcquire()
	if !tl.ok() {
		// Need to do this check because the caller won't have it.
		return
	}
	var ev traceEv
	switch mode {
	case 0:
		ev = traceEvUserRegionBegin
	case 1:
		ev = traceEvUserRegionEnd
	default:
		return
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(ev, traceArg(id), tl.string(name), tl.stack(3))
	traceRelease(tl)
}

// trace_userTaskEnd emits a UserRegionBegin or UserRegionEnd event.
//
//go:linkname trace_userLog runtime/trace.userLog
func trace_userLog(id uint64, category, message string) {
	tl := traceAcquire()
	if !tl.ok() {
		// Need to do this check because the caller won't have it.
		return
	}
	tl.eventWriter(traceGoRunning, traceProcRunning).commit(traceEvUserLog, traceArg(id), tl.string(category), tl.uniqueString(message), tl.stack(3))
	traceRelease(tl)
}

// traceThreadDestroy is called when a thread is removed from
// sched.freem.
//
// mp must not be able to emit trace events anymore.
//
// sched.lock must be held to synchronize with traceAdvance.
func traceThreadDestroy(mp *m) {
	assertLockHeld(&sched.lock)

	// Flush all outstanding buffers to maintain the invariant
	// that an M only has active buffers while on sched.freem
	// or allm.
	//
	// Perform a traceAcquire/traceRelease on behalf of mp to
	// synchronize with the tracer trying to flush our buffer
	// as well.
	seq := mp.trace.seqlock.Add(1)
	if debugTraceReentrancy && seq%2 != 1 {
		throw("bad use of trace.seqlock or tracer is reentrant")
	}
	systemstack(func() {
		lock(&trace.lock)
		for i := range mp.trace.buf {
			if mp.trace.buf[i] != nil {
				// N.B. traceBufFlush accepts a generation, but it
				// really just cares about gen%2.
				traceBufFlush(mp.trace.buf[i], uintptr(i))
				mp.trace.buf[i] = nil
			}
		}
		unlock(&trace.lock)
	})
	seq1 := mp.trace.seqlock.Add(1)
	if seq1 != seq+1 {
		print("runtime: seq1=", seq1, "\n")
		throw("bad use of trace.seqlock")
	}
}
