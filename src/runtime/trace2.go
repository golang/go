// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.exectracer2

// Go execution tracer.
// The tracer captures a wide range of execution events like goroutine
// creation/blocking/unblocking, syscall enter/exit/block, GC-related events,
// changes of heap size, processor start/stop, etc and writes them to a buffer
// in a compact form. A precise nanosecond-precision timestamp and a stack
// trace is captured for most events.
//
// Tracer invariants (to keep the synchronization making sense):
// - An m that has a trace buffer must be on either the allm or sched.freem lists.
// - Any trace buffer mutation must either be happening in traceAdvance or between
//   a traceAcquire and a subsequent traceRelease.
// - traceAdvance cannot return until the previous generation's buffers are all flushed.
//
// See https://go.dev/issue/60773 for a link to the full design.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// Trace state.

// trace is global tracing context.
var trace struct {
	// trace.lock must only be acquired on the system stack where
	// stack splits cannot happen while it is held.
	lock mutex

	// Trace buffer management.
	//
	// First we check the empty list for any free buffers. If not, buffers
	// are allocated directly from the OS. Once they're filled up and/or
	// flushed, they end up on the full queue for trace.gen%2.
	//
	// The trace reader takes buffers off the full list one-by-one and
	// places them into reading until they're finished being read from.
	// Then they're placed onto the empty list.
	//
	// Protected by trace.lock.
	reading       *traceBuf // buffer currently handed off to user
	empty         *traceBuf // stack of empty buffers
	full          [2]traceBufQueue
	workAvailable atomic.Bool

	// State for the trace reader goroutine.
	//
	// Protected by trace.lock.
	readerGen     atomic.Uintptr // the generation the reader is currently reading for
	flushedGen    atomic.Uintptr // the last completed generation
	headerWritten bool           // whether ReadTrace has emitted trace header

	// doneSema is used to synchronize the reader and traceAdvance. Specifically,
	// it notifies traceAdvance that the reader is done with a generation.
	// Both semaphores are 0 by default (so, acquires block). traceAdvance
	// attempts to acquire for gen%2 after flushing the last buffers for gen.
	// Meanwhile the reader releases the sema for gen%2 when it has finished
	// processing gen.
	doneSema [2]uint32

	// Trace data tables for deduplicating data going into the trace.
	// There are 2 of each: one for gen%2, one for 1-gen%2.
	stackTab  [2]traceStackTable  // maps stack traces to unique ids
	stringTab [2]traceStringTable // maps strings to unique ids

	// cpuLogRead accepts CPU profile samples from the signal handler where
	// they're generated. There are two profBufs here: one for gen%2, one for
	// 1-gen%2. These profBufs use a three-word header to hold the IDs of the P, G,
	// and M (respectively) that were active at the time of the sample. Because
	// profBuf uses a record with all zeros in its header to indicate overflow,
	// we make sure to make the P field always non-zero: The ID of a real P will
	// start at bit 1, and bit 0 will be set. Samples that arrive while no P is
	// running (such as near syscalls) will set the first header field to 0b10.
	// This careful handling of the first header field allows us to store ID of
	// the active G directly in the second field, even though that will be 0
	// when sampling g0.
	//
	// Initialization and teardown of these fields is protected by traceAdvanceSema.
	cpuLogRead  [2]*profBuf
	signalLock  atomic.Uint32              // protects use of the following member, only usable in signal handlers
	cpuLogWrite [2]atomic.Pointer[profBuf] // copy of cpuLogRead for use in signal handlers, set without signalLock
	cpuSleep    *wakeableSleep
	cpuLogDone  <-chan struct{}
	cpuBuf      [2]*traceBuf

	reader atomic.Pointer[g] // goroutine that called ReadTrace, or nil

	// Fast mappings from enumerations to string IDs that are prepopulated
	// in the trace.
	markWorkerLabels [2][len(gcMarkWorkerModeStrings)]traceArg
	goStopReasons    [2][len(traceGoStopReasonStrings)]traceArg
	goBlockReasons   [2][len(traceBlockReasonStrings)]traceArg

	// Trace generation counter.
	gen            atomic.Uintptr
	lastNonZeroGen uintptr // last non-zero value of gen

	// shutdown is set when we are waiting for trace reader to finish after setting gen to 0
	//
	// Writes protected by trace.lock.
	shutdown atomic.Bool

	// Number of goroutines in syscall exiting slow path.
	exitingSyscall atomic.Int32

	// seqGC is the sequence counter for GC begin/end.
	//
	// Mutated only during stop-the-world.
	seqGC uint64
}

// Trace public API.

var (
	traceAdvanceSema  uint32 = 1
	traceShutdownSema uint32 = 1
)

// StartTrace enables tracing for the current process.
// While tracing, the data will be buffered and available via [ReadTrace].
// StartTrace returns an error if tracing is already enabled.
// Most clients should use the [runtime/trace] package or the [testing] package's
// -test.trace flag instead of calling StartTrace directly.
func StartTrace() error {
	if traceEnabled() || traceShuttingDown() {
		return errorString("tracing is already enabled")
	}
	// Block until cleanup of the last trace is done.
	semacquire(&traceShutdownSema)
	semrelease(&traceShutdownSema)

	// Hold traceAdvanceSema across trace start, since we'll want it on
	// the other side of tracing being enabled globally.
	semacquire(&traceAdvanceSema)

	// Initialize CPU profile -> trace ingestion.
	traceInitReadCPU()

	// Compute the first generation for this StartTrace.
	//
	// Note: we start from the last non-zero generation rather than 1 so we
	// can avoid resetting all the arrays indexed by gen%2 or gen%3. There's
	// more than one of each per m, p, and goroutine.
	firstGen := traceNextGen(trace.lastNonZeroGen)

	// Reset GC sequencer.
	trace.seqGC = 1

	// Reset trace reader state.
	trace.headerWritten = false
	trace.readerGen.Store(firstGen)
	trace.flushedGen.Store(0)

	// Register some basic strings in the string tables.
	traceRegisterLabelsAndReasons(firstGen)

	// Stop the world.
	//
	// The purpose of stopping the world is to make sure that no goroutine is in a
	// context where it could emit an event by bringing all goroutines to a safe point
	// with no opportunity to transition.
	//
	// The exception to this rule are goroutines that are concurrently exiting a syscall.
	// Those will all be forced into the syscalling slow path, and we'll just make sure
	// that we don't observe any goroutines in that critical section before starting
	// the world again.
	//
	// A good follow-up question to this is why stopping the world is necessary at all
	// given that we have traceAcquire and traceRelease. Unfortunately, those only help
	// us when tracing is already active (for performance, so when tracing is off the
	// tracing seqlock is left untouched). The main issue here is subtle: we're going to
	// want to obtain a correct starting status for each goroutine, but there are windows
	// of time in which we could read and emit an incorrect status. Specifically:
	//
	//	trace := traceAcquire()
	//  // <----> problem window
	//	casgstatus(gp, _Gwaiting, _Grunnable)
	//	if trace.ok() {
	//		trace.GoUnpark(gp, 2)
	//		traceRelease(trace)
	//	}
	//
	// More precisely, if we readgstatus for a gp while another goroutine is in the problem
	// window and that goroutine didn't observe that tracing had begun, then we might write
	// a GoStatus(GoWaiting) event for that goroutine, but it won't trace an event marking
	// the transition from GoWaiting to GoRunnable. The trace will then be broken, because
	// future events will be emitted assuming the tracer sees GoRunnable.
	//
	// In short, what we really need here is to make sure that the next time *any goroutine*
	// hits a traceAcquire, it sees that the trace is enabled.
	//
	// Note also that stopping the world is necessary to make sure sweep-related events are
	// coherent. Since the world is stopped and sweeps are non-preemptible, we can never start
	// the world and see an unpaired sweep 'end' event. Other parts of the tracer rely on this.
	stw := stopTheWorld(stwStartTrace)

	// Prevent sysmon from running any code that could generate events.
	lock(&sched.sysmonlock)

	// Reset mSyscallID on all Ps while we have them stationary and the trace is disabled.
	for _, pp := range allp {
		pp.trace.mSyscallID = -1
	}

	// Start tracing.
	//
	// After this executes, other Ms may start creating trace buffers and emitting
	// data into them.
	trace.gen.Store(firstGen)

	// Wait for exitingSyscall to drain.
	//
	// It may not monotonically decrease to zero, but in the limit it will always become
	// zero because the world is stopped and there are no available Ps for syscall-exited
	// goroutines to run on.
	//
	// Because we set gen before checking this, and because exitingSyscall is always incremented
	// *after* traceAcquire (which checks gen), we can be certain that when exitingSyscall is zero
	// that any goroutine that goes to exit a syscall from then on *must* observe the new gen.
	//
	// The critical section on each goroutine here is going to be quite short, so the likelihood
	// that we observe a zero value is high.
	for trace.exitingSyscall.Load() != 0 {
		osyield()
	}

	// Record some initial pieces of information.
	//
	// N.B. This will also emit a status event for this goroutine.
	tl := traceAcquire()
	tl.Gomaxprocs(gomaxprocs)  // Get this as early in the trace as possible. See comment in traceAdvance.
	tl.STWStart(stwStartTrace) // We didn't trace this above, so trace it now.

	// Record the fact that a GC is active, if applicable.
	if gcphase == _GCmark || gcphase == _GCmarktermination {
		tl.GCActive()
	}

	// Record the heap goal so we have it at the very beginning of the trace.
	tl.HeapGoal()

	// Make sure a ProcStatus is emitted for every P, while we're here.
	for _, pp := range allp {
		tl.writer().writeProcStatusForP(pp, pp == tl.mp.p.ptr()).end()
	}
	traceRelease(tl)

	unlock(&sched.sysmonlock)
	startTheWorld(stw)

	traceStartReadCPU()
	traceAdvancer.start()

	semrelease(&traceAdvanceSema)
	return nil
}

// StopTrace stops tracing, if it was previously enabled.
// StopTrace only returns after all the reads for the trace have completed.
func StopTrace() {
	traceAdvance(true)
}

// traceAdvance moves tracing to the next generation, and cleans up the current generation,
// ensuring that it's flushed out before returning. If stopTrace is true, it disables tracing
// altogether instead of advancing to the next generation.
//
// traceAdvanceSema must not be held.
func traceAdvance(stopTrace bool) {
	semacquire(&traceAdvanceSema)

	// Get the gen that we're advancing from. In this function we don't really care much
	// about the generation we're advancing _into_ since we'll do all the cleanup in this
	// generation for the next advancement.
	gen := trace.gen.Load()
	if gen == 0 {
		// We may end up here traceAdvance is called concurrently with StopTrace.
		semrelease(&traceAdvanceSema)
		return
	}

	// Write an EvFrequency event for this generation.
	//
	// N.B. This may block for quite a while to get a good frequency estimate, so make sure we do
	// this here and not e.g. on the trace reader.
	traceFrequency(gen)

	// Collect all the untraced Gs.
	type untracedG struct {
		gp           *g
		goid         uint64
		mid          int64
		status       uint32
		waitreason   waitReason
		inMarkAssist bool
	}
	var untracedGs []untracedG
	forEachGRace(func(gp *g) {
		// Make absolutely sure all Gs are ready for the next
		// generation. We need to do this even for dead Gs because
		// they may come alive with a new identity, and its status
		// traced bookkeeping might end up being stale.
		// We may miss totally new goroutines, but they'll always
		// have clean bookkeeping.
		gp.trace.readyNextGen(gen)
		// If the status was traced, nothing else to do.
		if gp.trace.statusWasTraced(gen) {
			return
		}
		// Scribble down information about this goroutine.
		ug := untracedG{gp: gp, mid: -1}
		systemstack(func() {
			me := getg().m.curg
			// We don't have to handle this G status transition because we
			// already eliminated ourselves from consideration above.
			casGToWaiting(me, _Grunning, waitReasonTraceGoroutineStatus)
			// We need to suspend and take ownership of the G to safely read its
			// goid. Note that we can't actually emit the event at this point
			// because we might stop the G in a window where it's unsafe to write
			// events based on the G's status. We need the global trace buffer flush
			// coming up to make sure we're not racing with the G.
			//
			// It should be very unlikely that we try to preempt a running G here.
			// The only situation that we might is that we're racing with a G
			// that's running for the first time in this generation. Therefore,
			// this should be relatively fast.
			s := suspendG(gp)
			if !s.dead {
				ug.goid = s.g.goid
				if s.g.m != nil {
					ug.mid = int64(s.g.m.procid)
				}
				ug.status = readgstatus(s.g) &^ _Gscan
				ug.waitreason = s.g.waitreason
				ug.inMarkAssist = s.g.inMarkAssist
			}
			resumeG(s)
			casgstatus(me, _Gwaiting, _Grunning)
		})
		if ug.goid != 0 {
			untracedGs = append(untracedGs, ug)
		}
	})

	if !stopTrace {
		// Re-register runtime goroutine labels and stop/block reasons.
		traceRegisterLabelsAndReasons(traceNextGen(gen))
	}

	// Now that we've done some of the heavy stuff, prevent the world from stopping.
	// This is necessary to ensure the consistency of the STW events. If we're feeling
	// adventurous we could lift this restriction and add a STWActive event, but the
	// cost of maintaining this consistency is low. We're not going to hold this semaphore
	// for very long and most STW periods are very short.
	// Once we hold worldsema, prevent preemption as well so we're not interrupted partway
	// through this. We want to get this done as soon as possible.
	semacquire(&worldsema)
	mp := acquirem()

	// Advance the generation or stop the trace.
	trace.lastNonZeroGen = gen
	if stopTrace {
		systemstack(func() {
			// Ordering is important here. Set shutdown first, then disable tracing,
			// so that conditions like (traceEnabled() || traceShuttingDown()) have
			// no opportunity to be false. Hold the trace lock so this update appears
			// atomic to the trace reader.
			lock(&trace.lock)
			trace.shutdown.Store(true)
			trace.gen.Store(0)
			unlock(&trace.lock)
		})
	} else {
		trace.gen.Store(traceNextGen(gen))
	}

	// Emit a ProcsChange event so we have one on record for each generation.
	// Let's emit it as soon as possible so that downstream tools can rely on the value
	// being there fairly soon in a generation.
	//
	// It's important that we do this before allowing stop-the-worlds again,
	// because the procs count could change.
	if !stopTrace {
		tl := traceAcquire()
		tl.Gomaxprocs(gomaxprocs)
		traceRelease(tl)
	}

	// Emit a GCActive event in the new generation if necessary.
	//
	// It's important that we do this before allowing stop-the-worlds again,
	// because that could emit global GC-related events.
	if !stopTrace && (gcphase == _GCmark || gcphase == _GCmarktermination) {
		tl := traceAcquire()
		tl.GCActive()
		traceRelease(tl)
	}

	// Preemption is OK again after this. If the world stops or whatever it's fine.
	// We're just cleaning up the last generation after this point.
	//
	// We also don't care if the GC starts again after this for the same reasons.
	releasem(mp)
	semrelease(&worldsema)

	// Snapshot allm and freem.
	//
	// Snapshotting after the generation counter update is sufficient.
	// Because an m must be on either allm or sched.freem if it has an active trace
	// buffer, new threads added to allm after this point must necessarily observe
	// the new generation number (sched.lock acts as a barrier).
	//
	// Threads that exit before this point and are on neither list explicitly
	// flush their own buffers in traceThreadDestroy.
	//
	// Snapshotting freem is necessary because Ms can continue to emit events
	// while they're still on that list. Removal from sched.freem is serialized with
	// this snapshot, so either we'll capture an m on sched.freem and race with
	// the removal to flush its buffers (resolved by traceThreadDestroy acquiring
	// the thread's seqlock, which one of us must win, so at least its old gen buffer
	// will be flushed in time for the new generation) or it will have flushed its
	// buffers before we snapshotted it to begin with.
	lock(&sched.lock)
	mToFlush := allm
	for mp := mToFlush; mp != nil; mp = mp.alllink {
		mp.trace.link = mp.alllink
	}
	for mp := sched.freem; mp != nil; mp = mp.freelink {
		mp.trace.link = mToFlush
		mToFlush = mp
	}
	unlock(&sched.lock)

	// Iterate over our snapshot, flushing every buffer until we're done.
	//
	// Because trace writers read the generation while the seqlock is
	// held, we can be certain that when there are no writers there are
	// also no stale generation values left. Therefore, it's safe to flush
	// any buffers that remain in that generation's slot.
	const debugDeadlock = false
	systemstack(func() {
		// Track iterations for some rudimentary deadlock detection.
		i := 0
		detectedDeadlock := false

		for mToFlush != nil {
			prev := &mToFlush
			for mp := *prev; mp != nil; {
				if mp.trace.seqlock.Load()%2 != 0 {
					// The M is writing. Come back to it later.
					prev = &mp.trace.link
					mp = mp.trace.link
					continue
				}
				// Flush the trace buffer.
				//
				// trace.lock needed for traceBufFlush, but also to synchronize
				// with traceThreadDestroy, which flushes both buffers unconditionally.
				lock(&trace.lock)
				bufp := &mp.trace.buf[gen%2]
				if *bufp != nil {
					traceBufFlush(*bufp, gen)
					*bufp = nil
				}
				unlock(&trace.lock)

				// Remove the m from the flush list.
				*prev = mp.trace.link
				mp.trace.link = nil
				mp = *prev
			}
			// Yield only if we're going to be going around the loop again.
			if mToFlush != nil {
				osyield()
			}

			if debugDeadlock {
				// Try to detect a deadlock. We probably shouldn't loop here
				// this many times.
				if i > 100000 && !detectedDeadlock {
					detectedDeadlock = true
					println("runtime: failing to flush")
					for mp := mToFlush; mp != nil; mp = mp.trace.link {
						print("runtime: m=", mp.id, "\n")
					}
				}
				i++
			}
		}
	})

	// At this point, the old generation is fully flushed minus stack and string
	// tables, CPU samples, and goroutines that haven't run at all during the last
	// generation.

	// Check to see if any Gs still haven't had events written out for them.
	statusWriter := unsafeTraceWriter(gen, nil)
	for _, ug := range untracedGs {
		if ug.gp.trace.statusWasTraced(gen) {
			// It was traced, we don't need to do anything.
			continue
		}
		// It still wasn't traced. Because we ensured all Ms stopped writing trace
		// events to the last generation, that must mean the G never had its status
		// traced in gen between when we recorded it and now. If that's true, the goid
		// and status we recorded then is exactly what we want right now.
		status := goStatusToTraceGoStatus(ug.status, ug.waitreason)
		statusWriter = statusWriter.writeGoStatus(ug.goid, ug.mid, status, ug.inMarkAssist)
	}
	statusWriter.flush().end()

	systemstack(func() {
		// Flush CPU samples, stacks, and strings for the last generation. This is safe,
		// because we're now certain no M is writing to the last generation.
		//
		// Ordering is important here. traceCPUFlush may generate new stacks and dumping
		// stacks may generate new strings.
		traceCPUFlush(gen)
		trace.stackTab[gen%2].dump(gen)
		trace.stringTab[gen%2].reset(gen)

		// That's it. This generation is done producing buffers.
		lock(&trace.lock)
		trace.flushedGen.Store(gen)
		unlock(&trace.lock)
	})

	if stopTrace {
		semacquire(&traceShutdownSema)

		// Finish off CPU profile reading.
		traceStopReadCPU()
	} else {
		// Go over each P and emit a status event for it if necessary.
		//
		// We do this at the beginning of the new generation instead of the
		// end like we do for goroutines because forEachP doesn't give us a
		// hook to skip Ps that have already been traced. Since we have to
		// preempt all Ps anyway, might as well stay consistent with StartTrace
		// which does this during the STW.
		semacquire(&worldsema)
		forEachP(waitReasonTraceProcStatus, func(pp *p) {
			tl := traceAcquire()
			if !pp.trace.statusWasTraced(tl.gen) {
				tl.writer().writeProcStatusForP(pp, false).end()
			}
			traceRelease(tl)
		})
		// Perform status reset on dead Ps because they just appear as idle.
		//
		// Holding worldsema prevents allp from changing.
		//
		// TODO(mknyszek): Consider explicitly emitting ProcCreate and ProcDestroy
		// events to indicate whether a P exists, rather than just making its
		// existence implicit.
		for _, pp := range allp[len(allp):cap(allp)] {
			pp.trace.readyNextGen(traceNextGen(gen))
		}
		semrelease(&worldsema)
	}

	// Block until the trace reader has finished processing the last generation.
	semacquire(&trace.doneSema[gen%2])
	if raceenabled {
		raceacquire(unsafe.Pointer(&trace.doneSema[gen%2]))
	}

	// Double-check that things look as we expect after advancing and perform some
	// final cleanup if the trace has fully stopped.
	systemstack(func() {
		lock(&trace.lock)
		if !trace.full[gen%2].empty() {
			throw("trace: non-empty full trace buffer for done generation")
		}
		if stopTrace {
			if !trace.full[1-(gen%2)].empty() {
				throw("trace: non-empty full trace buffer for next generation")
			}
			if trace.reading != nil || trace.reader.Load() != nil {
				throw("trace: reading after shutdown")
			}
			// Free all the empty buffers.
			for trace.empty != nil {
				buf := trace.empty
				trace.empty = buf.link
				sysFree(unsafe.Pointer(buf), unsafe.Sizeof(*buf), &memstats.other_sys)
			}
			// Clear trace.shutdown and other flags.
			trace.headerWritten = false
			trace.shutdown.Store(false)
		}
		unlock(&trace.lock)
	})

	if stopTrace {
		// Clear the sweep state on every P for the next time tracing is enabled.
		//
		// It may be stale in the next trace because we may have ended tracing in
		// the middle of a sweep on a P.
		//
		// It's fine not to call forEachP here because tracing is disabled and we
		// know at this point that nothing is calling into the tracer, but we do
		// need to look at dead Ps too just because GOMAXPROCS could have been called
		// at any point since we stopped tracing, and we have to ensure there's no
		// bad state on dead Ps too. Prevent a STW and a concurrent GOMAXPROCS that
		// might mutate allp by making ourselves briefly non-preemptible.
		mp := acquirem()
		for _, pp := range allp[:cap(allp)] {
			pp.trace.inSweep = false
			pp.trace.maySweep = false
			pp.trace.swept = 0
			pp.trace.reclaimed = 0
		}
		releasem(mp)
	}

	// Release the advance semaphore. If stopTrace is true we're still holding onto
	// traceShutdownSema.
	//
	// Do a direct handoff. Don't let one caller of traceAdvance starve
	// other calls to traceAdvance.
	semrelease1(&traceAdvanceSema, true, 0)

	if stopTrace {
		// Stop the traceAdvancer. We can't be holding traceAdvanceSema here because
		// we'll deadlock (we're blocked on the advancer goroutine exiting, but it
		// may be currently trying to acquire traceAdvanceSema).
		traceAdvancer.stop()
		semrelease(&traceShutdownSema)
	}
}

func traceNextGen(gen uintptr) uintptr {
	if gen == ^uintptr(0) {
		// gen is used both %2 and %3 and we want both patterns to continue when we loop around.
		// ^uint32(0) and ^uint64(0) are both odd and multiples of 3. Therefore the next generation
		// we want is even and one more than a multiple of 3. The smallest such number is 4.
		return 4
	}
	return gen + 1
}

// traceRegisterLabelsAndReasons re-registers mark worker labels and
// goroutine stop/block reasons in the string table for the provided
// generation. Note: the provided generation must not have started yet.
func traceRegisterLabelsAndReasons(gen uintptr) {
	for i, label := range gcMarkWorkerModeStrings[:] {
		trace.markWorkerLabels[gen%2][i] = traceArg(trace.stringTab[gen%2].put(gen, label))
	}
	for i, str := range traceBlockReasonStrings[:] {
		trace.goBlockReasons[gen%2][i] = traceArg(trace.stringTab[gen%2].put(gen, str))
	}
	for i, str := range traceGoStopReasonStrings[:] {
		trace.goStopReasons[gen%2][i] = traceArg(trace.stringTab[gen%2].put(gen, str))
	}
}

// ReadTrace returns the next chunk of binary tracing data, blocking until data
// is available. If tracing is turned off and all the data accumulated while it
// was on has been returned, ReadTrace returns nil. The caller must copy the
// returned data before calling ReadTrace again.
// ReadTrace must be called from one goroutine at a time.
func ReadTrace() []byte {
top:
	var buf []byte
	var park bool
	systemstack(func() {
		buf, park = readTrace0()
	})
	if park {
		gopark(func(gp *g, _ unsafe.Pointer) bool {
			if !trace.reader.CompareAndSwapNoWB(nil, gp) {
				// We're racing with another reader.
				// Wake up and handle this case.
				return false
			}

			if g2 := traceReader(); gp == g2 {
				// New data arrived between unlocking
				// and the CAS and we won the wake-up
				// race, so wake up directly.
				return false
			} else if g2 != nil {
				printlock()
				println("runtime: got trace reader", g2, g2.goid)
				throw("unexpected trace reader")
			}

			return true
		}, nil, waitReasonTraceReaderBlocked, traceBlockSystemGoroutine, 2)
		goto top
	}

	return buf
}

// readTrace0 is ReadTrace's continuation on g0. This must run on the
// system stack because it acquires trace.lock.
//
//go:systemstack
func readTrace0() (buf []byte, park bool) {
	if raceenabled {
		// g0 doesn't have a race context. Borrow the user G's.
		if getg().racectx != 0 {
			throw("expected racectx == 0")
		}
		getg().racectx = getg().m.curg.racectx
		// (This defer should get open-coded, which is safe on
		// the system stack.)
		defer func() { getg().racectx = 0 }()
	}

	// This function must not allocate while holding trace.lock:
	// allocation can call heap allocate, which will try to emit a trace
	// event while holding heap lock.
	lock(&trace.lock)

	if trace.reader.Load() != nil {
		// More than one goroutine reads trace. This is bad.
		// But we rather do not crash the program because of tracing,
		// because tracing can be enabled at runtime on prod servers.
		unlock(&trace.lock)
		println("runtime: ReadTrace called from multiple goroutines simultaneously")
		return nil, false
	}
	// Recycle the old buffer.
	if buf := trace.reading; buf != nil {
		buf.link = trace.empty
		trace.empty = buf
		trace.reading = nil
	}
	// Write trace header.
	if !trace.headerWritten {
		trace.headerWritten = true
		unlock(&trace.lock)
		return []byte("go 1.22 trace\x00\x00\x00"), false
	}

	// Read the next buffer.

	if trace.readerGen.Load() == 0 {
		trace.readerGen.Store(1)
	}
	var gen uintptr
	for {
		assertLockHeld(&trace.lock)
		gen = trace.readerGen.Load()

		// Check to see if we need to block for more data in this generation
		// or if we need to move our generation forward.
		if !trace.full[gen%2].empty() {
			break
		}
		// Most of the time readerGen is one generation ahead of flushedGen, as the
		// current generation is being read from. Then, once the last buffer is flushed
		// into readerGen, flushedGen will rise to meet it. At this point, the tracer
		// is waiting on the reader to finish flushing the last generation so that it
		// can continue to advance.
		if trace.flushedGen.Load() == gen {
			if trace.shutdown.Load() {
				unlock(&trace.lock)

				// Wake up anyone waiting for us to be done with this generation.
				//
				// Do this after reading trace.shutdown, because the thread we're
				// waking up is going to clear trace.shutdown.
				if raceenabled {
					// Model synchronization on trace.doneSema, which te race
					// detector does not see. This is required to avoid false
					// race reports on writer passed to trace.Start.
					racerelease(unsafe.Pointer(&trace.doneSema[gen%2]))
				}
				semrelease(&trace.doneSema[gen%2])

				// We're shutting down, and the last generation is fully
				// read. We're done.
				return nil, false
			}
			// The previous gen has had all of its buffers flushed, and
			// there's nothing else for us to read. Advance the generation
			// we're reading from and try again.
			trace.readerGen.Store(trace.gen.Load())
			unlock(&trace.lock)

			// Wake up anyone waiting for us to be done with this generation.
			//
			// Do this after reading gen to make sure we can't have the trace
			// advance until we've read it.
			if raceenabled {
				// See comment above in the shutdown case.
				racerelease(unsafe.Pointer(&trace.doneSema[gen%2]))
			}
			semrelease(&trace.doneSema[gen%2])

			// Reacquire the lock and go back to the top of the loop.
			lock(&trace.lock)
			continue
		}
		// Wait for new data.
		//
		// We don't simply use a note because the scheduler
		// executes this goroutine directly when it wakes up
		// (also a note would consume an M).
		//
		// Before we drop the lock, clear the workAvailable flag. Work can
		// only be queued with trace.lock held, so this is at least true until
		// we drop the lock.
		trace.workAvailable.Store(false)
		unlock(&trace.lock)
		return nil, true
	}
	// Pull a buffer.
	tbuf := trace.full[gen%2].pop()
	trace.reading = tbuf
	unlock(&trace.lock)
	return tbuf.arr[:tbuf.pos], false
}

// traceReader returns the trace reader that should be woken up, if any.
// Callers should first check (traceEnabled() || traceShuttingDown()).
//
// This must run on the system stack because it acquires trace.lock.
//
//go:systemstack
func traceReader() *g {
	gp := traceReaderAvailable()
	if gp == nil || !trace.reader.CompareAndSwapNoWB(gp, nil) {
		return nil
	}
	return gp
}

// traceReaderAvailable returns the trace reader if it is not currently
// scheduled and should be. Callers should first check that
// (traceEnabled() || traceShuttingDown()) is true.
func traceReaderAvailable() *g {
	// There are three conditions under which we definitely want to schedule
	// the reader:
	// - The reader is lagging behind in finishing off the last generation.
	//   In this case, trace buffers could even be empty, but the trace
	//   advancer will be waiting on the reader, so we have to make sure
	//   to schedule the reader ASAP.
	// - The reader has pending work to process for it's reader generation
	//   (assuming readerGen is not lagging behind). Note that we also want
	//   to be careful *not* to schedule the reader if there's no work to do.
	// - The trace is shutting down. The trace stopper blocks on the reader
	//   to finish, much like trace advancement.
	//
	// We also want to be careful not to schedule the reader if there's no
	// reason to.
	if trace.flushedGen.Load() == trace.readerGen.Load() || trace.workAvailable.Load() || trace.shutdown.Load() {
		return trace.reader.Load()
	}
	return nil
}

// Trace advancer goroutine.
var traceAdvancer traceAdvancerState

type traceAdvancerState struct {
	timer *wakeableSleep
	done  chan struct{}
}

// start starts a new traceAdvancer.
func (s *traceAdvancerState) start() {
	// Start a goroutine to periodically advance the trace generation.
	s.done = make(chan struct{})
	s.timer = newWakeableSleep()
	go func() {
		for traceEnabled() {
			// Set a timer to wake us up
			s.timer.sleep(int64(debug.traceadvanceperiod))

			// Try to advance the trace.
			traceAdvance(false)
		}
		s.done <- struct{}{}
	}()
}

// stop stops a traceAdvancer and blocks until it exits.
func (s *traceAdvancerState) stop() {
	s.timer.wake()
	<-s.done
	close(s.done)
	s.timer.close()
}

// traceAdvancePeriod is the approximate period between
// new generations.
const defaultTraceAdvancePeriod = 1e9 // 1 second.

// wakeableSleep manages a wakeable goroutine sleep.
//
// Users of this type must call init before first use and
// close to free up resources. Once close is called, init
// must be called before another use.
type wakeableSleep struct {
	timer *timer

	// lock protects access to wakeup, but not send/recv on it.
	lock   mutex
	wakeup chan struct{}
}

// newWakeableSleep initializes a new wakeableSleep and returns it.
func newWakeableSleep() *wakeableSleep {
	s := new(wakeableSleep)
	lockInit(&s.lock, lockRankWakeableSleep)
	s.wakeup = make(chan struct{}, 1)
	s.timer = new(timer)
	s.timer.arg = s
	s.timer.f = func(s any, _ uintptr) {
		s.(*wakeableSleep).wake()
	}
	return s
}

// sleep sleeps for the provided duration in nanoseconds or until
// another goroutine calls wake.
//
// Must not be called by more than one goroutine at a time and
// must not be called concurrently with close.
func (s *wakeableSleep) sleep(ns int64) {
	resetTimer(s.timer, nanotime()+ns)
	lock(&s.lock)
	if raceenabled {
		raceacquire(unsafe.Pointer(&s.lock))
	}
	wakeup := s.wakeup
	if raceenabled {
		racerelease(unsafe.Pointer(&s.lock))
	}
	unlock(&s.lock)
	<-wakeup
	stopTimer(s.timer)
}

// wake awakens any goroutine sleeping on the timer.
//
// Safe for concurrent use with all other methods.
func (s *wakeableSleep) wake() {
	// Grab the wakeup channel, which may be nil if we're
	// racing with close.
	lock(&s.lock)
	if raceenabled {
		raceacquire(unsafe.Pointer(&s.lock))
	}
	if s.wakeup != nil {
		// Non-blocking send.
		//
		// Others may also write to this channel and we don't
		// want to block on the receiver waking up. This also
		// effectively batches together wakeup notifications.
		select {
		case s.wakeup <- struct{}{}:
		default:
		}
	}
	if raceenabled {
		racerelease(unsafe.Pointer(&s.lock))
	}
	unlock(&s.lock)
}

// close wakes any goroutine sleeping on the timer and prevents
// further sleeping on it.
//
// Once close is called, the wakeableSleep must no longer be used.
//
// It must only be called once no goroutine is sleeping on the
// timer *and* nothing else will call wake concurrently.
func (s *wakeableSleep) close() {
	// Set wakeup to nil so that a late timer ends up being a no-op.
	lock(&s.lock)
	if raceenabled {
		raceacquire(unsafe.Pointer(&s.lock))
	}
	wakeup := s.wakeup
	s.wakeup = nil

	// Close the channel.
	close(wakeup)

	if raceenabled {
		racerelease(unsafe.Pointer(&s.lock))
	}
	unlock(&s.lock)
	return
}
