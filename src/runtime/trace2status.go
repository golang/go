// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.exectracer2

// Trace goroutine and P status management.

package runtime

import "runtime/internal/atomic"

// traceGoStatus is the status of a goroutine.
//
// They correspond directly to the various goroutine
// statuses.
type traceGoStatus uint8

const (
	traceGoBad traceGoStatus = iota
	traceGoRunnable
	traceGoRunning
	traceGoSyscall
	traceGoWaiting
)

// traceProcStatus is the status of a P.
//
// They mostly correspond to the various P statuses.
type traceProcStatus uint8

const (
	traceProcBad traceProcStatus = iota
	traceProcRunning
	traceProcIdle
	traceProcSyscall

	// traceProcSyscallAbandoned is a special case of
	// traceProcSyscall. It's used in the very specific case
	// where the first a P is mentioned in a generation is
	// part of a ProcSteal event. If that's the first time
	// it's mentioned, then there's no GoSyscallBegin to
	// connect the P stealing back to at that point. This
	// special state indicates this to the parser, so it
	// doesn't try to find a GoSyscallEndBlocked that
	// corresponds with the ProcSteal.
	traceProcSyscallAbandoned
)

// writeGoStatus emits a GoStatus event as well as any active ranges on the goroutine.
func (w traceWriter) writeGoStatus(goid uint64, mid int64, status traceGoStatus, markAssist bool) traceWriter {
	// The status should never be bad. Some invariant must have been violated.
	if status == traceGoBad {
		print("runtime: goid=", goid, "\n")
		throw("attempted to trace a bad status for a goroutine")
	}

	// Trace the status.
	w = w.event(traceEvGoStatus, traceArg(goid), traceArg(uint64(mid)), traceArg(status))

	// Trace any special ranges that are in-progress.
	if markAssist {
		w = w.event(traceEvGCMarkAssistActive, traceArg(goid))
	}
	return w
}

// writeProcStatusForP emits a ProcStatus event for the provided p based on its status.
//
// The caller must fully own pp and it must be prevented from transitioning (e.g. this can be
// called by a forEachP callback or from a STW).
func (w traceWriter) writeProcStatusForP(pp *p, inSTW bool) traceWriter {
	if !pp.trace.acquireStatus(w.gen) {
		return w
	}
	var status traceProcStatus
	switch pp.status {
	case _Pidle, _Pgcstop:
		status = traceProcIdle
		if pp.status == _Pgcstop && inSTW {
			// N.B. a P that is running and currently has the world stopped will be
			// in _Pgcstop, but we model it as running in the tracer.
			status = traceProcRunning
		}
	case _Prunning:
		status = traceProcRunning
		// There's a short window wherein the goroutine may have entered _Gsyscall
		// but it still owns the P (it's not in _Psyscall yet). The goroutine entering
		// _Gsyscall is the tracer's signal that the P its bound to is also in a syscall,
		// so we need to emit a status that matches. See #64318.
		if w.mp.p.ptr() == pp && w.mp.curg != nil && readgstatus(w.mp.curg)&^_Gscan == _Gsyscall {
			status = traceProcSyscall
		}
	case _Psyscall:
		status = traceProcSyscall
	default:
		throw("attempt to trace invalid or unsupported P status")
	}
	w = w.writeProcStatus(uint64(pp.id), status, pp.trace.inSweep)
	return w
}

// writeProcStatus emits a ProcStatus event with all the provided information.
//
// The caller must have taken ownership of a P's status writing, and the P must be
// prevented from transitioning.
func (w traceWriter) writeProcStatus(pid uint64, status traceProcStatus, inSweep bool) traceWriter {
	// The status should never be bad. Some invariant must have been violated.
	if status == traceProcBad {
		print("runtime: pid=", pid, "\n")
		throw("attempted to trace a bad status for a proc")
	}

	// Trace the status.
	w = w.event(traceEvProcStatus, traceArg(pid), traceArg(status))

	// Trace any special ranges that are in-progress.
	if inSweep {
		w = w.event(traceEvGCSweepActive, traceArg(pid))
	}
	return w
}

// goStatusToTraceGoStatus translates the internal status to tracGoStatus.
//
// status must not be _Gdead or any status whose name has the suffix "_unused."
func goStatusToTraceGoStatus(status uint32, wr waitReason) traceGoStatus {
	// N.B. Ignore the _Gscan bit. We don't model it in the tracer.
	var tgs traceGoStatus
	switch status &^ _Gscan {
	case _Grunnable:
		tgs = traceGoRunnable
	case _Grunning, _Gcopystack:
		tgs = traceGoRunning
	case _Gsyscall:
		tgs = traceGoSyscall
	case _Gwaiting, _Gpreempted:
		// There are a number of cases where a G might end up in
		// _Gwaiting but it's actually running in a non-preemptive
		// state but needs to present itself as preempted to the
		// garbage collector. In these cases, we're not going to
		// emit an event, and we want these goroutines to appear in
		// the final trace as if they're running, not blocked.
		tgs = traceGoWaiting
		if status == _Gwaiting &&
			wr == waitReasonStoppingTheWorld ||
			wr == waitReasonGCMarkTermination ||
			wr == waitReasonGarbageCollection ||
			wr == waitReasonTraceProcStatus ||
			wr == waitReasonPageTraceFlush ||
			wr == waitReasonGCWorkerActive {
			tgs = traceGoRunning
		}
	case _Gdead:
		throw("tried to trace dead goroutine")
	default:
		throw("tried to trace goroutine with invalid or unsupported status")
	}
	return tgs
}

// traceSchedResourceState is shared state for scheduling resources (i.e. fields common to
// both Gs and Ps).
type traceSchedResourceState struct {
	// statusTraced indicates whether a status event was traced for this resource
	// a particular generation.
	//
	// There are 3 of these because when transitioning across generations, traceAdvance
	// needs to be able to reliably observe whether a status was traced for the previous
	// generation, while we need to clear the value for the next generation.
	statusTraced [3]atomic.Uint32

	// seq is the sequence counter for this scheduling resource's events.
	// The purpose of the sequence counter is to establish a partial order between
	// events that don't obviously happen serially (same M) in the stream ofevents.
	//
	// There are two of these so that we can reset the counter on each generation.
	// This saves space in the resulting trace by keeping the counter small and allows
	// GoStatus and GoCreate events to omit a sequence number (implicitly 0).
	seq [2]uint64
}

// acquireStatus acquires the right to emit a Status event for the scheduling resource.
func (r *traceSchedResourceState) acquireStatus(gen uintptr) bool {
	if !r.statusTraced[gen%3].CompareAndSwap(0, 1) {
		return false
	}
	r.readyNextGen(gen)
	return true
}

// readyNextGen readies r for the generation following gen.
func (r *traceSchedResourceState) readyNextGen(gen uintptr) {
	nextGen := traceNextGen(gen)
	r.seq[nextGen%2] = 0
	r.statusTraced[nextGen%3].Store(0)
}

// statusWasTraced returns true if the sched resource's status was already acquired for tracing.
func (r *traceSchedResourceState) statusWasTraced(gen uintptr) bool {
	return r.statusTraced[gen%3].Load() != 0
}

// setStatusTraced indicates that the resource's status was already traced, for example
// when a goroutine is created.
func (r *traceSchedResourceState) setStatusTraced(gen uintptr) {
	r.statusTraced[gen%3].Store(1)
}

// nextSeq returns the next sequence number for the resource.
func (r *traceSchedResourceState) nextSeq(gen uintptr) traceArg {
	r.seq[gen%2]++
	return traceArg(r.seq[gen%2])
}
