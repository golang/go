// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace event writing API for trace2runtime.go.

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
)

// Event types in the trace, args are given in square brackets.
//
// Naming scheme:
//   - Time range event pairs have suffixes "Begin" and "End".
//   - "Start", "Stop", "Create", "Destroy", "Block", "Unblock"
//     are suffixes reserved for scheduling resources.
//
// NOTE: If you add an event type, make sure you also update all
// tables in this file!
type traceEv uint8

const (
	traceEvNone traceEv = iota // unused

	// Structural events.
	traceEvEventBatch // start of per-M batch of events [generation, M ID, timestamp, batch length]
	traceEvStacks     // start of a section of the stack table [...traceEvStack]
	traceEvStack      // stack table entry [ID, ...{PC, func string ID, file string ID, line #}]
	traceEvStrings    // start of a section of the string dictionary [...traceEvString]
	traceEvString     // string dictionary entry [ID, length, string]
	traceEvCPUSamples // start of a section of CPU samples [...traceEvCPUSample]
	traceEvCPUSample  // CPU profiling sample [timestamp, M ID, P ID, goroutine ID, stack ID]
	traceEvFrequency  // timestamp units per sec [freq]

	// Procs.
	traceEvProcsChange // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack ID]
	traceEvProcStart   // start of P [timestamp, P ID, P seq]
	traceEvProcStop    // stop of P [timestamp]
	traceEvProcSteal   // P was stolen [timestamp, P ID, P seq, M ID]
	traceEvProcStatus  // P status at the start of a generation [timestamp, P ID, status]

	// Goroutines.
	traceEvGoCreate            // goroutine creation [timestamp, new goroutine ID, new stack ID, stack ID]
	traceEvGoCreateSyscall     // goroutine appears in syscall (cgo callback) [timestamp, new goroutine ID]
	traceEvGoStart             // goroutine starts running [timestamp, goroutine ID, goroutine seq]
	traceEvGoDestroy           // goroutine ends [timestamp]
	traceEvGoDestroySyscall    // goroutine ends in syscall (cgo callback) [timestamp]
	traceEvGoStop              // goroutine yields its time, but is runnable [timestamp, reason, stack ID]
	traceEvGoBlock             // goroutine blocks [timestamp, reason, stack ID]
	traceEvGoUnblock           // goroutine is unblocked [timestamp, goroutine ID, goroutine seq, stack ID]
	traceEvGoSyscallBegin      // syscall enter [timestamp, P seq, stack ID]
	traceEvGoSyscallEnd        // syscall exit [timestamp]
	traceEvGoSyscallEndBlocked // syscall exit and it blocked at some point [timestamp]
	traceEvGoStatus            // goroutine status at the start of a generation [timestamp, goroutine ID, M ID, status]

	// STW.
	traceEvSTWBegin // STW start [timestamp, kind]
	traceEvSTWEnd   // STW done [timestamp]

	// GC events.
	traceEvGCActive           // GC active [timestamp, seq]
	traceEvGCBegin            // GC start [timestamp, seq, stack ID]
	traceEvGCEnd              // GC done [timestamp, seq]
	traceEvGCSweepActive      // GC sweep active [timestamp, P ID]
	traceEvGCSweepBegin       // GC sweep start [timestamp, stack ID]
	traceEvGCSweepEnd         // GC sweep done [timestamp, swept bytes, reclaimed bytes]
	traceEvGCMarkAssistActive // GC mark assist active [timestamp, goroutine ID]
	traceEvGCMarkAssistBegin  // GC mark assist start [timestamp, stack ID]
	traceEvGCMarkAssistEnd    // GC mark assist done [timestamp]
	traceEvHeapAlloc          // gcController.heapLive change [timestamp, heap alloc in bytes]
	traceEvHeapGoal           // gcController.heapGoal() change [timestamp, heap goal in bytes]

	// Annotations.
	traceEvGoLabel         // apply string label to current running goroutine [timestamp, label string ID]
	traceEvUserTaskBegin   // trace.NewTask [timestamp, internal task ID, internal parent task ID, name string ID, stack ID]
	traceEvUserTaskEnd     // end of a task [timestamp, internal task ID, stack ID]
	traceEvUserRegionBegin // trace.{Start,With}Region [timestamp, internal task ID, name string ID, stack ID]
	traceEvUserRegionEnd   // trace.{End,With}Region [timestamp, internal task ID, name string ID, stack ID]
	traceEvUserLog         // trace.Log [timestamp, internal task ID, key string ID, stack, value string ID]

	// Coroutines.
	traceEvGoSwitch        // goroutine switch (coroswitch) [timestamp, goroutine ID, goroutine seq]
	traceEvGoSwitchDestroy // goroutine switch and destroy [timestamp, goroutine ID, goroutine seq]
	traceEvGoCreateBlocked // goroutine creation (starts blocked) [timestamp, new goroutine ID, new stack ID, stack ID]

	// GoStatus with stack.
	traceEvGoStatusStack // goroutine status at the start of a generation, with a stack [timestamp, goroutine ID, M ID, status, stack ID]

	// Batch event for an experimental batch with a custom format.
	traceEvExperimentalBatch // start of extra data [experiment ID, generation, M ID, timestamp, batch length, batch data...]
)

// traceArg is a simple wrapper type to help ensure that arguments passed
// to traces are well-formed.
type traceArg uint64

// traceEventWriter is the high-level API for writing trace events.
//
// See the comment on traceWriter about style for more details as to why
// this type and its methods are structured the way they are.
type traceEventWriter struct {
	w traceWriter
}

// eventWriter creates a new traceEventWriter. It is the main entrypoint for writing trace events.
//
// Before creating the event writer, this method will emit a status for the current goroutine
// or proc if it exists, and if it hasn't had its status emitted yet. goStatus and procStatus indicate
// what the status of goroutine or P should be immediately *before* the events that are about to
// be written using the eventWriter (if they exist). No status will be written if there's no active
// goroutine or P.
//
// Callers can elect to pass a constant value here if the status is clear (e.g. a goroutine must have
// been Runnable before a GoStart). Otherwise, callers can query the status of either the goroutine
// or P and pass the appropriate status.
//
// In this case, the default status should be traceGoBad or traceProcBad to help identify bugs sooner.
func (tl traceLocker) eventWriter(goStatus traceGoStatus, procStatus traceProcStatus) traceEventWriter {
	w := tl.writer()
	if pp := tl.mp.p.ptr(); pp != nil && !pp.trace.statusWasTraced(tl.gen) && pp.trace.acquireStatus(tl.gen) {
		w = w.writeProcStatus(uint64(pp.id), procStatus, pp.trace.inSweep)
	}
	if gp := tl.mp.curg; gp != nil && !gp.trace.statusWasTraced(tl.gen) && gp.trace.acquireStatus(tl.gen) {
		w = w.writeGoStatus(uint64(gp.goid), int64(tl.mp.procid), goStatus, gp.inMarkAssist, 0 /* no stack */)
	}
	return traceEventWriter{w}
}

// commit writes out a trace event and calls end. It's a helper to make the
// common case of writing out a single event less error-prone.
func (e traceEventWriter) commit(ev traceEv, args ...traceArg) {
	e = e.write(ev, args...)
	e.end()
}

// write writes an event into the trace.
func (e traceEventWriter) write(ev traceEv, args ...traceArg) traceEventWriter {
	e.w = e.w.event(ev, args...)
	return e
}

// end finishes writing to the trace. The traceEventWriter must not be used after this call.
func (e traceEventWriter) end() {
	e.w.end()
}

// traceEventWrite is the part of traceEvent that actually writes the event.
func (w traceWriter) event(ev traceEv, args ...traceArg) traceWriter {
	// Make sure we have room.
	w, _ = w.ensure(1 + (len(args)+1)*traceBytesPerNumber)

	// Compute the timestamp diff that we'll put in the trace.
	ts := traceClockNow()
	if ts <= w.traceBuf.lastTime {
		ts = w.traceBuf.lastTime + 1
	}
	tsDiff := uint64(ts - w.traceBuf.lastTime)
	w.traceBuf.lastTime = ts

	// Write out event.
	w.byte(byte(ev))
	w.varint(tsDiff)
	for _, arg := range args {
		w.varint(uint64(arg))
	}
	return w
}

// stack takes a stack trace skipping the provided number of frames.
// It then returns a traceArg representing that stack which may be
// passed to write.
func (tl traceLocker) stack(skip int) traceArg {
	return traceArg(traceStack(skip, nil, tl.gen))
}

// startPC takes a start PC for a goroutine and produces a unique
// stack ID for it.
//
// It then returns a traceArg representing that stack which may be
// passed to write.
func (tl traceLocker) startPC(pc uintptr) traceArg {
	// +PCQuantum because makeTraceFrame expects return PCs and subtracts PCQuantum.
	return traceArg(trace.stackTab[tl.gen%2].put([]uintptr{
		logicalStackSentinel,
		startPCForTrace(pc) + sys.PCQuantum,
	}))
}

// string returns a traceArg representing s which may be passed to write.
// The string is assumed to be relatively short and popular, so it may be
// stored for a while in the string dictionary.
func (tl traceLocker) string(s string) traceArg {
	return traceArg(trace.stringTab[tl.gen%2].put(tl.gen, s))
}

// uniqueString returns a traceArg representing s which may be passed to write.
// The string is assumed to be unique or long, so it will be written out to
// the trace eagerly.
func (tl traceLocker) uniqueString(s string) traceArg {
	return traceArg(trace.stringTab[tl.gen%2].emit(tl.gen, s))
}

// rtype returns a traceArg representing typ which may be passed to write.
func (tl traceLocker) rtype(typ *abi.Type) traceArg {
	return traceArg(trace.typeTab[tl.gen%2].put(typ))
}
