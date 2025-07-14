// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace event writing API for trace2runtime.go.

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
	"internal/trace/tracev2"
)

// traceArg is a simple wrapper type to help ensure that arguments passed
// to traces are well-formed.
type traceArg uint64

// traceEventWriter is the high-level API for writing trace events.
//
// See the comment on traceWriter about style for more details as to why
// this type and its methods are structured the way they are.
type traceEventWriter struct {
	tl traceLocker
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
// In this case, the default status should be tracev2.GoBad or tracev2.ProcBad to help identify bugs sooner.
func (tl traceLocker) eventWriter(goStatus tracev2.GoStatus, procStatus tracev2.ProcStatus) traceEventWriter {
	if pp := tl.mp.p.ptr(); pp != nil && !pp.trace.statusWasTraced(tl.gen) && pp.trace.acquireStatus(tl.gen) {
		tl.writer().writeProcStatus(uint64(pp.id), procStatus, pp.trace.inSweep).end()
	}
	if gp := tl.mp.curg; gp != nil && !gp.trace.statusWasTraced(tl.gen) && gp.trace.acquireStatus(tl.gen) {
		tl.writer().writeGoStatus(uint64(gp.goid), int64(tl.mp.procid), goStatus, gp.inMarkAssist, 0 /* no stack */).end()
	}
	return traceEventWriter{tl}
}

// event writes out a trace event.
func (e traceEventWriter) event(ev tracev2.EventType, args ...traceArg) {
	e.tl.writer().event(ev, args...).end()
}

// stack takes a stack trace skipping the provided number of frames.
// It then returns a traceArg representing that stack which may be
// passed to write.
func (tl traceLocker) stack(skip int) traceArg {
	return traceArg(traceStack(skip, nil, &trace.stackTab[tl.gen%2]))
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
