// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"internal/trace/traceviewer/format"
)

var _ generator = &procGenerator{}

type procGenerator struct {
	globalRangeGenerator
	globalMetricGenerator
	procRangeGenerator
	stackSampleGenerator[trace.ProcID]
	logEventGenerator[trace.ProcID]

	gStates   map[trace.GoID]*gState[trace.ProcID]
	inSyscall map[trace.ProcID]*gState[trace.ProcID]
	maxProc   trace.ProcID
}

func newProcGenerator() *procGenerator {
	pg := new(procGenerator)
	rg := func(ev *trace.Event) trace.ProcID {
		return ev.Proc()
	}
	pg.stackSampleGenerator.getResource = rg
	pg.logEventGenerator.getResource = rg
	pg.gStates = make(map[trace.GoID]*gState[trace.ProcID])
	pg.inSyscall = make(map[trace.ProcID]*gState[trace.ProcID])
	return pg
}

func (g *procGenerator) Sync() {
	g.globalRangeGenerator.Sync()
	g.procRangeGenerator.Sync()
}

func (g *procGenerator) GoroutineLabel(ctx *traceContext, ev *trace.Event) {
	l := ev.Label()
	g.gStates[l.Resource.Goroutine()].setLabel(l.Label)
}

func (g *procGenerator) GoroutineRange(ctx *traceContext, ev *trace.Event) {
	r := ev.Range()
	switch ev.Kind() {
	case trace.EventRangeBegin:
		g.gStates[r.Scope.Goroutine()].rangeBegin(ev.Time(), r.Name, ev.Stack())
	case trace.EventRangeActive:
		g.gStates[r.Scope.Goroutine()].rangeActive(r.Name)
	case trace.EventRangeEnd:
		gs := g.gStates[r.Scope.Goroutine()]
		gs.rangeEnd(ev.Time(), r.Name, ev.Stack(), ctx)
	}
}

func (g *procGenerator) GoroutineTransition(ctx *traceContext, ev *trace.Event) {
	st := ev.StateTransition()
	goID := st.Resource.Goroutine()

	// If we haven't seen this goroutine before, create a new
	// gState for it.
	gs, ok := g.gStates[goID]
	if !ok {
		gs = newGState[trace.ProcID](goID)
		g.gStates[goID] = gs
	}
	// If we haven't already named this goroutine, try to name it.
	gs.augmentName(st.Stack)

	// Handle the goroutine state transition.
	from, to := st.Goroutine()
	if from == to {
		// Filter out no-op events.
		return
	}
	if from == trace.GoRunning && !to.Executing() {
		if to == trace.GoWaiting {
			// Goroutine started blocking.
			gs.block(ev.Time(), ev.Stack(), st.Reason, ctx)
		} else {
			gs.stop(ev.Time(), ev.Stack(), ctx)
		}
	}
	if !from.Executing() && to == trace.GoRunning {
		start := ev.Time()
		if from == trace.GoUndetermined {
			// Back-date the event to the start of the trace.
			start = ctx.startTime
		}
		gs.start(start, ev.Proc(), ctx)
	}

	if from == trace.GoWaiting {
		// Goroutine was unblocked.
		gs.unblock(ev.Time(), ev.Stack(), ev.Proc(), ctx)
	}
	if from == trace.GoNotExist && to == trace.GoRunnable {
		// Goroutine was created.
		gs.created(ev.Time(), ev.Proc(), ev.Stack())
	}
	if from == trace.GoSyscall && to != trace.GoRunning {
		// Goroutine exited a blocked syscall.
		gs.blockedSyscallEnd(ev.Time(), ev.Stack(), ctx)
	}

	// Handle syscalls.
	if to == trace.GoSyscall && ev.Proc() != trace.NoProc {
		start := ev.Time()
		if from == trace.GoUndetermined {
			// Back-date the event to the start of the trace.
			start = ctx.startTime
		}
		// Write down that we've entered a syscall. Note: we might have no P here
		// if we're in a cgo callback or this is a transition from GoUndetermined
		// (i.e. the G has been blocked in a syscall).
		gs.syscallBegin(start, ev.Proc(), ev.Stack())
		g.inSyscall[ev.Proc()] = gs
	}
	// Check if we're exiting a non-blocking syscall.
	_, didNotBlock := g.inSyscall[ev.Proc()]
	if from == trace.GoSyscall && didNotBlock {
		gs.syscallEnd(ev.Time(), false, ctx)
		delete(g.inSyscall, ev.Proc())
	}

	// Note down the goroutine transition.
	_, inMarkAssist := gs.activeRanges["GC mark assist"]
	ctx.GoroutineTransition(ctx.elapsed(ev.Time()), viewerGState(from, inMarkAssist), viewerGState(to, inMarkAssist))
}

func (g *procGenerator) ProcTransition(ctx *traceContext, ev *trace.Event) {
	st := ev.StateTransition()
	proc := st.Resource.Proc()

	g.maxProc = max(g.maxProc, proc)
	viewerEv := traceviewer.InstantEvent{
		Resource: uint64(proc),
		Stack:    ctx.Stack(viewerFrames(ev.Stack())),
	}

	from, to := st.Proc()
	if from == to {
		// Filter out no-op events.
		return
	}
	if to.Executing() {
		start := ev.Time()
		if from == trace.ProcUndetermined {
			start = ctx.startTime
		}
		viewerEv.Name = "proc start"
		viewerEv.Arg = format.ThreadIDArg{ThreadID: uint64(ev.Thread())}
		viewerEv.Ts = ctx.elapsed(start)
		ctx.IncThreadStateCount(ctx.elapsed(start), traceviewer.ThreadStateRunning, 1)
	}
	if from.Executing() {
		start := ev.Time()
		viewerEv.Name = "proc stop"
		viewerEv.Ts = ctx.elapsed(start)
		ctx.IncThreadStateCount(ctx.elapsed(start), traceviewer.ThreadStateRunning, -1)

		// Check if this proc was in a syscall before it stopped.
		// This means the syscall blocked. We need to emit it to the
		// viewer at this point because we only display the time the
		// syscall occupied a P when the viewer is in per-P mode.
		//
		// TODO(mknyszek): We could do better in a per-M mode because
		// all events have to happen on *some* thread, and in v2 traces
		// we know what that thread is.
		gs, ok := g.inSyscall[proc]
		if ok {
			// Emit syscall slice for blocked syscall.
			gs.syscallEnd(start, true, ctx)
			gs.stop(start, ev.Stack(), ctx)
			delete(g.inSyscall, proc)
		}
	}
	// TODO(mknyszek): Consider modeling procs differently and have them be
	// transition to and from NotExist when GOMAXPROCS changes. We can emit
	// events for this to clearly delineate GOMAXPROCS changes.

	if viewerEv.Name != "" {
		ctx.Instant(viewerEv)
	}
}

func (g *procGenerator) Finish(ctx *traceContext) {
	ctx.SetResourceType("PROCS")

	// Finish off ranges first. It doesn't really matter for the global ranges,
	// but the proc ranges need to either be a subset of a goroutine slice or
	// their own slice entirely. If the former, it needs to end first.
	g.procRangeGenerator.Finish(ctx)
	g.globalRangeGenerator.Finish(ctx)

	// Finish off all the goroutine slices.
	for _, gs := range g.gStates {
		gs.finish(ctx)
	}

	// Name all the procs to the emitter.
	for i := uint64(0); i <= uint64(g.maxProc); i++ {
		ctx.Resource(i, fmt.Sprintf("Proc %v", i))
	}
}
