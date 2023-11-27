// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"internal/trace/traceviewer"
	"internal/trace/traceviewer/format"
	tracev2 "internal/trace/v2"
)

var _ generator = &threadGenerator{}

type threadGenerator struct {
	globalRangeGenerator
	globalMetricGenerator
	stackSampleGenerator[tracev2.ThreadID]

	gStates map[tracev2.GoID]*gState[tracev2.ThreadID]
	threads map[tracev2.ThreadID]struct{}
}

func newThreadGenerator() *threadGenerator {
	tg := new(threadGenerator)
	tg.stackSampleGenerator.getResource = func(ev *tracev2.Event) tracev2.ThreadID {
		return ev.Thread()
	}
	tg.gStates = make(map[tracev2.GoID]*gState[tracev2.ThreadID])
	tg.threads = make(map[tracev2.ThreadID]struct{})
	return tg
}

func (g *threadGenerator) Sync() {
	g.globalRangeGenerator.Sync()
}

func (g *threadGenerator) GoroutineLabel(ctx *traceContext, ev *tracev2.Event) {
	l := ev.Label()
	g.gStates[l.Resource.Goroutine()].setLabel(l.Label)
}

func (g *threadGenerator) GoroutineRange(ctx *traceContext, ev *tracev2.Event) {
	r := ev.Range()
	switch ev.Kind() {
	case tracev2.EventRangeBegin:
		g.gStates[r.Scope.Goroutine()].rangeBegin(ev.Time(), r.Name, ev.Stack())
	case tracev2.EventRangeActive:
		g.gStates[r.Scope.Goroutine()].rangeActive(r.Name)
	case tracev2.EventRangeEnd:
		gs := g.gStates[r.Scope.Goroutine()]
		gs.rangeEnd(ev.Time(), r.Name, ev.Stack(), ctx)
	}
}

func (g *threadGenerator) GoroutineTransition(ctx *traceContext, ev *tracev2.Event) {
	if ev.Thread() != tracev2.NoThread {
		if _, ok := g.threads[ev.Thread()]; !ok {
			g.threads[ev.Thread()] = struct{}{}
		}
	}

	st := ev.StateTransition()
	goID := st.Resource.Goroutine()

	// If we haven't seen this goroutine before, create a new
	// gState for it.
	gs, ok := g.gStates[goID]
	if !ok {
		gs = newGState[tracev2.ThreadID](goID)
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
	if from.Executing() && !to.Executing() {
		if to == tracev2.GoWaiting {
			// Goroutine started blocking.
			gs.block(ev.Time(), ev.Stack(), st.Reason, ctx)
		} else {
			gs.stop(ev.Time(), ev.Stack(), ctx)
		}
	}
	if !from.Executing() && to.Executing() {
		start := ev.Time()
		if from == tracev2.GoUndetermined {
			// Back-date the event to the start of the trace.
			start = ctx.startTime
		}
		gs.start(start, ev.Thread(), ctx)
	}

	if from == tracev2.GoWaiting {
		// Goroutine was unblocked.
		gs.unblock(ev.Time(), ev.Stack(), ev.Thread(), ctx)
	}
	if from == tracev2.GoNotExist && to == tracev2.GoRunnable {
		// Goroutine was created.
		gs.created(ev.Time(), ev.Thread(), ev.Stack())
	}
	if from == tracev2.GoSyscall {
		// Exiting syscall.
		gs.syscallEnd(ev.Time(), to != tracev2.GoRunning, ctx)
	}

	// Handle syscalls.
	if to == tracev2.GoSyscall {
		start := ev.Time()
		if from == tracev2.GoUndetermined {
			// Back-date the event to the start of the trace.
			start = ctx.startTime
		}
		// Write down that we've entered a syscall. Note: we might have no P here
		// if we're in a cgo callback or this is a transition from GoUndetermined
		// (i.e. the G has been blocked in a syscall).
		gs.syscallBegin(start, ev.Thread(), ev.Stack())
	}

	// Note down the goroutine transition.
	_, inMarkAssist := gs.activeRanges["GC mark assist"]
	ctx.GoroutineTransition(ctx.elapsed(ev.Time()), viewerGState(from, inMarkAssist), viewerGState(to, inMarkAssist))
}

func (g *threadGenerator) ProcTransition(ctx *traceContext, ev *tracev2.Event) {
	if ev.Thread() != tracev2.NoThread {
		if _, ok := g.threads[ev.Thread()]; !ok {
			g.threads[ev.Thread()] = struct{}{}
		}
	}

	type procArg struct {
		Proc uint64 `json:"proc,omitempty"`
	}
	st := ev.StateTransition()
	viewerEv := traceviewer.InstantEvent{
		Resource: uint64(ev.Thread()),
		Stack:    ctx.Stack(viewerFrames(ev.Stack())),
		Arg:      procArg{Proc: uint64(st.Resource.Proc())},
	}

	from, to := st.Proc()
	if from == to {
		// Filter out no-op events.
		return
	}
	if to.Executing() {
		start := ev.Time()
		if from == tracev2.ProcUndetermined {
			start = ctx.startTime
		}
		viewerEv.Name = "proc start"
		viewerEv.Arg = format.ThreadIDArg{ThreadID: uint64(ev.Thread())}
		viewerEv.Ts = ctx.elapsed(start)
		// TODO(mknyszek): We don't have a state machine for threads, so approximate
		// running threads with running Ps.
		ctx.IncThreadStateCount(ctx.elapsed(start), traceviewer.ThreadStateRunning, 1)
	}
	if from.Executing() {
		start := ev.Time()
		viewerEv.Name = "proc stop"
		viewerEv.Ts = ctx.elapsed(start)
		// TODO(mknyszek): We don't have a state machine for threads, so approximate
		// running threads with running Ps.
		ctx.IncThreadStateCount(ctx.elapsed(start), traceviewer.ThreadStateRunning, -1)
	}
	// TODO(mknyszek): Consider modeling procs differently and have them be
	// transition to and from NotExist when GOMAXPROCS changes. We can emit
	// events for this to clearly delineate GOMAXPROCS changes.

	if viewerEv.Name != "" {
		ctx.Instant(viewerEv)
	}
}

func (g *threadGenerator) ProcRange(ctx *traceContext, ev *tracev2.Event) {
	// TODO(mknyszek): Extend procRangeGenerator to support rendering proc ranges on threads.
}

func (g *threadGenerator) Finish(ctx *traceContext) {
	ctx.SetResourceType("OS THREADS")

	// Finish off global ranges.
	g.globalRangeGenerator.Finish(ctx)

	// Finish off all the goroutine slices.
	for _, gs := range g.gStates {
		gs.finish(ctx)
	}

	// Name all the threads to the emitter.
	for id := range g.threads {
		ctx.Resource(uint64(id), fmt.Sprintf("Thread %d", id))
	}
}
