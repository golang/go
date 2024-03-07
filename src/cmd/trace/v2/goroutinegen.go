// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	tracev2 "internal/trace/v2"
)

var _ generator = &goroutineGenerator{}

type goroutineGenerator struct {
	globalRangeGenerator
	globalMetricGenerator
	stackSampleGenerator[tracev2.GoID]
	logEventGenerator[tracev2.GoID]

	gStates map[tracev2.GoID]*gState[tracev2.GoID]
	focus   tracev2.GoID
	filter  map[tracev2.GoID]struct{}
}

func newGoroutineGenerator(ctx *traceContext, focus tracev2.GoID, filter map[tracev2.GoID]struct{}) *goroutineGenerator {
	gg := new(goroutineGenerator)
	rg := func(ev *tracev2.Event) tracev2.GoID {
		return ev.Goroutine()
	}
	gg.stackSampleGenerator.getResource = rg
	gg.logEventGenerator.getResource = rg
	gg.gStates = make(map[tracev2.GoID]*gState[tracev2.GoID])
	gg.focus = focus
	gg.filter = filter

	// Enable a filter on the emitter.
	if filter != nil {
		ctx.SetResourceFilter(func(resource uint64) bool {
			_, ok := filter[tracev2.GoID(resource)]
			return ok
		})
	}
	return gg
}

func (g *goroutineGenerator) Sync() {
	g.globalRangeGenerator.Sync()
}

func (g *goroutineGenerator) GoroutineLabel(ctx *traceContext, ev *tracev2.Event) {
	l := ev.Label()
	g.gStates[l.Resource.Goroutine()].setLabel(l.Label)
}

func (g *goroutineGenerator) GoroutineRange(ctx *traceContext, ev *tracev2.Event) {
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

func (g *goroutineGenerator) GoroutineTransition(ctx *traceContext, ev *tracev2.Event) {
	st := ev.StateTransition()
	goID := st.Resource.Goroutine()

	// If we haven't seen this goroutine before, create a new
	// gState for it.
	gs, ok := g.gStates[goID]
	if !ok {
		gs = newGState[tracev2.GoID](goID)
		g.gStates[goID] = gs
	}

	// Try to augment the name of the goroutine.
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
		gs.start(start, goID, ctx)
	}

	if from == tracev2.GoWaiting {
		// Goroutine unblocked.
		gs.unblock(ev.Time(), ev.Stack(), ev.Goroutine(), ctx)
	}
	if from == tracev2.GoNotExist && to == tracev2.GoRunnable {
		// Goroutine was created.
		gs.created(ev.Time(), ev.Goroutine(), ev.Stack())
	}
	if from == tracev2.GoSyscall && to != tracev2.GoRunning {
		// Exiting blocked syscall.
		gs.syscallEnd(ev.Time(), true, ctx)
		gs.blockedSyscallEnd(ev.Time(), ev.Stack(), ctx)
	} else if from == tracev2.GoSyscall {
		// Check if we're exiting a syscall in a non-blocking way.
		gs.syscallEnd(ev.Time(), false, ctx)
	}

	// Handle syscalls.
	if to == tracev2.GoSyscall {
		start := ev.Time()
		if from == tracev2.GoUndetermined {
			// Back-date the event to the start of the trace.
			start = ctx.startTime
		}
		// Write down that we've entered a syscall. Note: we might have no G or P here
		// if we're in a cgo callback or this is a transition from GoUndetermined
		// (i.e. the G has been blocked in a syscall).
		gs.syscallBegin(start, goID, ev.Stack())
	}

	// Note down the goroutine transition.
	_, inMarkAssist := gs.activeRanges["GC mark assist"]
	ctx.GoroutineTransition(ctx.elapsed(ev.Time()), viewerGState(from, inMarkAssist), viewerGState(to, inMarkAssist))
}

func (g *goroutineGenerator) ProcRange(ctx *traceContext, ev *tracev2.Event) {
	// TODO(mknyszek): Extend procRangeGenerator to support rendering proc ranges
	// that overlap with a goroutine's execution.
}

func (g *goroutineGenerator) ProcTransition(ctx *traceContext, ev *tracev2.Event) {
	// Not needed. All relevant information for goroutines can be derived from goroutine transitions.
}

func (g *goroutineGenerator) Finish(ctx *traceContext) {
	ctx.SetResourceType("G")

	// Finish off global ranges.
	g.globalRangeGenerator.Finish(ctx)

	// Finish off all the goroutine slices.
	for id, gs := range g.gStates {
		gs.finish(ctx)

		// Tell the emitter about the goroutines we want to render.
		ctx.Resource(uint64(id), gs.name())
	}

	// Set the goroutine to focus on.
	if g.focus != tracev2.NoGoroutine {
		ctx.Focus(uint64(g.focus))
	}
}
