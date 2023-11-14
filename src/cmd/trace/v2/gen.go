// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
	"strings"
)

// generator is an interface for generating a JSON trace for the trace viewer
// from a trace. Each method in this interface is a handler for a kind of event
// that is interesting to render in the UI via the JSON trace.
type generator interface {
	// Global parts.
	Sync() // Notifies the generator of an EventSync event.
	StackSample(ctx *traceContext, ev *tracev2.Event)
	GlobalRange(ctx *traceContext, ev *tracev2.Event)
	GlobalMetric(ctx *traceContext, ev *tracev2.Event)

	// Goroutine parts.
	GoroutineLabel(ctx *traceContext, ev *tracev2.Event)
	GoroutineRange(ctx *traceContext, ev *tracev2.Event)
	GoroutineTransition(ctx *traceContext, ev *tracev2.Event)

	// Proc parts.
	ProcRange(ctx *traceContext, ev *tracev2.Event)
	ProcTransition(ctx *traceContext, ev *tracev2.Event)

	// Finish indicates the end of the trace and finalizes generation.
	Finish(ctx *traceContext, endTime tracev2.Time)
}

// runGenerator produces a trace into ctx by running the generator over the parsed trace.
func runGenerator(ctx *traceContext, g generator, parsed *parsedTrace) {
	for i := range parsed.events {
		ev := &parsed.events[i]

		switch ev.Kind() {
		case tracev2.EventSync:
			g.Sync()
		case tracev2.EventStackSample:
			g.StackSample(ctx, ev)
		case tracev2.EventRangeBegin, tracev2.EventRangeActive, tracev2.EventRangeEnd:
			r := ev.Range()
			switch r.Scope.Kind {
			case tracev2.ResourceGoroutine:
				g.GoroutineRange(ctx, ev)
			case tracev2.ResourceProc:
				g.ProcRange(ctx, ev)
			case tracev2.ResourceNone:
				g.GlobalRange(ctx, ev)
			}
		case tracev2.EventMetric:
			g.GlobalMetric(ctx, ev)
		case tracev2.EventLabel:
			l := ev.Label()
			if l.Resource.Kind == tracev2.ResourceGoroutine {
				g.GoroutineLabel(ctx, ev)
			}
		case tracev2.EventStateTransition:
			switch ev.StateTransition().Resource.Kind {
			case tracev2.ResourceProc:
				g.ProcTransition(ctx, ev)
			case tracev2.ResourceGoroutine:
				g.GoroutineTransition(ctx, ev)
			}
		}
	}
	g.Finish(ctx, parsed.events[len(parsed.events)-1].Time())
}

// Building blocks for generators.

// stackSampleGenerator implements a generic handler for stack sample events.
// The provided resource is the resource the stack sample should count against.
type stackSampleGenerator[R resource] struct {
	// getResource is a function to extract a resource ID from a stack sample event.
	getResource func(*tracev2.Event) R
}

// StackSample implements a stack sample event handler. It expects ev to be one such event.
func (g *stackSampleGenerator[R]) StackSample(ctx *traceContext, ev *tracev2.Event) {
	id := g.getResource(ev)
	if id == R(noResource) {
		// We have nowhere to put this in the UI.
		return
	}
	ctx.Instant(traceviewer.InstantEvent{
		Name:     "CPU profile sample",
		Ts:       ctx.elapsed(ev.Time()),
		Resource: uint64(id),
		Stack:    ctx.Stack(viewerFrames(ev.Stack())),
	})
}

// globalRangeGenerator implements a generic handler for EventRange* events that pertain
// to tracev2.ResourceNone (the global scope).
type globalRangeGenerator struct {
	ranges   map[string]activeRange
	seenSync bool
}

// Sync notifies the generator of an EventSync event.
func (g *globalRangeGenerator) Sync() {
	g.seenSync = true
}

// GlobalRange implements a handler for EventRange* events whose Scope.Kind is ResourceNone.
// It expects ev to be one such event.
func (g *globalRangeGenerator) GlobalRange(ctx *traceContext, ev *tracev2.Event) {
	if g.ranges == nil {
		g.ranges = make(map[string]activeRange)
	}
	r := ev.Range()
	switch ev.Kind() {
	case tracev2.EventRangeBegin:
		g.ranges[r.Name] = activeRange{ev.Time(), ev.Stack()}
	case tracev2.EventRangeActive:
		// If we've seen a Sync event, then Active events are always redundant.
		if !g.seenSync {
			// Otherwise, they extend back to the start of the trace.
			g.ranges[r.Name] = activeRange{ctx.startTime, ev.Stack()}
		}
	case tracev2.EventRangeEnd:
		// Only emit GC events, because we have nowhere to
		// put other events.
		ar := g.ranges[r.Name]
		if strings.Contains(r.Name, "GC") {
			ctx.Slice(traceviewer.SliceEvent{
				Name:     r.Name,
				Ts:       ctx.elapsed(ar.time),
				Dur:      ev.Time().Sub(ar.time),
				Resource: trace.GCP,
				Stack:    ctx.Stack(viewerFrames(ar.stack)),
				EndStack: ctx.Stack(viewerFrames(ev.Stack())),
			})
		}
		delete(g.ranges, r.Name)
	}
}

// Finish flushes any outstanding ranges at the end of the trace.
func (g *globalRangeGenerator) Finish(ctx *traceContext, endTime tracev2.Time) {
	for name, ar := range g.ranges {
		if !strings.Contains(name, "GC") {
			continue
		}
		ctx.Slice(traceviewer.SliceEvent{
			Name:     name,
			Ts:       ctx.elapsed(ar.time),
			Dur:      endTime.Sub(ar.time),
			Resource: trace.GCP,
			Stack:    ctx.Stack(viewerFrames(ar.stack)),
		})
	}
}

// globalMetricGenerator implements a generic handler for Metric events.
type globalMetricGenerator struct {
}

// GlobalMetric implements an event handler for EventMetric events. ev must be one such event.
func (g *globalMetricGenerator) GlobalMetric(ctx *traceContext, ev *tracev2.Event) {
	m := ev.Metric()
	switch m.Name {
	case "/memory/classes/heap/objects:bytes":
		ctx.HeapAlloc(ctx.elapsed(ev.Time()), m.Value.Uint64())
	case "/gc/heap/goal:bytes":
		ctx.HeapGoal(ctx.elapsed(ev.Time()), m.Value.Uint64())
	case "/sched/gomaxprocs:threads":
		ctx.Gomaxprocs(m.Value.Uint64())
	}
}

// procRangeGenerator implements a generic handler for EventRange* events whose Scope.Kind is
// ResourceProc.
type procRangeGenerator struct {
	ranges   map[tracev2.Range]activeRange
	seenSync bool
}

// Sync notifies the generator of an EventSync event.
func (g *procRangeGenerator) Sync() {
	g.seenSync = true
}

// ProcRange implements a handler for EventRange* events whose Scope.Kind is ResourceProc.
// It expects ev to be one such event.
func (g *procRangeGenerator) ProcRange(ctx *traceContext, ev *tracev2.Event) {
	if g.ranges == nil {
		g.ranges = make(map[tracev2.Range]activeRange)
	}
	r := ev.Range()
	switch ev.Kind() {
	case tracev2.EventRangeBegin:
		g.ranges[r] = activeRange{ev.Time(), ev.Stack()}
	case tracev2.EventRangeActive:
		// If we've seen a Sync event, then Active events are always redundant.
		if !g.seenSync {
			// Otherwise, they extend back to the start of the trace.
			g.ranges[r] = activeRange{ctx.startTime, ev.Stack()}
		}
	case tracev2.EventRangeEnd:
		// Emit proc-based ranges.
		ar := g.ranges[r]
		ctx.Slice(traceviewer.SliceEvent{
			Name:     r.Name,
			Ts:       ctx.elapsed(ar.time),
			Dur:      ev.Time().Sub(ar.time),
			Resource: uint64(r.Scope.Proc()),
			Stack:    ctx.Stack(viewerFrames(ar.stack)),
			EndStack: ctx.Stack(viewerFrames(ev.Stack())),
		})
		delete(g.ranges, r)
	}
}

// Finish flushes any outstanding ranges at the end of the trace.
func (g *procRangeGenerator) Finish(ctx *traceContext, endTime tracev2.Time) {
	for r, ar := range g.ranges {
		ctx.Slice(traceviewer.SliceEvent{
			Name:     r.Name,
			Ts:       ctx.elapsed(ar.time),
			Dur:      endTime.Sub(ar.time),
			Resource: uint64(r.Scope.Proc()),
			Stack:    ctx.Stack(viewerFrames(ar.stack)),
		})
	}
}

// activeRange represents an active EventRange* range.
type activeRange struct {
	time  tracev2.Time
	stack tracev2.Stack
}

// completedRange represents a completed EventRange* range.
type completedRange struct {
	name       string
	startTime  tracev2.Time
	endTime    tracev2.Time
	startStack tracev2.Stack
	endStack   tracev2.Stack
	arg        any
}
