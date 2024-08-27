// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Serving of pprof-like profiles.

package main

import (
	"cmp"
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"net/http"
	"slices"
	"strings"
	"time"
)

func pprofByGoroutine(compute computePprofFunc, t *parsedTrace) traceviewer.ProfileFunc {
	return func(r *http.Request) ([]traceviewer.ProfileRecord, error) {
		name := r.FormValue("name")
		gToIntervals, err := pprofMatchingGoroutines(name, t)
		if err != nil {
			return nil, err
		}
		return compute(gToIntervals, t.events)
	}
}

func pprofByRegion(compute computePprofFunc, t *parsedTrace) traceviewer.ProfileFunc {
	return func(r *http.Request) ([]traceviewer.ProfileRecord, error) {
		filter, err := newRegionFilter(r)
		if err != nil {
			return nil, err
		}
		gToIntervals, err := pprofMatchingRegions(filter, t)
		if err != nil {
			return nil, err
		}
		return compute(gToIntervals, t.events)
	}
}

// pprofMatchingGoroutines returns the ids of goroutines of the matching name and its interval.
// If the id string is empty, returns nil without an error.
func pprofMatchingGoroutines(name string, t *parsedTrace) (map[trace.GoID][]interval, error) {
	res := make(map[trace.GoID][]interval)
	for _, g := range t.summary.Goroutines {
		if name != "" && g.Name != name {
			continue
		}
		endTime := g.EndTime
		if g.EndTime == 0 {
			endTime = t.endTime() // Use the trace end time, since the goroutine is still live then.
		}
		res[g.ID] = []interval{{start: g.StartTime, end: endTime}}
	}
	if len(res) == 0 {
		return nil, fmt.Errorf("failed to find matching goroutines for name: %s", name)
	}
	return res, nil
}

// pprofMatchingRegions returns the time intervals of matching regions
// grouped by the goroutine id. If the filter is nil, returns nil without an error.
func pprofMatchingRegions(filter *regionFilter, t *parsedTrace) (map[trace.GoID][]interval, error) {
	if filter == nil {
		return nil, nil
	}

	gToIntervals := make(map[trace.GoID][]interval)
	for _, g := range t.summary.Goroutines {
		for _, r := range g.Regions {
			if !filter.match(t, r) {
				continue
			}
			gToIntervals[g.ID] = append(gToIntervals[g.ID], regionInterval(t, r))
		}
	}

	for g, intervals := range gToIntervals {
		// In order to remove nested regions and
		// consider only the outermost regions,
		// first, we sort based on the start time
		// and then scan through to select only the outermost regions.
		slices.SortFunc(intervals, func(a, b interval) int {
			if c := cmp.Compare(a.start, b.start); c != 0 {
				return c
			}
			return cmp.Compare(a.end, b.end)
		})
		var lastTimestamp trace.Time
		var n int
		// Select only the outermost regions.
		for _, i := range intervals {
			if lastTimestamp <= i.start {
				intervals[n] = i // new non-overlapping region starts.
				lastTimestamp = i.end
				n++
			}
			// Otherwise, skip because this region overlaps with a previous region.
		}
		gToIntervals[g] = intervals[:n]
	}
	return gToIntervals, nil
}

type computePprofFunc func(gToIntervals map[trace.GoID][]interval, events []trace.Event) ([]traceviewer.ProfileRecord, error)

// computePprofIO returns a computePprofFunc that generates IO pprof-like profile (time spent in
// IO wait, currently only network blocking event).
func computePprofIO() computePprofFunc {
	return makeComputePprofFunc(trace.GoWaiting, func(reason string) bool {
		return reason == "network"
	})
}

// computePprofBlock returns a computePprofFunc that generates blocking pprof-like profile
// (time spent blocked on synchronization primitives).
func computePprofBlock() computePprofFunc {
	return makeComputePprofFunc(trace.GoWaiting, func(reason string) bool {
		return strings.Contains(reason, "chan") || strings.Contains(reason, "sync") || strings.Contains(reason, "select")
	})
}

// computePprofSyscall returns a computePprofFunc that generates a syscall pprof-like
// profile (time spent in syscalls).
func computePprofSyscall() computePprofFunc {
	return makeComputePprofFunc(trace.GoSyscall, func(_ string) bool {
		return true
	})
}

// computePprofSched returns a computePprofFunc that generates a scheduler latency pprof-like profile
// (time between a goroutine become runnable and actually scheduled for execution).
func computePprofSched() computePprofFunc {
	return makeComputePprofFunc(trace.GoRunnable, func(_ string) bool {
		return true
	})
}

// makeComputePprofFunc returns a computePprofFunc that generates a profile of time goroutines spend
// in a particular state for the specified reasons.
func makeComputePprofFunc(state trace.GoState, trackReason func(string) bool) computePprofFunc {
	return func(gToIntervals map[trace.GoID][]interval, events []trace.Event) ([]traceviewer.ProfileRecord, error) {
		stacks := newStackMap()
		tracking := make(map[trace.GoID]*trace.Event)
		for i := range events {
			ev := &events[i]

			// Filter out any non-state-transitions and events without stacks.
			if ev.Kind() != trace.EventStateTransition {
				continue
			}
			stack := ev.Stack()
			if stack == trace.NoStack {
				continue
			}

			// The state transition has to apply to a goroutine.
			st := ev.StateTransition()
			if st.Resource.Kind != trace.ResourceGoroutine {
				continue
			}
			id := st.Resource.Goroutine()
			_, new := st.Goroutine()

			// Check if we're tracking this goroutine.
			startEv := tracking[id]
			if startEv == nil {
				// We're not. Start tracking if the new state
				// matches what we want and the transition is
				// for one of the reasons we care about.
				if new == state && trackReason(st.Reason) {
					tracking[id] = ev
				}
				continue
			}
			// We're tracking this goroutine.
			if new == state {
				// We're tracking this goroutine, but it's just transitioning
				// to the same state (this is a no-ip
				continue
			}
			// The goroutine has transitioned out of the state we care about,
			// so remove it from tracking and record the stack.
			delete(tracking, id)

			overlapping := pprofOverlappingDuration(gToIntervals, id, interval{startEv.Time(), ev.Time()})
			if overlapping > 0 {
				rec := stacks.getOrAdd(startEv.Stack())
				rec.Count++
				rec.Time += overlapping
			}
		}
		return stacks.profile(), nil
	}
}

// pprofOverlappingDuration returns the overlapping duration between
// the time intervals in gToIntervals and the specified event.
// If gToIntervals is nil, this simply returns the event's duration.
func pprofOverlappingDuration(gToIntervals map[trace.GoID][]interval, id trace.GoID, sample interval) time.Duration {
	if gToIntervals == nil { // No filtering.
		return sample.duration()
	}
	intervals := gToIntervals[id]
	if len(intervals) == 0 {
		return 0
	}

	var overlapping time.Duration
	for _, i := range intervals {
		if o := i.overlap(sample); o > 0 {
			overlapping += o
		}
	}
	return overlapping
}

// interval represents a time interval in the trace.
type interval struct {
	start, end trace.Time
}

func (i interval) duration() time.Duration {
	return i.end.Sub(i.start)
}

func (i1 interval) overlap(i2 interval) time.Duration {
	// Assume start1 <= end1 and start2 <= end2
	if i1.end < i2.start || i2.end < i1.start {
		return 0
	}
	if i1.start < i2.start { // choose the later one
		i1.start = i2.start
	}
	if i1.end > i2.end { // choose the earlier one
		i1.end = i2.end
	}
	return i1.duration()
}

// pprofMaxStack is the extent of the deduplication we're willing to do.
//
// Because slices aren't comparable and we want to leverage maps for deduplication,
// we have to choose a fixed constant upper bound on the amount of frames we want
// to support. In practice this is fine because there's a maximum depth to these
// stacks anyway.
const pprofMaxStack = 128

// stackMap is a map of trace.Stack to some value V.
type stackMap struct {
	// stacks contains the full list of stacks in the set, however
	// it is insufficient for deduplication because trace.Stack
	// equality is only optimistic. If two trace.Stacks are equal,
	// then they are guaranteed to be equal in content. If they are
	// not equal, then they might still be equal in content.
	stacks map[trace.Stack]*traceviewer.ProfileRecord

	// pcs is the source-of-truth for deduplication. It is a map of
	// the actual PCs in the stack to a trace.Stack.
	pcs map[[pprofMaxStack]uint64]trace.Stack
}

func newStackMap() *stackMap {
	return &stackMap{
		stacks: make(map[trace.Stack]*traceviewer.ProfileRecord),
		pcs:    make(map[[pprofMaxStack]uint64]trace.Stack),
	}
}

func (m *stackMap) getOrAdd(stack trace.Stack) *traceviewer.ProfileRecord {
	// Fast path: check to see if this exact stack is already in the map.
	if rec, ok := m.stacks[stack]; ok {
		return rec
	}
	// Slow path: the stack may still be in the map.

	// Grab the stack's PCs as the source-of-truth.
	var pcs [pprofMaxStack]uint64
	pcsForStack(stack, &pcs)

	// Check the source-of-truth.
	var rec *traceviewer.ProfileRecord
	if existing, ok := m.pcs[pcs]; ok {
		// In the map.
		rec = m.stacks[existing]
		delete(m.stacks, existing)
	} else {
		// Not in the map.
		rec = new(traceviewer.ProfileRecord)
	}
	// Insert regardless of whether we have a match in m.pcs.
	// Even if we have a match, we want to keep the newest version
	// of that stack, since we're much more likely tos see it again
	// as we iterate through the trace linearly. Simultaneously, we
	// are likely to never see the old stack again.
	m.pcs[pcs] = stack
	m.stacks[stack] = rec
	return rec
}

func (m *stackMap) profile() []traceviewer.ProfileRecord {
	prof := make([]traceviewer.ProfileRecord, 0, len(m.stacks))
	for stack, record := range m.stacks {
		rec := *record
		i := 0
		stack.Frames()(func(frame trace.StackFrame) bool {
			rec.Stack = append(rec.Stack, &trace.Frame{
				PC:   frame.PC,
				Fn:   frame.Func,
				File: frame.File,
				Line: int(frame.Line),
			})
			i++
			// Cut this off at pprofMaxStack because that's as far
			// as our deduplication goes.
			return i < pprofMaxStack
		})
		prof = append(prof, rec)
	}
	return prof
}

// pcsForStack extracts the first pprofMaxStack PCs from stack into pcs.
func pcsForStack(stack trace.Stack, pcs *[pprofMaxStack]uint64) {
	i := 0
	stack.Frames()(func(frame trace.StackFrame) bool {
		pcs[i] = frame.PC
		i++
		return i < len(pcs)
	})
}
