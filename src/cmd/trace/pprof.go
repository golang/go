// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Serving of pprof-like profiles.

package main

import (
	"fmt"
	"internal/trace"
	"internal/trace/traceviewer"
	"net/http"
	"sort"
	"strconv"
	"time"
)

func init() {
	http.HandleFunc("/io", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofIO)))
	http.HandleFunc("/block", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofBlock)))
	http.HandleFunc("/syscall", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofSyscall)))
	http.HandleFunc("/sched", traceviewer.SVGProfileHandlerFunc(pprofByGoroutine(computePprofSched)))

	http.HandleFunc("/regionio", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofIO)))
	http.HandleFunc("/regionblock", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofBlock)))
	http.HandleFunc("/regionsyscall", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofSyscall)))
	http.HandleFunc("/regionsched", traceviewer.SVGProfileHandlerFunc(pprofByRegion(computePprofSched)))
}

// interval represents a time interval in the trace.
type interval struct {
	begin, end int64 // nanoseconds.
}

func pprofByGoroutine(compute computePprofFunc) traceviewer.ProfileFunc {
	return func(r *http.Request) ([]traceviewer.ProfileRecord, error) {
		id := r.FormValue("id")
		events, err := parseEvents()
		if err != nil {
			return nil, err
		}
		gToIntervals, err := pprofMatchingGoroutines(id, events)
		if err != nil {
			return nil, err
		}
		return compute(gToIntervals, events)
	}
}

func pprofByRegion(compute computePprofFunc) traceviewer.ProfileFunc {
	return func(r *http.Request) ([]traceviewer.ProfileRecord, error) {
		filter, err := newRegionFilter(r)
		if err != nil {
			return nil, err
		}
		gToIntervals, err := pprofMatchingRegions(filter)
		if err != nil {
			return nil, err
		}
		events, _ := parseEvents()

		return compute(gToIntervals, events)
	}
}

// pprofMatchingGoroutines parses the goroutine type id string (i.e. pc)
// and returns the ids of goroutines of the matching type and its interval.
// If the id string is empty, returns nil without an error.
func pprofMatchingGoroutines(id string, events []*trace.Event) (map[uint64][]interval, error) {
	if id == "" {
		return nil, nil
	}
	pc, err := strconv.ParseUint(id, 10, 64) // id is string
	if err != nil {
		return nil, fmt.Errorf("invalid goroutine type: %v", id)
	}
	analyzeGoroutines(events)
	var res map[uint64][]interval
	for _, g := range gs {
		if g.PC != pc {
			continue
		}
		if res == nil {
			res = make(map[uint64][]interval)
		}
		endTime := g.EndTime
		if g.EndTime == 0 {
			endTime = lastTimestamp() // the trace doesn't include the goroutine end event. Use the trace end time.
		}
		res[g.ID] = []interval{{begin: g.StartTime, end: endTime}}
	}
	if len(res) == 0 && id != "" {
		return nil, fmt.Errorf("failed to find matching goroutines for id: %s", id)
	}
	return res, nil
}

// pprofMatchingRegions returns the time intervals of matching regions
// grouped by the goroutine id. If the filter is nil, returns nil without an error.
func pprofMatchingRegions(filter *regionFilter) (map[uint64][]interval, error) {
	res, err := analyzeAnnotations()
	if err != nil {
		return nil, err
	}
	if filter == nil {
		return nil, nil
	}

	gToIntervals := make(map[uint64][]interval)
	for id, regions := range res.regions {
		for _, s := range regions {
			if filter.match(id, s) {
				gToIntervals[s.G] = append(gToIntervals[s.G], interval{begin: s.firstTimestamp(), end: s.lastTimestamp()})
			}
		}
	}

	for g, intervals := range gToIntervals {
		// in order to remove nested regions and
		// consider only the outermost regions,
		// first, we sort based on the start time
		// and then scan through to select only the outermost regions.
		sort.Slice(intervals, func(i, j int) bool {
			x := intervals[i].begin
			y := intervals[j].begin
			if x == y {
				return intervals[i].end < intervals[j].end
			}
			return x < y
		})
		var lastTimestamp int64
		var n int
		// select only the outermost regions.
		for _, i := range intervals {
			if lastTimestamp <= i.begin {
				intervals[n] = i // new non-overlapping region starts.
				lastTimestamp = i.end
				n++
			} // otherwise, skip because this region overlaps with a previous region.
		}
		gToIntervals[g] = intervals[:n]
	}
	return gToIntervals, nil
}

type computePprofFunc func(gToIntervals map[uint64][]interval, events []*trace.Event) ([]traceviewer.ProfileRecord, error)

// computePprofIO generates IO pprof-like profile (time spent in IO wait, currently only network blocking event).
func computePprofIO(gToIntervals map[uint64][]interval, events []*trace.Event) ([]traceviewer.ProfileRecord, error) {
	prof := make(map[uint64]traceviewer.ProfileRecord)
	for _, ev := range events {
		if ev.Type != trace.EvGoBlockNet || ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		overlapping := pprofOverlappingDuration(gToIntervals, ev)
		if overlapping > 0 {
			rec := prof[ev.StkID]
			rec.Stack = ev.Stk
			rec.Count++
			rec.Time += overlapping
			prof[ev.StkID] = rec
		}
	}
	return recordsOf(prof), nil
}

// computePprofBlock generates blocking pprof-like profile (time spent blocked on synchronization primitives).
func computePprofBlock(gToIntervals map[uint64][]interval, events []*trace.Event) ([]traceviewer.ProfileRecord, error) {
	prof := make(map[uint64]traceviewer.ProfileRecord)
	for _, ev := range events {
		switch ev.Type {
		case trace.EvGoBlockSend, trace.EvGoBlockRecv, trace.EvGoBlockSelect,
			trace.EvGoBlockSync, trace.EvGoBlockCond, trace.EvGoBlockGC:
			// TODO(hyangah): figure out why EvGoBlockGC should be here.
			// EvGoBlockGC indicates the goroutine blocks on GC assist, not
			// on synchronization primitives.
		default:
			continue
		}
		if ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		overlapping := pprofOverlappingDuration(gToIntervals, ev)
		if overlapping > 0 {
			rec := prof[ev.StkID]
			rec.Stack = ev.Stk
			rec.Count++
			rec.Time += overlapping
			prof[ev.StkID] = rec
		}
	}
	return recordsOf(prof), nil
}

// computePprofSyscall generates syscall pprof-like profile (time spent blocked in syscalls).
func computePprofSyscall(gToIntervals map[uint64][]interval, events []*trace.Event) ([]traceviewer.ProfileRecord, error) {
	prof := make(map[uint64]traceviewer.ProfileRecord)
	for _, ev := range events {
		if ev.Type != trace.EvGoSysCall || ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		overlapping := pprofOverlappingDuration(gToIntervals, ev)
		if overlapping > 0 {
			rec := prof[ev.StkID]
			rec.Stack = ev.Stk
			rec.Count++
			rec.Time += overlapping
			prof[ev.StkID] = rec
		}
	}
	return recordsOf(prof), nil
}

// computePprofSched generates scheduler latency pprof-like profile
// (time between a goroutine become runnable and actually scheduled for execution).
func computePprofSched(gToIntervals map[uint64][]interval, events []*trace.Event) ([]traceviewer.ProfileRecord, error) {
	prof := make(map[uint64]traceviewer.ProfileRecord)
	for _, ev := range events {
		if (ev.Type != trace.EvGoUnblock && ev.Type != trace.EvGoCreate) ||
			ev.Link == nil || ev.StkID == 0 || len(ev.Stk) == 0 {
			continue
		}
		overlapping := pprofOverlappingDuration(gToIntervals, ev)
		if overlapping > 0 {
			rec := prof[ev.StkID]
			rec.Stack = ev.Stk
			rec.Count++
			rec.Time += overlapping
			prof[ev.StkID] = rec
		}
	}
	return recordsOf(prof), nil
}

// pprofOverlappingDuration returns the overlapping duration between
// the time intervals in gToIntervals and the specified event.
// If gToIntervals is nil, this simply returns the event's duration.
func pprofOverlappingDuration(gToIntervals map[uint64][]interval, ev *trace.Event) time.Duration {
	if gToIntervals == nil { // No filtering.
		return time.Duration(ev.Link.Ts-ev.Ts) * time.Nanosecond
	}
	intervals := gToIntervals[ev.G]
	if len(intervals) == 0 {
		return 0
	}

	var overlapping time.Duration
	for _, i := range intervals {
		if o := overlappingDuration(i.begin, i.end, ev.Ts, ev.Link.Ts); o > 0 {
			overlapping += o
		}
	}
	return overlapping
}

func recordsOf(records map[uint64]traceviewer.ProfileRecord) []traceviewer.ProfileRecord {
	result := make([]traceviewer.ProfileRecord, 0, len(records))
	for _, record := range records {
		result = append(result, record)
	}
	return result
}
