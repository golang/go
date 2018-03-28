// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "sort"

// GDesc contains statistics and execution details of a single goroutine.
type GDesc struct {
	ID           uint64
	Name         string
	PC           uint64
	CreationTime int64
	StartTime    int64
	EndTime      int64

	// List of spans in the goroutine, sorted based on the start time.
	Spans []*UserSpanDesc

	// Statistics of execution time during the goroutine execution.
	GExecutionStat

	*gdesc // private part.
}

// UserSpanDesc represents a span and goroutine execution stats
// while the span was active.
type UserSpanDesc struct {
	TaskID uint64
	Name   string

	// Span start event. Normally EvUserSpan start event or nil,
	// but can be EvGoCreate event if the span is a synthetic
	// span representing task inheritance from the parent goroutine.
	Start *Event

	// Span end event. Normally EvUserSpan end event or nil,
	// but can be EvGoStop or EvGoEnd event if the goroutine
	// terminated without explicitely ending the span.
	End *Event

	GExecutionStat
}

// GExecutionStat contains statistics about a goroutine's execution
// during a period of time.
type GExecutionStat struct {
	ExecTime      int64
	SchedWaitTime int64
	IOTime        int64
	BlockTime     int64
	SyscallTime   int64
	GCTime        int64
	SweepTime     int64
	TotalTime     int64
}

// sub returns the stats v-s.
func (s GExecutionStat) sub(v GExecutionStat) (r GExecutionStat) {
	r = s
	r.ExecTime -= v.ExecTime
	r.SchedWaitTime -= v.SchedWaitTime
	r.IOTime -= v.IOTime
	r.BlockTime -= v.BlockTime
	r.SyscallTime -= v.SyscallTime
	r.GCTime -= v.GCTime
	r.SweepTime -= v.SweepTime
	r.TotalTime -= v.TotalTime
	return r
}

// snapshotStat returns the snapshot of the goroutine execution statistics.
// This is called as we process the ordered trace event stream. lastTs and
// activeGCStartTime are used to process pending statistics if this is called
// before any goroutine end event.
func (g *GDesc) snapshotStat(lastTs, activeGCStartTime int64) (ret GExecutionStat) {
	ret = g.GExecutionStat

	if g.gdesc == nil {
		return ret // finalized GDesc. No pending state.
	}

	if activeGCStartTime != 0 {
		ret.GCTime += lastTs - activeGCStartTime
	}

	if g.TotalTime == 0 {
		ret.TotalTime = lastTs - g.CreationTime
	}

	if g.lastStartTime != 0 {
		ret.ExecTime += lastTs - g.lastStartTime
	}
	if g.blockNetTime != 0 {
		ret.IOTime += lastTs - g.blockNetTime
	}
	if g.blockSyncTime != 0 {
		ret.BlockTime += lastTs - g.blockSyncTime
	}
	if g.blockSyscallTime != 0 {
		ret.SyscallTime += lastTs - g.blockSyscallTime
	}
	if g.blockSchedTime != 0 {
		ret.SchedWaitTime += lastTs - g.blockSchedTime
	}
	if g.blockSweepTime != 0 {
		ret.SweepTime += lastTs - g.blockSweepTime
	}
	return ret
}

// finalizeActiveSpans is called when processing a goroutine end event
// to finalize any active spans in the goroutine.
func (g *GDesc) finalizeActiveSpans(lastTs, activeGCStartTime int64, trigger *Event) {
	for _, s := range g.activeSpans {
		s.End = trigger
		s.GExecutionStat = g.snapshotStat(lastTs, activeGCStartTime).sub(s.GExecutionStat)
		g.Spans = append(g.Spans, s)
	}
	g.activeSpans = nil
}

// gdesc is a private part of GDesc that is required only during analysis.
type gdesc struct {
	lastStartTime    int64
	blockNetTime     int64
	blockSyncTime    int64
	blockSyscallTime int64
	blockSweepTime   int64
	blockGCTime      int64
	blockSchedTime   int64

	activeSpans []*UserSpanDesc // stack of active spans
}

// GoroutineStats generates statistics for all goroutines in the trace.
func GoroutineStats(events []*Event) map[uint64]*GDesc {
	gs := make(map[uint64]*GDesc)
	var lastTs int64
	var gcStartTime int64 // gcStartTime == 0 indicates gc is inactive.
	for _, ev := range events {
		lastTs = ev.Ts
		switch ev.Type {
		case EvGoCreate:
			g := &GDesc{ID: ev.Args[0], CreationTime: ev.Ts, gdesc: new(gdesc)}
			g.blockSchedTime = ev.Ts
			// When a goroutine is newly created, inherit the
			// task of the active span. For ease handling of
			// this case, we create a fake span description with
			// the task id.
			if creatorG := gs[ev.G]; creatorG != nil && len(creatorG.gdesc.activeSpans) > 0 {
				spans := creatorG.gdesc.activeSpans
				s := spans[len(spans)-1]
				if s.TaskID != 0 {
					g.gdesc.activeSpans = []*UserSpanDesc{
						{TaskID: s.TaskID, Start: ev},
					}
				}
			}
			gs[g.ID] = g
		case EvGoStart, EvGoStartLabel:
			g := gs[ev.G]
			if g.PC == 0 {
				g.PC = ev.Stk[0].PC
				g.Name = ev.Stk[0].Fn
			}
			g.lastStartTime = ev.Ts
			if g.StartTime == 0 {
				g.StartTime = ev.Ts
			}
			if g.blockSchedTime != 0 {
				g.SchedWaitTime += ev.Ts - g.blockSchedTime
				g.blockSchedTime = 0
			}
		case EvGoEnd, EvGoStop:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.TotalTime = ev.Ts - g.CreationTime
			g.EndTime = ev.Ts
			if gcStartTime != 0 { // terminating while GC is active
				if g.CreationTime < gcStartTime {
					g.GCTime += ev.Ts - gcStartTime
				} else {
					// The goroutine's lifetime overlaps
					// with a GC completely.
					g.GCTime += ev.Ts - g.CreationTime
				}
			}
			g.finalizeActiveSpans(lastTs, gcStartTime, ev)
		case EvGoBlockSend, EvGoBlockRecv, EvGoBlockSelect,
			EvGoBlockSync, EvGoBlockCond:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.blockSyncTime = ev.Ts
		case EvGoSched, EvGoPreempt:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.blockSchedTime = ev.Ts
		case EvGoSleep, EvGoBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
		case EvGoBlockNet:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.blockNetTime = ev.Ts
		case EvGoBlockGC:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.blockGCTime = ev.Ts
		case EvGoUnblock:
			g := gs[ev.Args[0]]
			if g.blockNetTime != 0 {
				g.IOTime += ev.Ts - g.blockNetTime
				g.blockNetTime = 0
			}
			if g.blockSyncTime != 0 {
				g.BlockTime += ev.Ts - g.blockSyncTime
				g.blockSyncTime = 0
			}
			g.blockSchedTime = ev.Ts
		case EvGoSysBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.lastStartTime = 0
			g.blockSyscallTime = ev.Ts
		case EvGoSysExit:
			g := gs[ev.G]
			if g.blockSyscallTime != 0 {
				g.SyscallTime += ev.Ts - g.blockSyscallTime
				g.blockSyscallTime = 0
			}
			g.blockSchedTime = ev.Ts
		case EvGCSweepStart:
			g := gs[ev.G]
			if g != nil {
				// Sweep can happen during GC on system goroutine.
				g.blockSweepTime = ev.Ts
			}
		case EvGCSweepDone:
			g := gs[ev.G]
			if g != nil && g.blockSweepTime != 0 {
				g.SweepTime += ev.Ts - g.blockSweepTime
				g.blockSweepTime = 0
			}
		case EvGCStart:
			gcStartTime = ev.Ts
		case EvGCDone:
			for _, g := range gs {
				if g.EndTime != 0 {
					continue
				}
				if gcStartTime < g.CreationTime {
					g.GCTime += ev.Ts - g.CreationTime
				} else {
					g.GCTime += ev.Ts - gcStartTime
				}
			}
			gcStartTime = 0 // indicates gc is inactive.
		case EvUserSpan:
			g := gs[ev.G]
			switch mode := ev.Args[1]; mode {
			case 0: // span start
				g.activeSpans = append(g.activeSpans, &UserSpanDesc{
					Name:           ev.SArgs[0],
					TaskID:         ev.Args[0],
					Start:          ev,
					GExecutionStat: g.snapshotStat(lastTs, gcStartTime),
				})
			case 1: // span end
				var sd *UserSpanDesc
				if spanStk := g.activeSpans; len(spanStk) > 0 {
					n := len(spanStk)
					sd = spanStk[n-1]
					spanStk = spanStk[:n-1] // pop
					g.activeSpans = spanStk
				} else {
					sd = &UserSpanDesc{
						Name:   ev.SArgs[0],
						TaskID: ev.Args[0],
					}
				}
				sd.GExecutionStat = g.snapshotStat(lastTs, gcStartTime).sub(sd.GExecutionStat)
				sd.End = ev
				g.Spans = append(g.Spans, sd)
			}
		}
	}

	for _, g := range gs {
		g.GExecutionStat = g.snapshotStat(lastTs, gcStartTime)
		g.finalizeActiveSpans(lastTs, gcStartTime, nil)
		// sort based on span start time
		sort.Slice(g.Spans, func(i, j int) bool {
			x := g.Spans[i].Start
			y := g.Spans[j].Start
			if x == nil {
				return true
			}
			if y == nil {
				return false
			}
			return x.Ts < y.Ts
		})
		g.gdesc = nil
	}

	return gs
}

// RelatedGoroutines finds a set of goroutines related to goroutine goid.
func RelatedGoroutines(events []*Event, goid uint64) map[uint64]bool {
	// BFS of depth 2 over "unblock" edges
	// (what goroutines unblock goroutine goid?).
	gmap := make(map[uint64]bool)
	gmap[goid] = true
	for i := 0; i < 2; i++ {
		gmap1 := make(map[uint64]bool)
		for g := range gmap {
			gmap1[g] = true
		}
		for _, ev := range events {
			if ev.Type == EvGoUnblock && gmap[ev.Args[0]] {
				gmap1[ev.G] = true
			}
		}
		gmap = gmap1
	}
	gmap[0] = true // for GC events
	return gmap
}
