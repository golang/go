// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	tracev2 "internal/trace/v2"
	"sort"
	"strings"
	"time"
)

// Summary is the analysis result produced by the summarizer.
type Summary struct {
	Goroutines map[tracev2.GoID]*GoroutineSummary
	Tasks      map[tracev2.TaskID]*UserTaskSummary
}

// GoroutineSummary contains statistics and execution details of a single goroutine.
// (For v2 traces.)
type GoroutineSummary struct {
	ID           tracev2.GoID
	Name         string       // A non-unique human-friendly identifier for the goroutine.
	PC           uint64       // The first PC we saw for the entry function of the goroutine
	CreationTime tracev2.Time // Timestamp of the first appearance in the trace.
	StartTime    tracev2.Time // Timestamp of the first time it started running. 0 if the goroutine never ran.
	EndTime      tracev2.Time // Timestamp of when the goroutine exited. 0 if the goroutine never exited.

	// List of regions in the goroutine, sorted based on the start time.
	Regions []*UserRegionSummary

	// Statistics of execution time during the goroutine execution.
	GoroutineExecStats

	// goroutineSummary is state used just for computing this structure.
	// It's dropped before being returned to the caller.
	//
	// More specifically, if it's nil, it indicates that this summary has
	// already been finalized.
	*goroutineSummary
}

// UserTaskSummary represents a task in the trace.
type UserTaskSummary struct {
	ID       tracev2.TaskID
	Name     string
	Parent   *UserTaskSummary // nil if the parent is unknown.
	Children []*UserTaskSummary

	// Task begin event. An EventTaskBegin event or nil.
	Start *tracev2.Event

	// End end event. Normally EventTaskEnd event or nil.
	End *tracev2.Event

	// Logs is a list of tracev2.EventLog events associated with the task.
	Logs []*tracev2.Event

	// List of regions in the task, sorted based on the start time.
	Regions []*UserRegionSummary

	// Goroutines is the set of goroutines associated with this task.
	Goroutines map[tracev2.GoID]*GoroutineSummary
}

// Complete returns true if we have complete information about the task
// from the trace: both a start and an end.
func (s *UserTaskSummary) Complete() bool {
	return s.Start != nil && s.End != nil
}

// Descendents returns a slice consisting of itself (always the first task returned),
// and the transitive closure of all of its children.
func (s *UserTaskSummary) Descendents() []*UserTaskSummary {
	descendents := []*UserTaskSummary{s}
	for _, child := range s.Children {
		descendents = append(descendents, child.Descendents()...)
	}
	return descendents
}

// UserRegionSummary represents a region and goroutine execution stats
// while the region was active. (For v2 traces.)
type UserRegionSummary struct {
	TaskID tracev2.TaskID
	Name   string

	// Region start event. Normally EventRegionBegin event or nil,
	// but can be a state transition event from NotExist or Undetermined
	// if the region is a synthetic region representing task inheritance
	// from the parent goroutine.
	Start *tracev2.Event

	// Region end event. Normally EventRegionEnd event or nil,
	// but can be a state transition event to NotExist if the goroutine
	// terminated without explicitly ending the region.
	End *tracev2.Event

	GoroutineExecStats
}

// GoroutineExecStats contains statistics about a goroutine's execution
// during a period of time.
type GoroutineExecStats struct {
	// These stats are all non-overlapping.
	ExecTime          time.Duration
	SchedWaitTime     time.Duration
	BlockTimeByReason map[string]time.Duration
	SyscallTime       time.Duration
	SyscallBlockTime  time.Duration

	// TotalTime is the duration of the goroutine's presence in the trace.
	// Necessarily overlaps with other stats.
	TotalTime time.Duration

	// Total time the goroutine spent in certain ranges; may overlap
	// with other stats.
	RangeTime map[string]time.Duration
}

func (s GoroutineExecStats) NonOverlappingStats() map[string]time.Duration {
	stats := map[string]time.Duration{
		"Execution time":         s.ExecTime,
		"Sched wait time":        s.SchedWaitTime,
		"Syscall execution time": s.SyscallTime,
		"Block time (syscall)":   s.SyscallBlockTime,
		"Unknown time":           s.UnknownTime(),
	}
	for reason, dt := range s.BlockTimeByReason {
		stats["Block time ("+reason+")"] += dt
	}
	// N.B. Don't include RangeTime or TotalTime; they overlap with these other
	// stats.
	return stats
}

// UnknownTime returns whatever isn't accounted for in TotalTime.
func (s GoroutineExecStats) UnknownTime() time.Duration {
	sum := s.ExecTime + s.SchedWaitTime + s.SyscallTime +
		s.SyscallBlockTime
	for _, dt := range s.BlockTimeByReason {
		sum += dt
	}
	// N.B. Don't include range time. Ranges overlap with
	// other stats, whereas these stats are non-overlapping.
	if sum < s.TotalTime {
		return s.TotalTime - sum
	}
	return 0
}

// sub returns the stats v-s.
func (s GoroutineExecStats) sub(v GoroutineExecStats) (r GoroutineExecStats) {
	r = s.clone()
	r.ExecTime -= v.ExecTime
	r.SchedWaitTime -= v.SchedWaitTime
	for reason := range s.BlockTimeByReason {
		r.BlockTimeByReason[reason] -= v.BlockTimeByReason[reason]
	}
	r.SyscallTime -= v.SyscallTime
	r.SyscallBlockTime -= v.SyscallBlockTime
	r.TotalTime -= v.TotalTime
	for name := range s.RangeTime {
		r.RangeTime[name] -= v.RangeTime[name]
	}
	return r
}

func (s GoroutineExecStats) clone() (r GoroutineExecStats) {
	r = s
	r.BlockTimeByReason = make(map[string]time.Duration)
	for reason, dt := range s.BlockTimeByReason {
		r.BlockTimeByReason[reason] = dt
	}
	r.RangeTime = make(map[string]time.Duration)
	for name, dt := range s.RangeTime {
		r.RangeTime[name] = dt
	}
	return r
}

// snapshotStat returns the snapshot of the goroutine execution statistics.
// This is called as we process the ordered trace event stream. lastTs is used
// to process pending statistics if this is called before any goroutine end event.
func (g *GoroutineSummary) snapshotStat(lastTs tracev2.Time) (ret GoroutineExecStats) {
	ret = g.GoroutineExecStats.clone()

	if g.goroutineSummary == nil {
		return ret // Already finalized; no pending state.
	}

	// Set the total time if necessary.
	if g.TotalTime == 0 {
		ret.TotalTime = lastTs.Sub(g.CreationTime)
	}

	// Add in time since lastTs.
	if g.lastStartTime != 0 {
		ret.ExecTime += lastTs.Sub(g.lastStartTime)
	}
	if g.lastRunnableTime != 0 {
		ret.SchedWaitTime += lastTs.Sub(g.lastRunnableTime)
	}
	if g.lastBlockTime != 0 {
		ret.BlockTimeByReason[g.lastBlockReason] += lastTs.Sub(g.lastBlockTime)
	}
	if g.lastSyscallTime != 0 {
		ret.SyscallTime += lastTs.Sub(g.lastSyscallTime)
	}
	if g.lastSyscallBlockTime != 0 {
		ret.SchedWaitTime += lastTs.Sub(g.lastSyscallBlockTime)
	}
	for name, ts := range g.lastRangeTime {
		ret.RangeTime[name] += lastTs.Sub(ts)
	}
	return ret
}

// finalize is called when processing a goroutine end event or at
// the end of trace processing. This finalizes the execution stat
// and any active regions in the goroutine, in which case trigger is nil.
func (g *GoroutineSummary) finalize(lastTs tracev2.Time, trigger *tracev2.Event) {
	if trigger != nil {
		g.EndTime = trigger.Time()
	}
	finalStat := g.snapshotStat(lastTs)

	g.GoroutineExecStats = finalStat

	// System goroutines are never part of regions, even though they
	// "inherit" a task due to creation (EvGoCreate) from within a region.
	// This may happen e.g. if the first GC is triggered within a region,
	// starting the GC worker goroutines.
	if !IsSystemGoroutine(g.Name) {
		for _, s := range g.activeRegions {
			s.End = trigger
			s.GoroutineExecStats = finalStat.sub(s.GoroutineExecStats)
			g.Regions = append(g.Regions, s)
		}
	}
	*(g.goroutineSummary) = goroutineSummary{}
}

// goroutineSummary is a private part of GoroutineSummary that is required only during analysis.
type goroutineSummary struct {
	lastStartTime        tracev2.Time
	lastRunnableTime     tracev2.Time
	lastBlockTime        tracev2.Time
	lastBlockReason      string
	lastSyscallTime      tracev2.Time
	lastSyscallBlockTime tracev2.Time
	lastRangeTime        map[string]tracev2.Time
	activeRegions        []*UserRegionSummary // stack of active regions
}

// Summarizer constructs per-goroutine time statistics for v2 traces.
type Summarizer struct {
	// gs contains the map of goroutine summaries we're building up to return to the caller.
	gs map[tracev2.GoID]*GoroutineSummary

	// tasks contains the map of task summaries we're building up to return to the caller.
	tasks map[tracev2.TaskID]*UserTaskSummary

	// syscallingP and syscallingG represent a binding between a P and G in a syscall.
	// Used to correctly identify and clean up after syscalls (blocking or otherwise).
	syscallingP map[tracev2.ProcID]tracev2.GoID
	syscallingG map[tracev2.GoID]tracev2.ProcID

	// rangesP is used for optimistic tracking of P-based ranges for goroutines.
	//
	// It's a best-effort mapping of an active range on a P to the goroutine we think
	// is associated with it.
	rangesP map[rangeP]tracev2.GoID

	lastTs tracev2.Time // timestamp of the last event processed.
	syncTs tracev2.Time // timestamp of the last sync event processed (or the first timestamp in the trace).
}

// NewSummarizer creates a new struct to build goroutine stats from a trace.
func NewSummarizer() *Summarizer {
	return &Summarizer{
		gs:          make(map[tracev2.GoID]*GoroutineSummary),
		tasks:       make(map[tracev2.TaskID]*UserTaskSummary),
		syscallingP: make(map[tracev2.ProcID]tracev2.GoID),
		syscallingG: make(map[tracev2.GoID]tracev2.ProcID),
		rangesP:     make(map[rangeP]tracev2.GoID),
	}
}

type rangeP struct {
	id   tracev2.ProcID
	name string
}

// Event feeds a single event into the stats summarizer.
func (s *Summarizer) Event(ev *tracev2.Event) {
	if s.syncTs == 0 {
		s.syncTs = ev.Time()
	}
	s.lastTs = ev.Time()

	switch ev.Kind() {
	// Record sync time for the RangeActive events.
	case tracev2.EventSync:
		s.syncTs = ev.Time()

	// Handle state transitions.
	case tracev2.EventStateTransition:
		st := ev.StateTransition()
		switch st.Resource.Kind {
		// Handle goroutine transitions, which are the meat of this computation.
		case tracev2.ResourceGoroutine:
			id := st.Resource.Goroutine()
			old, new := st.Goroutine()
			if old == new {
				// Skip these events; they're not telling us anything new.
				break
			}

			// Handle transition out.
			g := s.gs[id]
			switch old {
			case tracev2.GoUndetermined, tracev2.GoNotExist:
				g = &GoroutineSummary{ID: id, goroutineSummary: &goroutineSummary{}}
				// If we're coming out of GoUndetermined, then the creation time is the
				// time of the last sync.
				if old == tracev2.GoUndetermined {
					g.CreationTime = s.syncTs
				} else {
					g.CreationTime = ev.Time()
				}
				// The goroutine is being created, or it's being named for the first time.
				g.lastRangeTime = make(map[string]tracev2.Time)
				g.BlockTimeByReason = make(map[string]time.Duration)
				g.RangeTime = make(map[string]time.Duration)

				// When a goroutine is newly created, inherit the task
				// of the active region. For ease handling of this
				// case, we create a fake region description with the
				// task id. This isn't strictly necessary as this
				// goroutine may not be associated with the task, but
				// it can be convenient to see all children created
				// during a region.
				//
				// N.B. ev.Goroutine() will always be NoGoroutine for the
				// Undetermined case, so this is will simply not fire.
				if creatorG := s.gs[ev.Goroutine()]; creatorG != nil && len(creatorG.activeRegions) > 0 {
					regions := creatorG.activeRegions
					s := regions[len(regions)-1]
					g.activeRegions = []*UserRegionSummary{{TaskID: s.TaskID, Start: ev}}
				}
				s.gs[g.ID] = g
			case tracev2.GoRunning:
				// Record execution time as we transition out of running
				g.ExecTime += ev.Time().Sub(g.lastStartTime)
				g.lastStartTime = 0
			case tracev2.GoWaiting:
				// Record block time as we transition out of waiting.
				if g.lastBlockTime != 0 {
					g.BlockTimeByReason[g.lastBlockReason] += ev.Time().Sub(g.lastBlockTime)
					g.lastBlockTime = 0
				}
			case tracev2.GoRunnable:
				// Record sched latency time as we transition out of runnable.
				if g.lastRunnableTime != 0 {
					g.SchedWaitTime += ev.Time().Sub(g.lastRunnableTime)
					g.lastRunnableTime = 0
				}
			case tracev2.GoSyscall:
				// Record syscall execution time and syscall block time as we transition out of syscall.
				if g.lastSyscallTime != 0 {
					if g.lastSyscallBlockTime != 0 {
						g.SyscallBlockTime += ev.Time().Sub(g.lastSyscallBlockTime)
						g.SyscallTime += g.lastSyscallBlockTime.Sub(g.lastSyscallTime)
					} else {
						g.SyscallTime += ev.Time().Sub(g.lastSyscallTime)
					}
					g.lastSyscallTime = 0
					g.lastSyscallBlockTime = 0

					// Clear the syscall map.
					delete(s.syscallingP, s.syscallingG[id])
					delete(s.syscallingG, id)
				}
			}

			// The goroutine hasn't been identified yet. Take the transition stack
			// and identify the goroutine by the root frame of that stack.
			// This root frame will be identical for all transitions on this
			// goroutine, because it represents its immutable start point.
			if g.Name == "" {
				stk := st.Stack
				if stk != tracev2.NoStack {
					var frame tracev2.StackFrame
					var ok bool
					stk.Frames(func(f tracev2.StackFrame) bool {
						frame = f
						ok = true
						return true
					})
					if ok {
						// NB: this PC won't actually be consistent for
						// goroutines which existed at the start of the
						// trace. The UI doesn't use it directly; this
						// mainly serves as an indication that we
						// actually saw a call stack for the goroutine
						g.PC = frame.PC
						g.Name = frame.Func
					}
				}
			}

			// Handle transition in.
			switch new {
			case tracev2.GoRunning:
				// We started running. Record it.
				g.lastStartTime = ev.Time()
				if g.StartTime == 0 {
					g.StartTime = ev.Time()
				}
			case tracev2.GoRunnable:
				g.lastRunnableTime = ev.Time()
			case tracev2.GoWaiting:
				if st.Reason != "forever" {
					g.lastBlockTime = ev.Time()
					g.lastBlockReason = st.Reason
					break
				}
				// "Forever" is like goroutine death.
				fallthrough
			case tracev2.GoNotExist:
				g.finalize(ev.Time(), ev)
			case tracev2.GoSyscall:
				s.syscallingP[ev.Proc()] = id
				s.syscallingG[id] = ev.Proc()
				g.lastSyscallTime = ev.Time()
			}

		// Handle procs to detect syscall blocking, which si identifiable as a
		// proc going idle while the goroutine it was attached to is in a syscall.
		case tracev2.ResourceProc:
			id := st.Resource.Proc()
			old, new := st.Proc()
			if old != new && new == tracev2.ProcIdle {
				if goid, ok := s.syscallingP[id]; ok {
					g := s.gs[goid]
					g.lastSyscallBlockTime = ev.Time()
					delete(s.syscallingP, id)
				}
			}
		}

	// Handle ranges of all kinds.
	case tracev2.EventRangeBegin, tracev2.EventRangeActive:
		r := ev.Range()
		var g *GoroutineSummary
		switch r.Scope.Kind {
		case tracev2.ResourceGoroutine:
			// Simple goroutine range. We attribute the entire range regardless of
			// goroutine stats. Lots of situations are still identifiable, e.g. a
			// goroutine blocked often in mark assist will have both high mark assist
			// and high block times. Those interested in a deeper view can look at the
			// trace viewer.
			g = s.gs[r.Scope.Goroutine()]
		case tracev2.ResourceProc:
			// N.B. These ranges are not actually bound to the goroutine, they're
			// bound to the P. But if we happen to be on the P the whole time, let's
			// try to attribute it to the goroutine. (e.g. GC sweeps are here.)
			g = s.gs[ev.Goroutine()]
			if g != nil {
				s.rangesP[rangeP{id: r.Scope.Proc(), name: r.Name}] = ev.Goroutine()
			}
		}
		if g == nil {
			break
		}
		if ev.Kind() == tracev2.EventRangeActive {
			if ts := g.lastRangeTime[r.Name]; ts != 0 {
				g.RangeTime[r.Name] += s.syncTs.Sub(ts)
			}
			g.lastRangeTime[r.Name] = s.syncTs
		} else {
			g.lastRangeTime[r.Name] = ev.Time()
		}
	case tracev2.EventRangeEnd:
		r := ev.Range()
		var g *GoroutineSummary
		switch r.Scope.Kind {
		case tracev2.ResourceGoroutine:
			g = s.gs[r.Scope.Goroutine()]
		case tracev2.ResourceProc:
			rp := rangeP{id: r.Scope.Proc(), name: r.Name}
			if goid, ok := s.rangesP[rp]; ok {
				if goid == ev.Goroutine() {
					// As the comment in the RangeBegin case states, this is only OK
					// if we finish on the same goroutine we started on.
					g = s.gs[goid]
				}
				delete(s.rangesP, rp)
			}
		}
		if g == nil {
			break
		}
		ts := g.lastRangeTime[r.Name]
		if ts == 0 {
			break
		}
		g.RangeTime[r.Name] += ev.Time().Sub(ts)
		delete(g.lastRangeTime, r.Name)

	// Handle user-defined regions.
	case tracev2.EventRegionBegin:
		g := s.gs[ev.Goroutine()]
		r := ev.Region()
		region := &UserRegionSummary{
			Name:               r.Type,
			TaskID:             r.Task,
			Start:              ev,
			GoroutineExecStats: g.snapshotStat(ev.Time()),
		}
		g.activeRegions = append(g.activeRegions, region)
		// Associate the region and current goroutine to the task.
		task := s.getOrAddTask(r.Task)
		task.Regions = append(task.Regions, region)
		task.Goroutines[g.ID] = g
	case tracev2.EventRegionEnd:
		g := s.gs[ev.Goroutine()]
		r := ev.Region()
		var sd *UserRegionSummary
		if regionStk := g.activeRegions; len(regionStk) > 0 {
			// Pop the top region from the stack since that's what must have ended.
			n := len(regionStk)
			sd = regionStk[n-1]
			regionStk = regionStk[:n-1]
			g.activeRegions = regionStk
			// N.B. No need to add the region to a task; the EventRegionBegin already handled it.
		} else {
			// This is an "end" without a start. Just fabricate the region now.
			sd = &UserRegionSummary{Name: r.Type, TaskID: r.Task}
			// Associate the region and current goroutine to the task.
			task := s.getOrAddTask(r.Task)
			task.Goroutines[g.ID] = g
			task.Regions = append(task.Regions, sd)
		}
		sd.GoroutineExecStats = g.snapshotStat(ev.Time()).sub(sd.GoroutineExecStats)
		sd.End = ev
		g.Regions = append(g.Regions, sd)

	// Handle tasks and logs.
	case tracev2.EventTaskBegin, tracev2.EventTaskEnd:
		// Initialize the task.
		t := ev.Task()
		task := s.getOrAddTask(t.ID)
		task.Name = t.Type
		task.Goroutines[ev.Goroutine()] = s.gs[ev.Goroutine()]
		if ev.Kind() == tracev2.EventTaskBegin {
			task.Start = ev
		} else {
			task.End = ev
		}
		// Initialize the parent, if one exists and it hasn't been done yet.
		// We need to avoid doing it twice, otherwise we could appear twice
		// in the parent's Children list.
		if t.Parent != tracev2.NoTask && task.Parent == nil {
			parent := s.getOrAddTask(t.Parent)
			task.Parent = parent
			parent.Children = append(parent.Children, task)
		}
	case tracev2.EventLog:
		log := ev.Log()
		// Just add the log to the task. We'll create the task if it
		// doesn't exist (it's just been mentioned now).
		task := s.getOrAddTask(log.Task)
		task.Goroutines[ev.Goroutine()] = s.gs[ev.Goroutine()]
		task.Logs = append(task.Logs, ev)
	}
}

func (s *Summarizer) getOrAddTask(id tracev2.TaskID) *UserTaskSummary {
	task := s.tasks[id]
	if task == nil {
		task = &UserTaskSummary{ID: id, Goroutines: make(map[tracev2.GoID]*GoroutineSummary)}
		s.tasks[id] = task
	}
	return task
}

// Finalize indicates to the summarizer that we're done processing the trace.
// It cleans up any remaining state and returns the full summary.
func (s *Summarizer) Finalize() *Summary {
	for _, g := range s.gs {
		g.finalize(s.lastTs, nil)

		// Sort based on region start time.
		sort.Slice(g.Regions, func(i, j int) bool {
			x := g.Regions[i].Start
			y := g.Regions[j].Start
			if x == nil {
				return true
			}
			if y == nil {
				return false
			}
			return x.Time() < y.Time()
		})
		g.goroutineSummary = nil
	}
	return &Summary{
		Goroutines: s.gs,
		Tasks:      s.tasks,
	}
}

// RelatedGoroutinesV2 finds a set of goroutines related to goroutine goid for v2 traces.
// The association is based on whether they have synchronized with each other in the Go
// scheduler (one has unblocked another).
func RelatedGoroutinesV2(events []tracev2.Event, goid tracev2.GoID) map[tracev2.GoID]struct{} {
	// Process all the events, looking for transitions of goroutines
	// out of GoWaiting. If there was an active goroutine when this
	// happened, then we know that active goroutine unblocked another.
	// Scribble all these down so we can process them.
	type unblockEdge struct {
		operator tracev2.GoID
		operand  tracev2.GoID
	}
	var unblockEdges []unblockEdge
	for _, ev := range events {
		if ev.Goroutine() == tracev2.NoGoroutine {
			continue
		}
		if ev.Kind() != tracev2.EventStateTransition {
			continue
		}
		st := ev.StateTransition()
		if st.Resource.Kind != tracev2.ResourceGoroutine {
			continue
		}
		id := st.Resource.Goroutine()
		old, new := st.Goroutine()
		if old == new || old != tracev2.GoWaiting {
			continue
		}
		unblockEdges = append(unblockEdges, unblockEdge{
			operator: ev.Goroutine(),
			operand:  id,
		})
	}
	// Compute the transitive closure of depth 2 of goroutines that have unblocked each other
	// (starting from goid).
	gmap := make(map[tracev2.GoID]struct{})
	gmap[goid] = struct{}{}
	for i := 0; i < 2; i++ {
		// Copy the map.
		gmap1 := make(map[tracev2.GoID]struct{})
		for g := range gmap {
			gmap1[g] = struct{}{}
		}
		for _, edge := range unblockEdges {
			if _, ok := gmap[edge.operand]; ok {
				gmap1[edge.operator] = struct{}{}
			}
		}
		gmap = gmap1
	}
	return gmap
}

func IsSystemGoroutine(entryFn string) bool {
	// This mimics runtime.isSystemGoroutine as closely as
	// possible.
	// Also, locked g in extra M (with empty entryFn) is system goroutine.
	return entryFn == "" || entryFn != "runtime.main" && strings.HasPrefix(entryFn, "runtime.")
}
