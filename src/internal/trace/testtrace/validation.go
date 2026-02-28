// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testtrace

import (
	"errors"
	"fmt"
	"internal/trace"
	"internal/trace/version"
	"slices"
	"strings"
)

// Validator is a type used for validating a stream of trace.Events.
type Validator struct {
	lastTs    trace.Time
	gs        map[trace.GoID]*goState
	ps        map[trace.ProcID]*procState
	ms        map[trace.ThreadID]*schedContext
	ranges    map[trace.ResourceID][]string
	tasks     map[trace.TaskID]string
	lastSync  trace.Sync
	GoVersion version.Version

	// Flags to modify validation behavior.
	skipClockSnapshotChecks bool // Some platforms can't guarantee a monotonically increasing clock reading.
}

type schedContext struct {
	M trace.ThreadID
	P trace.ProcID
	G trace.GoID
}

type goState struct {
	state   trace.GoState
	binding *schedContext
}

type procState struct {
	state   trace.ProcState
	binding *schedContext
}

// NewValidator creates a new Validator.
func NewValidator() *Validator {
	return &Validator{
		gs:        make(map[trace.GoID]*goState),
		ps:        make(map[trace.ProcID]*procState),
		ms:        make(map[trace.ThreadID]*schedContext),
		ranges:    make(map[trace.ResourceID][]string),
		tasks:     make(map[trace.TaskID]string),
		GoVersion: version.Current,
	}
}

// SkipClockSnapshotChecks causes the validator to skip checks on the clock snapshots.
//
// Some platforms like Windows, with a small enough trace period, are unable to produce
// monotonically increasing timestamps due to very coarse clock granularity.
func (v *Validator) SkipClockSnapshotChecks() {
	v.skipClockSnapshotChecks = true
}

// Event validates ev as the next event in a stream of trace.Events.
//
// Returns an error if validation fails.
func (v *Validator) Event(ev trace.Event) error {
	e := new(errAccumulator)

	// Validate timestamp order.
	if v.lastTs != 0 {
		if ev.Time() <= v.lastTs {
			e.Errorf("timestamp out-of-order (want > %v) for %+v", v.lastTs, ev)
		} else {
			v.lastTs = ev.Time()
		}
	} else {
		v.lastTs = ev.Time()
	}

	// Validate event stack.
	checkStack(e, ev.Stack())

	switch ev.Kind() {
	case trace.EventSync:
		s := ev.Sync()
		if s.N != v.lastSync.N+1 {
			e.Errorf("sync count is not sequential: expected %d, got %d", v.lastSync.N+1, s.N)
		}
		// The trace reader currently emits synthetic sync events at the end of
		// a trace. Those don't contain clock snapshots data, so we don't try
		// to validate them.
		//
		// TODO(felixge): Drop the synthetic syncs as discussed in CL 653576.
		if v.GoVersion >= version.Go125 && !(s.N > 1 && s.ClockSnapshot == nil) {
			if s.ClockSnapshot == nil {
				e.Errorf("sync %d has no clock snapshot", s.N)
			} else {
				if s.ClockSnapshot.Wall.IsZero() {
					e.Errorf("sync %d has zero wall time", s.N)
				}
				if s.ClockSnapshot.Mono == 0 {
					e.Errorf("sync %d has zero mono time", s.N)
				}
				if s.ClockSnapshot.Trace == 0 {
					e.Errorf("sync %d has zero trace time", s.N)
				}
				if !v.skipClockSnapshotChecks {
					if s.N >= 2 && !s.ClockSnapshot.Wall.After(v.lastSync.ClockSnapshot.Wall) {
						e.Errorf("sync %d has non-increasing wall time: %v vs %v", s.N, s.ClockSnapshot.Wall, v.lastSync.ClockSnapshot.Wall)
					}
					if s.N >= 2 && !(s.ClockSnapshot.Mono > v.lastSync.ClockSnapshot.Mono) {
						e.Errorf("sync %d has non-increasing mono time: %v vs %v", s.N, s.ClockSnapshot.Mono, v.lastSync.ClockSnapshot.Mono)
					}
					if s.N >= 2 && !(s.ClockSnapshot.Trace > v.lastSync.ClockSnapshot.Trace) {
						e.Errorf("sync %d has non-increasing trace time: %v vs %v", s.N, s.ClockSnapshot.Trace, v.lastSync.ClockSnapshot.Trace)
					}
				}
			}
		}
		v.lastSync = s
	case trace.EventMetric:
		m := ev.Metric()
		if !strings.Contains(m.Name, ":") {
			// Should have a ":" as per runtime/metrics convention.
			e.Errorf("invalid metric name %q", m.Name)
		}
		// Make sure the value is OK.
		if m.Value.Kind() == trace.ValueBad {
			e.Errorf("invalid value")
		}
		switch m.Value.Kind() {
		case trace.ValueUint64:
			// Just make sure it doesn't panic.
			_ = m.Value.Uint64()
		}
	case trace.EventLabel:
		l := ev.Label()

		// Check label.
		if l.Label == "" {
			e.Errorf("invalid label %q", l.Label)
		}

		// Check label resource.
		if l.Resource.Kind == trace.ResourceNone {
			e.Errorf("label resource none")
		}
		switch l.Resource.Kind {
		case trace.ResourceGoroutine:
			id := l.Resource.Goroutine()
			if _, ok := v.gs[id]; !ok {
				e.Errorf("label for invalid goroutine %d", id)
			}
		case trace.ResourceProc:
			id := l.Resource.Proc()
			if _, ok := v.ps[id]; !ok {
				e.Errorf("label for invalid proc %d", id)
			}
		case trace.ResourceThread:
			id := l.Resource.Thread()
			if _, ok := v.ms[id]; !ok {
				e.Errorf("label for invalid thread %d", id)
			}
		}
	case trace.EventStackSample:
		// Not much to check here. It's basically a sched context and a stack.
		// The sched context is also not guaranteed to align with other events.
		// We already checked the stack above.
	case trace.EventStateTransition:
		// Validate state transitions.
		//
		// TODO(mknyszek): A lot of logic is duplicated between goroutines and procs.
		// The two are intentionally handled identically; from the perspective of the
		// API, resources all have the same general properties. Consider making this
		// code generic over resources and implementing validation just once.
		tr := ev.StateTransition()
		checkStack(e, tr.Stack)
		switch tr.Resource.Kind {
		case trace.ResourceGoroutine:
			// Basic state transition validation.
			id := tr.Resource.Goroutine()
			old, new := tr.Goroutine()
			if new == trace.GoUndetermined {
				e.Errorf("transition to undetermined state for goroutine %d", id)
			}
			if v.lastSync.N > 1 && old == trace.GoUndetermined {
				e.Errorf("undetermined goroutine %d after first global sync", id)
			}
			if new == trace.GoNotExist && v.hasAnyRange(trace.MakeResourceID(id)) {
				e.Errorf("goroutine %d died with active ranges", id)
			}
			state, ok := v.gs[id]
			if ok {
				if old != state.state {
					e.Errorf("bad old state for goroutine %d: got %s, want %s", id, old, state.state)
				}
				state.state = new
			} else {
				if old != trace.GoUndetermined && old != trace.GoNotExist {
					e.Errorf("bad old state for unregistered goroutine %d: %s", id, old)
				}
				state = &goState{state: new}
				v.gs[id] = state
			}
			// Validate sched context.
			if new.Executing() {
				ctx := v.getOrCreateThread(e, ev, ev.Thread())
				if ctx != nil {
					if ctx.G != trace.NoGoroutine && ctx.G != id {
						e.Errorf("tried to run goroutine %d when one was already executing (%d) on thread %d", id, ctx.G, ev.Thread())
					}
					ctx.G = id
					state.binding = ctx
				}
			} else if old.Executing() && !new.Executing() {
				if tr.Stack != ev.Stack() {
					// This is a case where the transition is happening to a goroutine that is also executing, so
					// these two stacks should always match.
					e.Errorf("StateTransition.Stack doesn't match Event.Stack")
				}
				ctx := state.binding
				if ctx != nil {
					if ctx.G != id {
						e.Errorf("tried to stop goroutine %d when it wasn't currently executing (currently executing %d) on thread %d", id, ctx.G, ev.Thread())
					}
					ctx.G = trace.NoGoroutine
					state.binding = nil
				} else {
					e.Errorf("stopping goroutine %d not bound to any active context", id)
				}
			}
		case trace.ResourceProc:
			// Basic state transition validation.
			id := tr.Resource.Proc()
			old, new := tr.Proc()
			if new == trace.ProcUndetermined {
				e.Errorf("transition to undetermined state for proc %d", id)
			}
			if v.lastSync.N > 1 && old == trace.ProcUndetermined {
				e.Errorf("undetermined proc %d after first global sync", id)
			}
			if new == trace.ProcNotExist && v.hasAnyRange(trace.MakeResourceID(id)) {
				e.Errorf("proc %d died with active ranges", id)
			}
			state, ok := v.ps[id]
			if ok {
				if old != state.state {
					e.Errorf("bad old state for proc %d: got %s, want %s", id, old, state.state)
				}
				state.state = new
			} else {
				if old != trace.ProcUndetermined && old != trace.ProcNotExist {
					e.Errorf("bad old state for unregistered proc %d: %s", id, old)
				}
				state = &procState{state: new}
				v.ps[id] = state
			}
			// Validate sched context.
			if new.Executing() {
				ctx := v.getOrCreateThread(e, ev, ev.Thread())
				if ctx != nil {
					if ctx.P != trace.NoProc && ctx.P != id {
						e.Errorf("tried to run proc %d when one was already executing (%d) on thread %d", id, ctx.P, ev.Thread())
					}
					ctx.P = id
					state.binding = ctx
				}
			} else if old.Executing() && !new.Executing() {
				ctx := state.binding
				if ctx != nil {
					if ctx.P != id {
						e.Errorf("tried to stop proc %d when it wasn't currently executing (currently executing %d) on thread %d", id, ctx.P, ctx.M)
					}
					ctx.P = trace.NoProc
					state.binding = nil
				} else {
					e.Errorf("stopping proc %d not bound to any active context", id)
				}
			}
		}
	case trace.EventRangeBegin, trace.EventRangeActive, trace.EventRangeEnd:
		// Validate ranges.
		r := ev.Range()
		switch ev.Kind() {
		case trace.EventRangeBegin:
			if v.hasRange(r.Scope, r.Name) {
				e.Errorf("already active range %q on %v begun again", r.Name, r.Scope)
			}
			v.addRange(r.Scope, r.Name)
		case trace.EventRangeActive:
			if !v.hasRange(r.Scope, r.Name) {
				v.addRange(r.Scope, r.Name)
			}
		case trace.EventRangeEnd:
			if !v.hasRange(r.Scope, r.Name) {
				e.Errorf("inactive range %q on %v ended", r.Name, r.Scope)
			}
			v.deleteRange(r.Scope, r.Name)
		}
	case trace.EventTaskBegin:
		// Validate task begin.
		t := ev.Task()
		if t.ID == trace.NoTask || t.ID == trace.BackgroundTask {
			// The background task should never have an event emitted for it.
			e.Errorf("found invalid task ID for task of type %s", t.Type)
		}
		if t.Parent == trace.BackgroundTask {
			// It's not possible for a task to be a subtask of the background task.
			e.Errorf("found background task as the parent for task of type %s", t.Type)
		}
		// N.B. Don't check the task type. Empty string is a valid task type.
		v.tasks[t.ID] = t.Type
	case trace.EventTaskEnd:
		// Validate task end.
		// We can see a task end without a begin, so ignore a task without information.
		// Instead, if we've seen the task begin, just make sure the task end lines up.
		t := ev.Task()
		if typ, ok := v.tasks[t.ID]; ok {
			if t.Type != typ {
				e.Errorf("task end type %q doesn't match task start type %q for task %d", t.Type, typ, t.ID)
			}
			delete(v.tasks, t.ID)
		}
	case trace.EventLog:
		// There's really not much here to check, except that we can
		// generate a Log. The category and message are entirely user-created,
		// so we can't make any assumptions as to what they are. We also
		// can't validate the task, because proving the task's existence is very
		// much best-effort.
		_ = ev.Log()
	}
	return e.Errors()
}

func (v *Validator) hasRange(r trace.ResourceID, name string) bool {
	ranges, ok := v.ranges[r]
	return ok && slices.Contains(ranges, name)
}

func (v *Validator) addRange(r trace.ResourceID, name string) {
	ranges, _ := v.ranges[r]
	ranges = append(ranges, name)
	v.ranges[r] = ranges
}

func (v *Validator) hasAnyRange(r trace.ResourceID) bool {
	ranges, ok := v.ranges[r]
	return ok && len(ranges) != 0
}

func (v *Validator) deleteRange(r trace.ResourceID, name string) {
	ranges, ok := v.ranges[r]
	if !ok {
		return
	}
	i := slices.Index(ranges, name)
	if i < 0 {
		return
	}
	v.ranges[r] = slices.Delete(ranges, i, i+1)
}

func (v *Validator) getOrCreateThread(e *errAccumulator, ev trace.Event, m trace.ThreadID) *schedContext {
	lenient := func() bool {
		// Be lenient about GoUndetermined -> GoSyscall transitions if they
		// originate from an old trace. These transitions lack thread
		// information in trace formats older than 1.22.
		if v.GoVersion >= version.Go122 {
			return false
		}
		if ev.Kind() != trace.EventStateTransition {
			return false
		}
		tr := ev.StateTransition()
		if tr.Resource.Kind != trace.ResourceGoroutine {
			return false
		}
		from, to := tr.Goroutine()
		return from == trace.GoUndetermined && to == trace.GoSyscall
	}
	if m == trace.NoThread && !lenient() {
		e.Errorf("must have thread, but thread ID is none")
		return nil
	}
	s, ok := v.ms[m]
	if !ok {
		s = &schedContext{M: m, P: trace.NoProc, G: trace.NoGoroutine}
		v.ms[m] = s
		return s
	}
	return s
}

func checkStack(e *errAccumulator, stk trace.Stack) {
	// Check for non-empty values, but we also check for crashes due to incorrect validation.
	for i, f := range slices.Collect(stk.Frames()) {
		if i == 0 {
			// Allow for one fully zero stack.
			//
			// TODO(mknyszek): Investigate why that happens.
			continue
		}
		if f.Func == "" || f.File == "" || f.PC == 0 || f.Line == 0 {
			e.Errorf("invalid stack frame %#v: missing information", f)
		}
	}
}

type errAccumulator struct {
	errs []error
}

func (e *errAccumulator) Errorf(f string, args ...any) {
	e.errs = append(e.errs, fmt.Errorf(f, args...))
}

func (e *errAccumulator) Errors() error {
	return errors.Join(e.errs...)
}
