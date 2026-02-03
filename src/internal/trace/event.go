// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"internal/trace/tracev2"
	"internal/trace/version"
)

// EventKind indicates the kind of event this is.
//
// Use this information to obtain a more specific event that
// allows access to more detailed information.
type EventKind uint16

const (
	EventBad EventKind = iota

	// EventKindSync is an event that indicates a global synchronization
	// point in the trace. At the point of a sync event, the
	// trace reader can be certain that all resources (e.g. threads,
	// goroutines) that have existed until that point have been enumerated.
	EventSync

	// EventMetric is an event that represents the value of a metric at
	// a particular point in time.
	EventMetric

	// EventLabel attaches a label to a resource.
	EventLabel

	// EventStackSample represents an execution sample, indicating what a
	// thread/proc/goroutine was doing at a particular point in time via
	// its backtrace.
	//
	// Note: Samples should be considered a close approximation of
	// what a thread/proc/goroutine was executing at a given point in time.
	// These events may slightly contradict the situation StateTransitions
	// describe, so they should only be treated as a best-effort annotation.
	EventStackSample

	// EventRangeBegin and EventRangeEnd are a pair of generic events representing
	// a special range of time. Ranges are named and scoped to some resource
	// (identified via ResourceKind). A range that has begun but has not ended
	// is considered active.
	//
	// EvRangeBegin and EvRangeEnd will share the same name, and an End will always
	// follow a Begin on the same instance of the resource. The associated
	// resource ID can be obtained from the Event. ResourceNone indicates the
	// range is globally scoped. That is, any goroutine/proc/thread can start or
	// stop, but only one such range may be active at any given time.
	//
	// EventRangeActive is like EventRangeBegin, but indicates that the range was
	// already active. In this case, the resource referenced may not be in the current
	// context.
	EventRangeBegin
	EventRangeActive
	EventRangeEnd

	// EvTaskBegin and EvTaskEnd are a pair of events representing a runtime/trace.Task.
	EventTaskBegin
	EventTaskEnd

	// EventRegionBegin and EventRegionEnd are a pair of events represent a runtime/trace.Region.
	EventRegionBegin
	EventRegionEnd

	// EventLog represents a runtime/trace.Log call.
	EventLog

	// EventStateTransition represents a state change for some resource.
	EventStateTransition

	// EventExperimental is an experimental event that is unvalidated and exposed in a raw form.
	// Users are expected to understand the format and perform their own validation. These events
	// may always be safely ignored.
	EventExperimental
)

// String returns a string form of the EventKind.
func (e EventKind) String() string {
	if int(e) >= len(eventKindStrings) {
		return eventKindStrings[0]
	}
	return eventKindStrings[e]
}

var eventKindStrings = [...]string{
	EventBad:             "Bad",
	EventSync:            "Sync",
	EventMetric:          "Metric",
	EventLabel:           "Label",
	EventStackSample:     "StackSample",
	EventRangeBegin:      "RangeBegin",
	EventRangeActive:     "RangeActive",
	EventRangeEnd:        "RangeEnd",
	EventTaskBegin:       "TaskBegin",
	EventTaskEnd:         "TaskEnd",
	EventRegionBegin:     "RegionBegin",
	EventRegionEnd:       "RegionEnd",
	EventLog:             "Log",
	EventStateTransition: "StateTransition",
	EventExperimental:    "Experimental",
}

const maxTime = Time(math.MaxInt64)

// Time is a timestamp in nanoseconds.
//
// It corresponds to the monotonic clock on the platform that the
// trace was taken, and so is possible to correlate with timestamps
// for other traces taken on the same machine using the same clock
// (i.e. no reboots in between).
//
// The actual absolute value of the timestamp is only meaningful in
// relation to other timestamps from the same clock.
//
// BUG: Timestamps coming from traces on Windows platforms are
// only comparable with timestamps from the same trace. Timestamps
// across traces cannot be compared, because the system clock is
// not used as of Go 1.22.
//
// BUG: Traces produced by Go versions 1.21 and earlier cannot be
// compared with timestamps from other traces taken on the same
// machine. This is because the system clock was not used at all
// to collect those timestamps.
type Time int64

// Sub subtracts t0 from t, returning the duration in nanoseconds.
func (t Time) Sub(t0 Time) time.Duration {
	return time.Duration(int64(t) - int64(t0))
}

// Metric provides details about a Metric event.
type Metric struct {
	// Name is the name of the sampled metric.
	//
	// Names follow the same convention as metric names in the
	// runtime/metrics package, meaning they include the unit.
	// Names that match with the runtime/metrics package represent
	// the same quantity. Note that this corresponds to the
	// runtime/metrics package for the Go version this trace was
	// collected for.
	Name string

	// Value is the sampled value of the metric.
	//
	// The Value's Kind is tied to the name of the metric, and so is
	// guaranteed to be the same for metric samples for the same metric.
	Value Value
}

// Label provides details about a Label event.
type Label struct {
	// Label is the label applied to some resource.
	Label string

	// Resource is the resource to which this label should be applied.
	Resource ResourceID
}

// Range provides details about a Range event.
type Range struct {
	// Name is a human-readable name for the range.
	//
	// This name can be used to identify the end of the range for the resource
	// its scoped to, because only one of each type of range may be active on
	// a particular resource. The relevant resource should be obtained from the
	// Event that produced these details. The corresponding RangeEnd will have
	// an identical name.
	Name string

	// Scope is the resource that the range is scoped to.
	//
	// For example, a ResourceGoroutine scope means that the same goroutine
	// must have a start and end for the range, and that goroutine can only
	// have one range of a particular name active at any given time. The
	// ID that this range is scoped to may be obtained via Event.Goroutine.
	//
	// The ResourceNone scope means that the range is globally scoped. As a
	// result, any goroutine/proc/thread may start or end the range, and only
	// one such named range may be active globally at any given time.
	//
	// For RangeBegin and RangeEnd events, this will always reference some
	// resource ID in the current execution context. For RangeActive events,
	// this may reference a resource not in the current context. Prefer Scope
	// over the current execution context.
	Scope ResourceID
}

// RangeAttribute provides attributes about a completed Range.
type RangeAttribute struct {
	// Name is the human-readable name for the range.
	Name string

	// Value is the value of the attribute.
	Value Value
}

// TaskID is the internal ID of a task used to disambiguate tasks (even if they
// are of the same type).
type TaskID uint64

const (
	// NoTask indicates the lack of a task.
	NoTask = TaskID(^uint64(0))

	// BackgroundTask is the global task that events are attached to if there was
	// no other task in the context at the point the event was emitted.
	BackgroundTask = TaskID(0)
)

// Task provides details about a Task event.
type Task struct {
	// ID is a unique identifier for the task.
	//
	// This can be used to associate the beginning of a task with its end.
	ID TaskID

	// ParentID is the ID of the parent task.
	Parent TaskID

	// Type is the taskType that was passed to runtime/trace.NewTask.
	//
	// May be "" if a task's TaskBegin event isn't present in the trace.
	Type string
}

// Region provides details about a Region event.
type Region struct {
	// Task is the ID of the task this region is associated with.
	Task TaskID

	// Type is the regionType that was passed to runtime/trace.StartRegion or runtime/trace.WithRegion.
	Type string
}

// Log provides details about a Log event.
type Log struct {
	// Task is the ID of the task this region is associated with.
	Task TaskID

	// Category is the category that was passed to runtime/trace.Log or runtime/trace.Logf.
	Category string

	// Message is the message that was passed to runtime/trace.Log or runtime/trace.Logf.
	Message string
}

// StackSample is used to construct StackSample events via MakeEvent. There are
// no details associated with it, use EventConfig.Stack instead.
type StackSample struct{}

// MakeStack create a stack from a list of stack frames.
func MakeStack(frames []StackFrame) Stack {
	// TODO(felixge): support evTable reuse.
	tbl := &evTable{pcs: make(map[uint64]frame)}
	tbl.strings.compactify()
	tbl.stacks.compactify()
	return Stack{table: tbl, id: addStack(tbl, frames)}
}

// Stack represents a stack. It's really a handle to a stack and it's trivially comparable.
//
// If two Stacks are equal then their Frames are guaranteed to be identical. If they are not
// equal, however, their Frames may still be equal.
type Stack struct {
	table *evTable
	id    stackID
}

// Frames is an iterator over the frames in a Stack.
func (s Stack) Frames() iter.Seq[StackFrame] {
	return func(yield func(StackFrame) bool) {
		if s.id == 0 {
			return
		}
		stk := s.table.stacks.mustGet(s.id)
		for _, pc := range stk.pcs {
			f := s.table.pcs[pc]
			sf := StackFrame{
				PC:   f.pc,
				Func: s.table.strings.mustGet(f.funcID),
				File: s.table.strings.mustGet(f.fileID),
				Line: f.line,
			}
			if !yield(sf) {
				return
			}
		}
	}
}

// String returns the stack as a human-readable string.
//
// The format of the string is intended for debugging and is subject to change.
func (s Stack) String() string {
	var sb strings.Builder
	printStack(&sb, "", s.Frames())
	return sb.String()
}

func printStack(w io.Writer, prefix string, frames iter.Seq[StackFrame]) {
	for f := range frames {
		fmt.Fprintf(w, "%s%s @ 0x%x\n", prefix, f.Func, f.PC)
		fmt.Fprintf(w, "%s\t%s:%d\n", prefix, f.File, f.Line)
	}
}

// NoStack is a sentinel value that can be compared against any Stack value, indicating
// a lack of a stack trace.
var NoStack = Stack{}

// StackFrame represents a single frame of a stack.
type StackFrame struct {
	// PC is the program counter of the function call if this
	// is not a leaf frame. If it's a leaf frame, it's the point
	// at which the stack trace was taken.
	PC uint64

	// Func is the name of the function this frame maps to.
	Func string

	// File is the file which contains the source code of Func.
	File string

	// Line is the line number within File which maps to PC.
	Line uint64
}

// ExperimentalEvent presents a raw view of an experimental event's arguments and their names.
type ExperimentalEvent struct {
	// Name is the name of the event.
	Name string

	// Experiment is the name of the experiment this event is a part of.
	Experiment string

	// Args lists the names of the event's arguments in order.
	Args []string

	// argValues contains the raw integer arguments which are interpreted
	// by ArgValue using table.
	table     *evTable
	argValues []uint64
}

// ArgValue returns a typed Value for the i'th argument in the experimental event.
func (e ExperimentalEvent) ArgValue(i int) Value {
	if i < 0 || i >= len(e.Args) {
		panic(fmt.Sprintf("experimental event argument index %d out of bounds [0, %d)", i, len(e.Args)))
	}
	if strings.HasSuffix(e.Args[i], "string") {
		s := e.table.strings.mustGet(stringID(e.argValues[i]))
		return StringValue(s)
	}
	return Uint64Value(e.argValues[i])
}

// ExperimentalBatch represents a packet of unparsed data along with metadata about that packet.
type ExperimentalBatch struct {
	// Thread is the ID of the thread that produced a packet of data.
	Thread ThreadID

	// Data is a packet of unparsed data all produced by one thread.
	Data []byte
}

type EventDetails interface {
	Metric | Label | Range | StateTransition | Sync | Task | Region | Log | StackSample
}

// EventConfig holds the data for constructing a trace event.
type EventConfig[T EventDetails] struct {
	// Time is the timestamp of the event.
	Time Time

	// Kind is the kind of the event.
	Kind EventKind

	// Goroutine is the goroutine ID of the event.
	Goroutine GoID

	// Proc is the proc ID of the event.
	Proc ProcID

	// Thread is the thread ID of the event.
	Thread ThreadID

	// Stack is the stack of the event.
	Stack Stack

	// Details is the kind specific details of the event.
	Details T
}

// MakeEvent creates a new trace event from the given configuration.
func MakeEvent[T EventDetails](c EventConfig[T]) (e Event, err error) {
	// TODO(felixge): make the evTable reusable.
	e = Event{
		table: &evTable{pcs: make(map[uint64]frame), sync: sync{freq: 1}},
		base:  baseEvent{time: c.Time},
		ctx:   schedCtx{G: c.Goroutine, P: c.Proc, M: c.Thread},
	}
	defer func() {
		// N.b. evSync is not in tracev2.Specs()
		if err != nil || e.base.typ == evSync {
			return
		}
		spec := tracev2.Specs()[e.base.typ]
		if len(spec.StackIDs) > 0 && c.Stack != NoStack {
			// The stack for the main execution context is always the
			// first stack listed in StackIDs. Subtract one from this
			// because we've peeled away the timestamp argument.
			e.base.args[spec.StackIDs[0]-1] = uint64(addStack(e.table, slices.Collect(c.Stack.Frames())))
		}

		e.table.strings.compactify()
		e.table.stacks.compactify()
	}()
	var defaultKind EventKind
	switch c.Kind {
	case defaultKind:
		return Event{}, fmt.Errorf("the Kind field must be provided")
	case EventMetric:
		if m, ok := any(c.Details).(Metric); ok {
			return makeMetricEvent(e, m)
		}
	case EventLabel:
		if l, ok := any(c.Details).(Label); ok {
			return makeLabelEvent(e, l)
		}
	case EventRangeBegin, EventRangeActive, EventRangeEnd:
		if r, ok := any(c.Details).(Range); ok {
			return makeRangeEvent(e, c.Kind, r)
		}
	case EventStateTransition:
		if t, ok := any(c.Details).(StateTransition); ok {
			return makeStateTransitionEvent(e, t)
		}
	case EventSync:
		if s, ok := any(c.Details).(Sync); ok {
			return makeSyncEvent(e, s)
		}
	case EventTaskBegin, EventTaskEnd:
		if t, ok := any(c.Details).(Task); ok {
			return makeTaskEvent(e, c.Kind, t)
		}
	case EventRegionBegin, EventRegionEnd:
		if r, ok := any(c.Details).(Region); ok {
			return makeRegionEvent(e, c.Kind, r)
		}
	case EventLog:
		if l, ok := any(c.Details).(Log); ok {
			return makeLogEvent(e, l)
		}
	case EventStackSample:
		if _, ok := any(c.Details).(StackSample); ok {
			return makeStackSampleEvent(e, c.Stack)
		}
	}
	return Event{}, fmt.Errorf("the Kind field %s is incompatible with Details type %T", c.Kind, c.Details)
}

func makeMetricEvent(e Event, m Metric) (Event, error) {
	if m.Value.Kind() != ValueUint64 {
		return Event{}, fmt.Errorf("metric value must be a uint64, got: %s", m.Value.String())
	}
	switch m.Name {
	case "/sched/gomaxprocs:threads":
		e.base.typ = tracev2.EvProcsChange
	case "/memory/classes/heap/objects:bytes":
		e.base.typ = tracev2.EvHeapAlloc
	case "/gc/heap/goal:bytes":
		e.base.typ = tracev2.EvHeapGoal
	default:
		return Event{}, fmt.Errorf("unknown metric name: %s", m.Name)
	}
	e.base.args[0] = uint64(m.Value.Uint64())
	return e, nil
}

func makeLabelEvent(e Event, l Label) (Event, error) {
	if l.Resource.Kind != ResourceGoroutine {
		return Event{}, fmt.Errorf("resource must be a goroutine: %s", l.Resource)
	}
	e.base.typ = tracev2.EvGoLabel
	e.base.args[0] = uint64(e.table.strings.append(l.Label))
	// TODO(felixge): check against sched ctx and return error on mismatch
	e.ctx.G = l.Resource.Goroutine()
	return e, nil
}

var stwRangeRegexp = regexp.MustCompile(`^stop-the-world \((.*)\)$`)

// TODO(felixge): should this ever manipulate the e ctx? Or just report mismatches?
func makeRangeEvent(e Event, kind EventKind, r Range) (Event, error) {
	// TODO(felixge): Should we add dedicated range kinds rather than using
	// string names?
	switch r.Name {
	case "GC concurrent mark phase":
		if r.Scope.Kind != ResourceNone {
			return Event{}, fmt.Errorf("unexpected scope: %s", r.Scope)
		}
		switch kind {
		case EventRangeBegin:
			e.base.typ = tracev2.EvGCBegin
		case EventRangeActive:
			e.base.typ = tracev2.EvGCActive
		case EventRangeEnd:
			e.base.typ = tracev2.EvGCEnd
		default:
			return Event{}, fmt.Errorf("unexpected range kind: %s", kind)
		}
	case "GC incremental sweep":
		if r.Scope.Kind != ResourceProc {
			return Event{}, fmt.Errorf("unexpected scope: %s", r.Scope)
		}
		switch kind {
		case EventRangeBegin:
			e.base.typ = tracev2.EvGCSweepBegin
			e.ctx.P = r.Scope.Proc()
		case EventRangeActive:
			e.base.typ = tracev2.EvGCSweepActive
			e.base.args[0] = uint64(r.Scope.Proc())
		case EventRangeEnd:
			e.base.typ = tracev2.EvGCSweepEnd
			// TODO(felixge): check against sched ctx and return error on mismatch
			e.ctx.P = r.Scope.Proc()
		default:
			return Event{}, fmt.Errorf("unexpected range kind: %s", kind)
		}
	case "GC mark assist":
		if r.Scope.Kind != ResourceGoroutine {
			return Event{}, fmt.Errorf("unexpected scope: %s", r.Scope)
		}
		switch kind {
		case EventRangeBegin:
			e.base.typ = tracev2.EvGCMarkAssistBegin
			e.ctx.G = r.Scope.Goroutine()
		case EventRangeActive:
			e.base.typ = tracev2.EvGCMarkAssistActive
			e.base.args[0] = uint64(r.Scope.Goroutine())
		case EventRangeEnd:
			e.base.typ = tracev2.EvGCMarkAssistEnd
			// TODO(felixge): check against sched ctx and return error on mismatch
			e.ctx.G = r.Scope.Goroutine()
		default:
			return Event{}, fmt.Errorf("unexpected range kind: %s", kind)
		}
	default:
		match := stwRangeRegexp.FindStringSubmatch(r.Name)
		if len(match) != 2 {
			return Event{}, fmt.Errorf("unexpected range name: %s", r.Name)
		}
		if r.Scope.Kind != ResourceGoroutine {
			return Event{}, fmt.Errorf("unexpected scope: %s", r.Scope)
		}
		switch kind {
		case EventRangeBegin:
			e.base.typ = tracev2.EvSTWBegin
			// TODO(felixge): check against sched ctx and return error on mismatch
			e.ctx.G = r.Scope.Goroutine()
		case EventRangeEnd:
			e.base.typ = tracev2.EvSTWEnd
			// TODO(felixge): check against sched ctx and return error on mismatch
			e.ctx.G = r.Scope.Goroutine()
		default:
			return Event{}, fmt.Errorf("unexpected range kind: %s", kind)
		}
		e.base.args[0] = uint64(e.table.strings.append(match[1]))
	}
	return e, nil
}

func makeStateTransitionEvent(e Event, t StateTransition) (Event, error) {
	switch t.Resource.Kind {
	case ResourceProc:
		from, to := ProcState(t.oldState), ProcState(t.newState)
		switch {
		case from == ProcIdle && to == ProcIdle:
			// TODO(felixge): Could this also be a ProcStatus event?
			e.base.typ = tracev2.EvProcSteal
			e.base.args[0] = uint64(t.Resource.Proc())
			e.base.extra(version.Go122)[0] = uint64(tracev2.ProcSyscallAbandoned)
		case from == ProcIdle && to == ProcRunning:
			e.base.typ = tracev2.EvProcStart
			e.base.args[0] = uint64(t.Resource.Proc())
		case from == ProcRunning && to == ProcIdle:
			e.base.typ = tracev2.EvProcStop
			if t.Resource.Proc() != e.ctx.P {
				e.base.typ = tracev2.EvProcSteal
				e.base.args[0] = uint64(t.Resource.Proc())
			}
		default:
			e.base.typ = tracev2.EvProcStatus
			e.base.args[0] = uint64(t.Resource.Proc())
			e.base.args[1] = uint64(procState2Tracev2ProcStatus[to])
			e.base.extra(version.Go122)[0] = uint64(procState2Tracev2ProcStatus[from])
			return e, nil
		}
	case ResourceGoroutine:
		from, to := GoState(t.oldState), GoState(t.newState)
		stack := slices.Collect(t.Stack.Frames())
		goroutine := t.Resource.Goroutine()

		if (from == GoUndetermined || from == to) && from != GoNotExist {
			e.base.typ = tracev2.EvGoStatus
			if len(stack) > 0 {
				e.base.typ = tracev2.EvGoStatusStack
			}
			e.base.args[0] = uint64(goroutine)
			e.base.args[2] = uint64(from)<<32 | uint64(goState2Tracev2GoStatus[to])
		} else {
			switch from {
			case GoNotExist:
				switch to {
				case GoWaiting:
					e.base.typ = tracev2.EvGoCreateBlocked
					e.base.args[0] = uint64(goroutine)
					e.base.args[1] = uint64(addStack(e.table, stack))
				case GoRunnable:
					e.base.typ = tracev2.EvGoCreate
					e.base.args[0] = uint64(goroutine)
					e.base.args[1] = uint64(addStack(e.table, stack))
				case GoSyscall:
					e.base.typ = tracev2.EvGoCreateSyscall
					e.base.args[0] = uint64(goroutine)
				default:
					return Event{}, fmt.Errorf("unexpected transition: %s -> %s", from, to)
				}
			case GoRunnable:
				e.base.typ = tracev2.EvGoStart
				e.base.args[0] = uint64(goroutine)
			case GoRunning:
				switch to {
				case GoNotExist:
					e.base.typ = tracev2.EvGoDestroy
					e.ctx.G = goroutine
				case GoRunnable:
					e.base.typ = tracev2.EvGoStop
					e.ctx.G = goroutine
					e.base.args[0] = uint64(e.table.strings.append(t.Reason))
				case GoWaiting:
					e.base.typ = tracev2.EvGoBlock
					e.ctx.G = goroutine
					e.base.args[0] = uint64(e.table.strings.append(t.Reason))
				case GoSyscall:
					e.base.typ = tracev2.EvGoSyscallBegin
					e.ctx.G = goroutine
				default:
					return Event{}, fmt.Errorf("unexpected transition: %s -> %s", from, to)
				}
			case GoSyscall:
				switch to {
				case GoNotExist:
					e.base.typ = tracev2.EvGoDestroySyscall
					e.ctx.G = goroutine
				case GoRunning:
					e.base.typ = tracev2.EvGoSyscallEnd
					e.ctx.G = goroutine
				case GoRunnable:
					e.base.typ = tracev2.EvGoSyscallEndBlocked
					e.ctx.G = goroutine
				default:
					return Event{}, fmt.Errorf("unexpected transition: %s -> %s", from, to)
				}
			case GoWaiting:
				switch to {
				case GoRunnable:
					e.base.typ = tracev2.EvGoUnblock
					e.base.args[0] = uint64(goroutine)
				default:
					return Event{}, fmt.Errorf("unexpected transition: %s -> %s", from, to)
				}
			default:
				return Event{}, fmt.Errorf("unexpected transition: %s -> %s", from, to)
			}
		}
	default:
		return Event{}, fmt.Errorf("unsupported state transition resource: %s", t.Resource)
	}
	return e, nil
}

func makeSyncEvent(e Event, s Sync) (Event, error) {
	e.base.typ = evSync
	e.base.args[0] = uint64(s.N)
	if e.table.expBatches == nil {
		e.table.expBatches = make(map[tracev2.Experiment][]ExperimentalBatch)
	}
	for name, batches := range s.ExperimentalBatches {
		var found bool
		for id, exp := range tracev2.Experiments() {
			if exp == name {
				found = true
				e.table.expBatches[tracev2.Experiment(id)] = batches
				break
			}
		}
		if !found {
			return Event{}, fmt.Errorf("unknown experiment: %s", name)
		}
	}
	if s.ClockSnapshot != nil {
		e.table.hasClockSnapshot = true
		e.table.snapWall = s.ClockSnapshot.Wall
		e.table.snapMono = s.ClockSnapshot.Mono
		// N.b. MakeEvent sets e.table.freq to 1.
		e.table.snapTime = timestamp(s.ClockSnapshot.Trace)
	}
	return e, nil
}

func makeTaskEvent(e Event, kind EventKind, t Task) (Event, error) {
	if t.ID == NoTask {
		return Event{}, errors.New("task ID cannot be NoTask")
	}
	e.base.args[0] = uint64(t.ID)
	switch kind {
	case EventTaskBegin:
		e.base.typ = tracev2.EvUserTaskBegin
		e.base.args[1] = uint64(t.Parent)
		e.base.args[2] = uint64(e.table.strings.append(t.Type))
	case EventTaskEnd:
		e.base.typ = tracev2.EvUserTaskEnd
		e.base.extra(version.Go122)[0] = uint64(t.Parent)
		e.base.extra(version.Go122)[1] = uint64(e.table.addExtraString(t.Type))
	default:
		// TODO(felixge): also do this for ranges?
		panic("unexpected task kind")
	}
	return e, nil
}

func makeRegionEvent(e Event, kind EventKind, r Region) (Event, error) {
	e.base.args[0] = uint64(r.Task)
	e.base.args[1] = uint64(e.table.strings.append(r.Type))
	switch kind {
	case EventRegionBegin:
		e.base.typ = tracev2.EvUserRegionBegin
	case EventRegionEnd:
		e.base.typ = tracev2.EvUserRegionEnd
	default:
		panic("unexpected region kind")
	}
	return e, nil
}

func makeLogEvent(e Event, l Log) (Event, error) {
	e.base.typ = tracev2.EvUserLog
	e.base.args[0] = uint64(l.Task)
	e.base.args[1] = uint64(e.table.strings.append(l.Category))
	e.base.args[2] = uint64(e.table.strings.append(l.Message))
	return e, nil
}

func makeStackSampleEvent(e Event, s Stack) (Event, error) {
	e.base.typ = tracev2.EvCPUSample
	frames := slices.Collect(s.Frames())
	e.base.args[0] = uint64(addStack(e.table, frames))
	return e, nil
}

func addStack(table *evTable, frames []StackFrame) stackID {
	var pcs []uint64
	for _, f := range frames {
		table.pcs[f.PC] = frame{
			pc:     f.PC,
			funcID: table.strings.append(f.Func),
			fileID: table.strings.append(f.File),
			line:   f.Line,
		}
		pcs = append(pcs, f.PC)
	}
	return table.stacks.append(stack{pcs: pcs})
}

// Event represents a single event in the trace.
type Event struct {
	table *evTable
	ctx   schedCtx
	base  baseEvent
}

// Kind returns the kind of event that this is.
func (e Event) Kind() EventKind {
	return tracev2Type2Kind[e.base.typ]
}

// Time returns the timestamp of the event.
func (e Event) Time() Time {
	return e.base.time
}

// Goroutine returns the ID of the goroutine that was executing when
// this event happened. It describes part of the execution context
// for this event.
//
// Note that for goroutine state transitions this always refers to the
// state before the transition. For example, if a goroutine is just
// starting to run on this thread and/or proc, then this will return
// NoGoroutine. In this case, the goroutine starting to run will be
// can be found at Event.StateTransition().Resource.
func (e Event) Goroutine() GoID {
	return e.ctx.G
}

// Proc returns the ID of the proc this event event pertains to.
//
// Note that for proc state transitions this always refers to the
// state before the transition. For example, if a proc is just
// starting to run on this thread, then this will return NoProc.
func (e Event) Proc() ProcID {
	return e.ctx.P
}

// Thread returns the ID of the thread this event pertains to.
//
// Note that for thread state transitions this always refers to the
// state before the transition. For example, if a thread is just
// starting to run, then this will return NoThread.
//
// Note: tracking thread state is not currently supported, so this
// will always return a valid thread ID. However thread state transitions
// may be tracked in the future, and callers must be robust to this
// possibility.
func (e Event) Thread() ThreadID {
	return e.ctx.M
}

// Stack returns a handle to a stack associated with the event.
//
// This represents a stack trace at the current moment in time for
// the current execution context.
func (e Event) Stack() Stack {
	if e.base.typ == evSync {
		return NoStack
	}
	if e.base.typ == tracev2.EvCPUSample {
		return Stack{table: e.table, id: stackID(e.base.args[0])}
	}
	spec := tracev2.Specs()[e.base.typ]
	if len(spec.StackIDs) == 0 {
		return NoStack
	}
	// The stack for the main execution context is always the
	// first stack listed in StackIDs. Subtract one from this
	// because we've peeled away the timestamp argument.
	id := stackID(e.base.args[spec.StackIDs[0]-1])
	if id == 0 {
		return NoStack
	}
	return Stack{table: e.table, id: id}
}

// Metric returns details about a Metric event.
//
// Panics if Kind != EventMetric.
func (e Event) Metric() Metric {
	if e.Kind() != EventMetric {
		panic("Metric called on non-Metric event")
	}
	var m Metric
	switch e.base.typ {
	case tracev2.EvProcsChange:
		m.Name = "/sched/gomaxprocs:threads"
		m.Value = Uint64Value(e.base.args[0])
	case tracev2.EvHeapAlloc:
		m.Name = "/memory/classes/heap/objects:bytes"
		m.Value = Uint64Value(e.base.args[0])
	case tracev2.EvHeapGoal:
		m.Name = "/gc/heap/goal:bytes"
		m.Value = Uint64Value(e.base.args[0])
	default:
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for Metric kind: %d", e.base.typ))
	}
	return m
}

// Label returns details about a Label event.
//
// Panics if Kind != EventLabel.
func (e Event) Label() Label {
	if e.Kind() != EventLabel {
		panic("Label called on non-Label event")
	}
	if e.base.typ != tracev2.EvGoLabel {
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for Label kind: %d", e.base.typ))
	}
	return Label{
		Label:    e.table.strings.mustGet(stringID(e.base.args[0])),
		Resource: ResourceID{Kind: ResourceGoroutine, id: int64(e.ctx.G)},
	}
}

// Range returns details about an EventRangeBegin, EventRangeActive, or EventRangeEnd event.
//
// Panics if Kind != EventRangeBegin, Kind != EventRangeActive, and Kind != EventRangeEnd.
func (e Event) Range() Range {
	if kind := e.Kind(); kind != EventRangeBegin && kind != EventRangeActive && kind != EventRangeEnd {
		panic("Range called on non-Range event")
	}
	var r Range
	switch e.base.typ {
	case tracev2.EvSTWBegin, tracev2.EvSTWEnd:
		// N.B. ordering.advance smuggles in the STW reason as e.base.args[0]
		// for tracev2.EvSTWEnd (it's already there for Begin).
		r.Name = "stop-the-world (" + e.table.strings.mustGet(stringID(e.base.args[0])) + ")"
		r.Scope = ResourceID{Kind: ResourceGoroutine, id: int64(e.Goroutine())}
	case tracev2.EvGCBegin, tracev2.EvGCActive, tracev2.EvGCEnd:
		r.Name = "GC concurrent mark phase"
		r.Scope = ResourceID{Kind: ResourceNone}
	case tracev2.EvGCSweepBegin, tracev2.EvGCSweepActive, tracev2.EvGCSweepEnd:
		r.Name = "GC incremental sweep"
		r.Scope = ResourceID{Kind: ResourceProc}
		if e.base.typ == tracev2.EvGCSweepActive {
			r.Scope.id = int64(e.base.args[0])
		} else {
			r.Scope.id = int64(e.Proc())
		}
	case tracev2.EvGCMarkAssistBegin, tracev2.EvGCMarkAssistActive, tracev2.EvGCMarkAssistEnd:
		r.Name = "GC mark assist"
		r.Scope = ResourceID{Kind: ResourceGoroutine}
		if e.base.typ == tracev2.EvGCMarkAssistActive {
			r.Scope.id = int64(e.base.args[0])
		} else {
			r.Scope.id = int64(e.Goroutine())
		}
	default:
		panic(fmt.Sprintf("internal error: unexpected wire-event type for Range kind: %d", e.base.typ))
	}
	return r
}

// RangeAttributes returns attributes for a completed range.
//
// Panics if Kind != EventRangeEnd.
func (e Event) RangeAttributes() []RangeAttribute {
	if e.Kind() != EventRangeEnd {
		panic("Range called on non-Range event")
	}
	if e.base.typ != tracev2.EvGCSweepEnd {
		return nil
	}
	return []RangeAttribute{
		{
			Name:  "bytes swept",
			Value: Uint64Value(e.base.args[0]),
		},
		{
			Name:  "bytes reclaimed",
			Value: Uint64Value(e.base.args[1]),
		},
	}
}

// Task returns details about a TaskBegin or TaskEnd event.
//
// Panics if Kind != EventTaskBegin and Kind != EventTaskEnd.
func (e Event) Task() Task {
	if kind := e.Kind(); kind != EventTaskBegin && kind != EventTaskEnd {
		panic("Task called on non-Task event")
	}
	parentID := NoTask
	var typ string
	switch e.base.typ {
	case tracev2.EvUserTaskBegin:
		parentID = TaskID(e.base.args[1])
		typ = e.table.strings.mustGet(stringID(e.base.args[2]))
	case tracev2.EvUserTaskEnd:
		parentID = TaskID(e.base.extra(version.Go122)[0])
		typ = e.table.getExtraString(extraStringID(e.base.extra(version.Go122)[1]))
	default:
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for Task kind: %d", e.base.typ))
	}
	return Task{
		ID:     TaskID(e.base.args[0]),
		Parent: parentID,
		Type:   typ,
	}
}

// Region returns details about a RegionBegin or RegionEnd event.
//
// Panics if Kind != EventRegionBegin and Kind != EventRegionEnd.
func (e Event) Region() Region {
	if kind := e.Kind(); kind != EventRegionBegin && kind != EventRegionEnd {
		panic("Region called on non-Region event")
	}
	if e.base.typ != tracev2.EvUserRegionBegin && e.base.typ != tracev2.EvUserRegionEnd {
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for Region kind: %d", e.base.typ))
	}
	return Region{
		Task: TaskID(e.base.args[0]),
		Type: e.table.strings.mustGet(stringID(e.base.args[1])),
	}
}

// Log returns details about a Log event.
//
// Panics if Kind != EventLog.
func (e Event) Log() Log {
	if e.Kind() != EventLog {
		panic("Log called on non-Log event")
	}
	if e.base.typ != tracev2.EvUserLog {
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for Log kind: %d", e.base.typ))
	}
	return Log{
		Task:     TaskID(e.base.args[0]),
		Category: e.table.strings.mustGet(stringID(e.base.args[1])),
		Message:  e.table.strings.mustGet(stringID(e.base.args[2])),
	}
}

// StateTransition returns details about a StateTransition event.
//
// Panics if Kind != EventStateTransition.
func (e Event) StateTransition() StateTransition {
	if e.Kind() != EventStateTransition {
		panic("StateTransition called on non-StateTransition event")
	}
	var s StateTransition
	switch e.base.typ {
	case tracev2.EvProcStart:
		s = MakeProcStateTransition(ProcID(e.base.args[0]), ProcIdle, ProcRunning)
	case tracev2.EvProcStop:
		s = MakeProcStateTransition(e.ctx.P, ProcRunning, ProcIdle)
	case tracev2.EvProcSteal:
		// N.B. ordering.advance populates e.base.extra.
		beforeState := ProcRunning
		if tracev2.ProcStatus(e.base.extra(version.Go122)[0]) == tracev2.ProcSyscallAbandoned {
			// We've lost information because this ProcSteal advanced on a
			// SyscallAbandoned state. Treat the P as idle because ProcStatus
			// treats SyscallAbandoned as Idle. Otherwise we'll have an invalid
			// transition.
			beforeState = ProcIdle
		}
		s = MakeProcStateTransition(ProcID(e.base.args[0]), beforeState, ProcIdle)
	case tracev2.EvProcStatus:
		// N.B. ordering.advance populates e.base.extra.
		s = MakeProcStateTransition(ProcID(e.base.args[0]), ProcState(e.base.extra(version.Go122)[0]), tracev2ProcStatus2ProcState[e.base.args[1]])
	case tracev2.EvGoCreate, tracev2.EvGoCreateBlocked:
		status := GoRunnable
		if e.base.typ == tracev2.EvGoCreateBlocked {
			status = GoWaiting
		}
		s = MakeGoStateTransition(GoID(e.base.args[0]), GoNotExist, status)
		s.Stack = Stack{table: e.table, id: stackID(e.base.args[1])}
	case tracev2.EvGoCreateSyscall:
		s = MakeGoStateTransition(GoID(e.base.args[0]), GoNotExist, GoSyscall)
	case tracev2.EvGoStart:
		s = MakeGoStateTransition(GoID(e.base.args[0]), GoRunnable, GoRunning)
	case tracev2.EvGoDestroy:
		s = MakeGoStateTransition(e.ctx.G, GoRunning, GoNotExist)
	case tracev2.EvGoDestroySyscall:
		s = MakeGoStateTransition(e.ctx.G, GoSyscall, GoNotExist)
	case tracev2.EvGoStop:
		s = MakeGoStateTransition(e.ctx.G, GoRunning, GoRunnable)
		s.Reason = e.table.strings.mustGet(stringID(e.base.args[0]))
		s.Stack = e.Stack() // This event references the resource the event happened on.
	case tracev2.EvGoBlock:
		s = MakeGoStateTransition(e.ctx.G, GoRunning, GoWaiting)
		s.Reason = e.table.strings.mustGet(stringID(e.base.args[0]))
		s.Stack = e.Stack() // This event references the resource the event happened on.
	case tracev2.EvGoUnblock, tracev2.EvGoSwitch, tracev2.EvGoSwitchDestroy:
		// N.B. GoSwitch and GoSwitchDestroy both emit additional events, but
		// the first thing they both do is unblock the goroutine they name,
		// identically to an unblock event (even their arguments match).
		s = MakeGoStateTransition(GoID(e.base.args[0]), GoWaiting, GoRunnable)
	case tracev2.EvGoSyscallBegin:
		s = MakeGoStateTransition(e.ctx.G, GoRunning, GoSyscall)
		s.Stack = e.Stack() // This event references the resource the event happened on.
	case tracev2.EvGoSyscallEnd:
		s = MakeGoStateTransition(e.ctx.G, GoSyscall, GoRunning)
	case tracev2.EvGoSyscallEndBlocked:
		s = MakeGoStateTransition(e.ctx.G, GoSyscall, GoRunnable)
	case tracev2.EvGoStatus, tracev2.EvGoStatusStack:
		packedStatus := e.base.args[2]
		from, to := packedStatus>>32, packedStatus&((1<<32)-1)
		s = MakeGoStateTransition(GoID(e.base.args[0]), GoState(from), tracev2GoStatus2GoState[to])
		s.Stack = e.Stack() // This event references the resource the event happened on.
	default:
		panic(fmt.Sprintf("internal error: unexpected wire-format event type for StateTransition kind: %d", e.base.typ))
	}
	return s
}

// Sync returns details that are relevant for the following events, up to but excluding the
// next EventSync event.
func (e Event) Sync() Sync {
	if e.Kind() != EventSync {
		panic("Sync called on non-Sync event")
	}
	s := Sync{N: int(e.base.args[0])}
	if e.table != nil {
		expBatches := make(map[string][]ExperimentalBatch)
		for exp, batches := range e.table.expBatches {
			expBatches[tracev2.Experiments()[exp]] = batches
		}
		s.ExperimentalBatches = expBatches
		if e.table.hasClockSnapshot {
			s.ClockSnapshot = &ClockSnapshot{
				Trace: e.table.freq.mul(e.table.snapTime),
				Wall:  e.table.snapWall,
				Mono:  e.table.snapMono,
			}
		}
	}
	return s
}

// Sync contains details potentially relevant to all the following events, up to but excluding
// the next EventSync event.
type Sync struct {
	// N indicates that this is the Nth sync event in the trace.
	N int

	// ClockSnapshot represents a near-simultaneous clock reading of several
	// different system clocks. The snapshot can be used as a reference to
	// convert timestamps to different clocks, which is helpful for correlating
	// timestamps with data captured by other tools. The value is nil for traces
	// before go1.25.
	ClockSnapshot *ClockSnapshot

	// ExperimentalBatches contain all the unparsed batches of data for a given experiment.
	ExperimentalBatches map[string][]ExperimentalBatch
}

// ClockSnapshot represents a near-simultaneous clock reading of several
// different system clocks. The snapshot can be used as a reference to convert
// timestamps to different clocks, which is helpful for correlating timestamps
// with data captured by other tools.
type ClockSnapshot struct {
	// Trace is a snapshot of the trace clock.
	Trace Time

	// Wall is a snapshot of the system's wall clock.
	Wall time.Time

	// Mono is a snapshot of the system's monotonic clock.
	Mono uint64
}

// Experimental returns a view of the raw event for an experimental event.
//
// Panics if Kind != EventExperimental.
func (e Event) Experimental() ExperimentalEvent {
	if e.Kind() != EventExperimental {
		panic("Experimental called on non-Experimental event")
	}
	spec := tracev2.Specs()[e.base.typ]
	argNames := spec.Args[1:] // Skip timestamp; already handled.
	return ExperimentalEvent{
		Name:       spec.Name,
		Experiment: tracev2.Experiments()[spec.Experiment],
		Args:       argNames,
		table:      e.table,
		argValues:  e.base.args[:len(argNames)],
	}
}

const evSync = ^tracev2.EventType(0)

var tracev2Type2Kind = [...]EventKind{
	tracev2.EvCPUSample:           EventStackSample,
	tracev2.EvProcsChange:         EventMetric,
	tracev2.EvProcStart:           EventStateTransition,
	tracev2.EvProcStop:            EventStateTransition,
	tracev2.EvProcSteal:           EventStateTransition,
	tracev2.EvProcStatus:          EventStateTransition,
	tracev2.EvGoCreate:            EventStateTransition,
	tracev2.EvGoCreateSyscall:     EventStateTransition,
	tracev2.EvGoStart:             EventStateTransition,
	tracev2.EvGoDestroy:           EventStateTransition,
	tracev2.EvGoDestroySyscall:    EventStateTransition,
	tracev2.EvGoStop:              EventStateTransition,
	tracev2.EvGoBlock:             EventStateTransition,
	tracev2.EvGoUnblock:           EventStateTransition,
	tracev2.EvGoSyscallBegin:      EventStateTransition,
	tracev2.EvGoSyscallEnd:        EventStateTransition,
	tracev2.EvGoSyscallEndBlocked: EventStateTransition,
	tracev2.EvGoStatus:            EventStateTransition,
	tracev2.EvSTWBegin:            EventRangeBegin,
	tracev2.EvSTWEnd:              EventRangeEnd,
	tracev2.EvGCActive:            EventRangeActive,
	tracev2.EvGCBegin:             EventRangeBegin,
	tracev2.EvGCEnd:               EventRangeEnd,
	tracev2.EvGCSweepActive:       EventRangeActive,
	tracev2.EvGCSweepBegin:        EventRangeBegin,
	tracev2.EvGCSweepEnd:          EventRangeEnd,
	tracev2.EvGCMarkAssistActive:  EventRangeActive,
	tracev2.EvGCMarkAssistBegin:   EventRangeBegin,
	tracev2.EvGCMarkAssistEnd:     EventRangeEnd,
	tracev2.EvHeapAlloc:           EventMetric,
	tracev2.EvHeapGoal:            EventMetric,
	tracev2.EvGoLabel:             EventLabel,
	tracev2.EvUserTaskBegin:       EventTaskBegin,
	tracev2.EvUserTaskEnd:         EventTaskEnd,
	tracev2.EvUserRegionBegin:     EventRegionBegin,
	tracev2.EvUserRegionEnd:       EventRegionEnd,
	tracev2.EvUserLog:             EventLog,
	tracev2.EvGoSwitch:            EventStateTransition,
	tracev2.EvGoSwitchDestroy:     EventStateTransition,
	tracev2.EvGoCreateBlocked:     EventStateTransition,
	tracev2.EvGoStatusStack:       EventStateTransition,
	tracev2.EvSpan:                EventExperimental,
	tracev2.EvSpanAlloc:           EventExperimental,
	tracev2.EvSpanFree:            EventExperimental,
	tracev2.EvHeapObject:          EventExperimental,
	tracev2.EvHeapObjectAlloc:     EventExperimental,
	tracev2.EvHeapObjectFree:      EventExperimental,
	tracev2.EvGoroutineStack:      EventExperimental,
	tracev2.EvGoroutineStackAlloc: EventExperimental,
	tracev2.EvGoroutineStackFree:  EventExperimental,
	evSync:                        EventSync,
}

var tracev2GoStatus2GoState = [...]GoState{
	tracev2.GoRunnable: GoRunnable,
	tracev2.GoRunning:  GoRunning,
	tracev2.GoWaiting:  GoWaiting,
	tracev2.GoSyscall:  GoSyscall,
}

var goState2Tracev2GoStatus = [...]tracev2.GoStatus{
	GoRunnable: tracev2.GoRunnable,
	GoRunning:  tracev2.GoRunning,
	GoWaiting:  tracev2.GoWaiting,
	GoSyscall:  tracev2.GoSyscall,
}

var tracev2ProcStatus2ProcState = [...]ProcState{
	tracev2.ProcRunning:          ProcRunning,
	tracev2.ProcIdle:             ProcIdle,
	tracev2.ProcSyscall:          ProcRunning,
	tracev2.ProcSyscallAbandoned: ProcIdle,
}

var procState2Tracev2ProcStatus = [...]tracev2.ProcStatus{
	ProcRunning: tracev2.ProcRunning,
	ProcIdle:    tracev2.ProcIdle,
	// TODO(felixge): how to map ProcSyscall and ProcSyscallAbandoned?
}

// String returns the event as a human-readable string.
//
// The format of the string is intended for debugging and is subject to change.
func (e Event) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "M=%d P=%d G=%d", e.Thread(), e.Proc(), e.Goroutine())
	fmt.Fprintf(&sb, " %s Time=%d", e.Kind(), e.Time())
	// Kind-specific fields.
	switch kind := e.Kind(); kind {
	case EventMetric:
		m := e.Metric()
		v := m.Value.String()
		if m.Value.Kind() == ValueString {
			v = strconv.Quote(v)
		}
		fmt.Fprintf(&sb, " Name=%q Value=%s", m.Name, m.Value)
	case EventLabel:
		l := e.Label()
		fmt.Fprintf(&sb, " Label=%q Resource=%s", l.Label, l.Resource)
	case EventRangeBegin, EventRangeActive, EventRangeEnd:
		r := e.Range()
		fmt.Fprintf(&sb, " Name=%q Scope=%s", r.Name, r.Scope)
		if kind == EventRangeEnd {
			fmt.Fprintf(&sb, " Attributes=[")
			for i, attr := range e.RangeAttributes() {
				if i != 0 {
					fmt.Fprintf(&sb, " ")
				}
				fmt.Fprintf(&sb, "%q=%s", attr.Name, attr.Value)
			}
			fmt.Fprintf(&sb, "]")
		}
	case EventTaskBegin, EventTaskEnd:
		t := e.Task()
		fmt.Fprintf(&sb, " ID=%d Parent=%d Type=%q", t.ID, t.Parent, t.Type)
	case EventRegionBegin, EventRegionEnd:
		r := e.Region()
		fmt.Fprintf(&sb, " Task=%d Type=%q", r.Task, r.Type)
	case EventLog:
		l := e.Log()
		fmt.Fprintf(&sb, " Task=%d Category=%q Message=%q", l.Task, l.Category, l.Message)
	case EventStateTransition:
		s := e.StateTransition()
		switch s.Resource.Kind {
		case ResourceGoroutine:
			id := s.Resource.Goroutine()
			old, new := s.Goroutine()
			fmt.Fprintf(&sb, " GoID=%d %s->%s", id, old, new)
		case ResourceProc:
			id := s.Resource.Proc()
			old, new := s.Proc()
			fmt.Fprintf(&sb, " ProcID=%d %s->%s", id, old, new)
		}
		fmt.Fprintf(&sb, " Reason=%q", s.Reason)
		if s.Stack != NoStack {
			fmt.Fprintln(&sb)
			fmt.Fprintln(&sb, "TransitionStack=")
			printStack(&sb, "\t", s.Stack.Frames())
		}
	case EventExperimental:
		r := e.Experimental()
		fmt.Fprintf(&sb, " Name=%s Args=[", r.Name)
		for i, arg := range r.Args {
			if i != 0 {
				fmt.Fprintf(&sb, ", ")
			}
			fmt.Fprintf(&sb, "%s=%s", arg, r.ArgValue(i).String())
		}
		fmt.Fprintf(&sb, "]")
	case EventSync:
		s := e.Sync()
		fmt.Fprintf(&sb, " N=%d", s.N)
		if s.ClockSnapshot != nil {
			fmt.Fprintf(&sb, " Trace=%d Mono=%d Wall=%s",
				s.ClockSnapshot.Trace,
				s.ClockSnapshot.Mono,
				s.ClockSnapshot.Wall.Format(time.RFC3339Nano),
			)
		}
	}
	if stk := e.Stack(); stk != NoStack {
		fmt.Fprintln(&sb)
		fmt.Fprintln(&sb, "Stack=")
		printStack(&sb, "\t", stk.Frames())
	}
	return sb.String()
}

// validateTableIDs checks to make sure lookups in e.table
// will work.
func (e Event) validateTableIDs() error {
	if e.base.typ == evSync {
		return nil
	}
	spec := tracev2.Specs()[e.base.typ]

	// Check stacks.
	for _, i := range spec.StackIDs {
		id := stackID(e.base.args[i-1])
		_, ok := e.table.stacks.get(id)
		if !ok {
			return fmt.Errorf("found invalid stack ID %d for event %s", id, spec.Name)
		}
	}
	// N.B. Strings referenced by stack frames are validated
	// early on, when reading the stacks in to begin with.

	// Check strings.
	for _, i := range spec.StringIDs {
		id := stringID(e.base.args[i-1])
		_, ok := e.table.strings.get(id)
		if !ok {
			return fmt.Errorf("found invalid string ID %d for event %s", id, spec.Name)
		}
	}
	return nil
}

func syncEvent(table *evTable, ts Time, n int) Event {
	ev := Event{
		table: table,
		ctx: schedCtx{
			G: NoGoroutine,
			P: NoProc,
			M: NoThread,
		},
		base: baseEvent{
			typ:  evSync,
			time: ts,
		},
	}
	ev.base.args[0] = uint64(n)
	return ev
}
