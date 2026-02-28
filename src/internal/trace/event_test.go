// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"internal/diff"
	"reflect"
	"slices"
	"testing"
	"time"
)

func TestMakeEvent(t *testing.T) {
	checkTime := func(t *testing.T, ev Event, want Time) {
		t.Helper()
		if ev.Time() != want {
			t.Errorf("expected time to be %d, got %d", want, ev.Time())
		}
	}
	checkValid := func(t *testing.T, err error, valid bool) bool {
		t.Helper()
		if valid && err == nil {
			return true
		}
		if valid && err != nil {
			t.Errorf("expected no error, got %v", err)
		} else if !valid && err == nil {
			t.Errorf("expected error, got %v", err)
		}
		return false
	}
	type stackType string
	const (
		schedStack stackType = "sched stack"
		stStack    stackType = "state transition stack"
	)
	checkStack := func(t *testing.T, got Stack, want Stack, which stackType) {
		t.Helper()
		diff := diff.Diff("want", []byte(want.String()), "got", []byte(got.String()))
		if len(diff) > 0 {
			t.Errorf("unexpected %s: %s", which, diff)
		}
	}
	stk1 := MakeStack([]StackFrame{
		{PC: 1, Func: "foo", File: "foo.go", Line: 10},
		{PC: 2, Func: "bar", File: "bar.go", Line: 20},
	})
	stk2 := MakeStack([]StackFrame{
		{PC: 1, Func: "foo", File: "foo.go", Line: 10},
		{PC: 2, Func: "bar", File: "bar.go", Line: 20},
	})

	t.Run("Metric", func(t *testing.T) {
		tests := []struct {
			name   string
			metric string
			val    uint64
			stack  Stack
			valid  bool
		}{
			{name: "gomaxprocs", metric: "/sched/gomaxprocs:threads", valid: true, val: 1, stack: NoStack},
			{name: "gomaxprocs with stack", metric: "/sched/gomaxprocs:threads", valid: true, val: 1, stack: stk1},
			{name: "heap objects", metric: "/memory/classes/heap/objects:bytes", valid: true, val: 2, stack: NoStack},
			{name: "heap goal", metric: "/gc/heap/goal:bytes", valid: true, val: 3, stack: NoStack},
			{name: "invalid metric", metric: "/test", valid: false, val: 4, stack: NoStack},
		}
		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Metric]{
					Kind:    EventMetric,
					Time:    Time(42 + i),
					Details: Metric{Name: test.metric, Value: Uint64Value(test.val)},
					Stack:   test.stack,
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				checkStack(t, ev.Stack(), test.stack, schedStack)
				got := ev.Metric()
				if got.Name != test.metric {
					t.Errorf("expected name to be %q, got %q", test.metric, got.Name)
				}
				if got.Value.Uint64() != test.val {
					t.Errorf("expected value to be %d, got %d", test.val, got.Value.Uint64())
				}
			})
		}
	})

	t.Run("Label", func(t *testing.T) {
		ev, err := MakeEvent(EventConfig[Label]{
			Kind:    EventLabel,
			Time:    42,
			Details: Label{Label: "test", Resource: MakeResourceID(GoID(23))},
		})
		if !checkValid(t, err, true) {
			return
		}
		label := ev.Label()
		if label.Label != "test" {
			t.Errorf("expected label to be test, got %q", label.Label)
		}
		if label.Resource.Kind != ResourceGoroutine {
			t.Errorf("expected label resource to be goroutine, got %d", label.Resource.Kind)
		}
		if label.Resource.id != 23 {
			t.Errorf("expected label resource to be 23, got %d", label.Resource.id)
		}
		checkTime(t, ev, 42)
	})

	t.Run("Range", func(t *testing.T) {
		tests := []struct {
			kind  EventKind
			name  string
			scope ResourceID
			valid bool
		}{
			{kind: EventRangeBegin, name: "GC concurrent mark phase", scope: ResourceID{}, valid: true},
			{kind: EventRangeActive, name: "GC concurrent mark phase", scope: ResourceID{}, valid: true},
			{kind: EventRangeEnd, name: "GC concurrent mark phase", scope: ResourceID{}, valid: true},
			{kind: EventMetric, name: "GC concurrent mark phase", scope: ResourceID{}, valid: false},
			{kind: EventRangeBegin, name: "GC concurrent mark phase - INVALID", scope: ResourceID{}, valid: false},

			{kind: EventRangeBegin, name: "GC incremental sweep", scope: MakeResourceID(ProcID(1)), valid: true},
			{kind: EventRangeActive, name: "GC incremental sweep", scope: MakeResourceID(ProcID(2)), valid: true},
			{kind: EventRangeEnd, name: "GC incremental sweep", scope: MakeResourceID(ProcID(3)), valid: true},
			{kind: EventMetric, name: "GC incremental sweep", scope: MakeResourceID(ProcID(4)), valid: false},
			{kind: EventRangeBegin, name: "GC incremental sweep - INVALID", scope: MakeResourceID(ProcID(5)), valid: false},

			{kind: EventRangeBegin, name: "GC mark assist", scope: MakeResourceID(GoID(1)), valid: true},
			{kind: EventRangeActive, name: "GC mark assist", scope: MakeResourceID(GoID(2)), valid: true},
			{kind: EventRangeEnd, name: "GC mark assist", scope: MakeResourceID(GoID(3)), valid: true},
			{kind: EventMetric, name: "GC mark assist", scope: MakeResourceID(GoID(4)), valid: false},
			{kind: EventRangeBegin, name: "GC mark assist - INVALID", scope: MakeResourceID(GoID(5)), valid: false},

			{kind: EventRangeBegin, name: "stop-the-world (for a good reason)", scope: MakeResourceID(GoID(1)), valid: true},
			{kind: EventRangeActive, name: "stop-the-world (for a good reason)", scope: MakeResourceID(GoID(2)), valid: false},
			{kind: EventRangeEnd, name: "stop-the-world (for a good reason)", scope: MakeResourceID(GoID(3)), valid: true},
			{kind: EventMetric, name: "stop-the-world (for a good reason)", scope: MakeResourceID(GoID(4)), valid: false},
			{kind: EventRangeBegin, name: "stop-the-world (for a good reason) - INVALID", scope: MakeResourceID(GoID(5)), valid: false},
		}

		for i, test := range tests {
			name := fmt.Sprintf("%s/%s/%s", test.kind, test.name, test.scope)
			t.Run(name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Range]{
					Time:    Time(42 + i),
					Kind:    test.kind,
					Details: Range{Name: test.name, Scope: test.scope},
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				got := ev.Range()
				if got.Name != test.name {
					t.Errorf("expected name to be %q, got %q", test.name, got.Name)
				}
				if ev.Kind() != test.kind {
					t.Errorf("expected kind to be %s, got %s", test.kind, ev.Kind())
				}
				if got.Scope.String() != test.scope.String() {
					t.Errorf("expected scope to be %s, got %s", test.scope.String(), got.Scope.String())
				}
				checkTime(t, ev, Time(42+i))
			})
		}
	})

	t.Run("GoroutineTransition", func(t *testing.T) {
		const anotherG = 999 // indicates hat sched g is different from transition g
		tests := []struct {
			name    string
			g       GoID
			stack   Stack
			stG     GoID
			from    GoState
			to      GoState
			reason  string
			stStack Stack
			valid   bool
		}{
			{
				name:    "EvGoCreate",
				g:       anotherG,
				stack:   stk1,
				stG:     1,
				from:    GoNotExist,
				to:      GoRunnable,
				reason:  "",
				stStack: stk2,
				valid:   true,
			},
			{
				name:    "EvGoCreateBlocked",
				g:       anotherG,
				stack:   stk1,
				stG:     2,
				from:    GoNotExist,
				to:      GoWaiting,
				reason:  "",
				stStack: stk2,
				valid:   true,
			},
			{
				name:    "EvGoCreateSyscall",
				g:       anotherG,
				stack:   NoStack,
				stG:     3,
				from:    GoNotExist,
				to:      GoSyscall,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "EvGoStart",
				g:       anotherG,
				stack:   NoStack,
				stG:     4,
				from:    GoRunnable,
				to:      GoRunning,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "EvGoDestroy",
				g:       5,
				stack:   NoStack,
				stG:     5,
				from:    GoRunning,
				to:      GoNotExist,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "EvGoDestroySyscall",
				g:       6,
				stack:   NoStack,
				stG:     6,
				from:    GoSyscall,
				to:      GoNotExist,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "EvGoStop",
				g:       7,
				stack:   stk1,
				stG:     7,
				from:    GoRunning,
				to:      GoRunnable,
				reason:  "preempted",
				stStack: stk1,
				valid:   true,
			},
			{
				name:    "EvGoBlock",
				g:       8,
				stack:   stk1,
				stG:     8,
				from:    GoRunning,
				to:      GoWaiting,
				reason:  "blocked",
				stStack: stk1,
				valid:   true,
			},
			{
				name:    "EvGoUnblock",
				g:       9,
				stack:   stk1,
				stG:     anotherG,
				from:    GoWaiting,
				to:      GoRunnable,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			// N.b. EvGoUnblock, EvGoSwitch and EvGoSwitchDestroy cannot be
			// distinguished from each other in Event form, so MakeEvent only
			// produces EvGoUnblock events for Waiting -> Runnable transitions.
			{
				name:    "EvGoSyscallBegin",
				g:       10,
				stack:   stk1,
				stG:     10,
				from:    GoRunning,
				to:      GoSyscall,
				reason:  "",
				stStack: stk1,
				valid:   true,
			},
			{
				name:    "EvGoSyscallEnd",
				g:       11,
				stack:   NoStack,
				stG:     11,
				from:    GoSyscall,
				to:      GoRunning,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "EvGoSyscallEndBlocked",
				g:       12,
				stack:   NoStack,
				stG:     12,
				from:    GoSyscall,
				to:      GoRunnable,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			// TODO(felixge): Use coverage testsing to check if we need all these GoStatus/GoStatusStack cases
			{
				name:    "GoStatus Undetermined->Waiting",
				g:       anotherG,
				stack:   NoStack,
				stG:     13,
				from:    GoUndetermined,
				to:      GoWaiting,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "GoStatus Undetermined->Running",
				g:       anotherG,
				stack:   NoStack,
				stG:     14,
				from:    GoUndetermined,
				to:      GoRunning,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "GoStatusStack Undetermined->Waiting",
				g:       anotherG,
				stack:   stk1,
				stG:     15,
				from:    GoUndetermined,
				to:      GoWaiting,
				reason:  "",
				stStack: stk1,
				valid:   true,
			},
			{
				name:    "GoStatusStack Undetermined->Runnable",
				g:       anotherG,
				stack:   stk1,
				stG:     16,
				from:    GoUndetermined,
				to:      GoRunnable,
				reason:  "",
				stStack: stk1,
				valid:   true,
			},
			{
				name:    "GoStatus Runnable->Runnable",
				g:       anotherG,
				stack:   NoStack,
				stG:     17,
				from:    GoRunnable,
				to:      GoRunnable,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "GoStatus Runnable->Running",
				g:       anotherG,
				stack:   NoStack,
				stG:     18,
				from:    GoRunnable,
				to:      GoRunning,
				reason:  "",
				stStack: NoStack,
				valid:   true,
			},
			{
				name:    "invalid NotExits->NotExists",
				g:       anotherG,
				stack:   stk1,
				stG:     18,
				from:    GoNotExist,
				to:      GoNotExist,
				reason:  "",
				stStack: NoStack,
				valid:   false,
			},
			{
				name:    "invalid Running->Undetermined",
				g:       anotherG,
				stack:   stk1,
				stG:     19,
				from:    GoRunning,
				to:      GoUndetermined,
				reason:  "",
				stStack: NoStack,
				valid:   false,
			},
		}

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				st := MakeGoStateTransition(test.stG, test.from, test.to)
				st.Stack = test.stStack
				st.Reason = test.reason
				ev, err := MakeEvent(EventConfig[StateTransition]{
					Kind:      EventStateTransition,
					Time:      Time(42 + i),
					Goroutine: test.g,
					Stack:     test.stack,
					Details:   st,
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkStack(t, ev.Stack(), test.stack, schedStack)
				if ev.Goroutine() != test.g {
					t.Errorf("expected goroutine to be %d, got %d", test.g, ev.Goroutine())
				}
				got := ev.StateTransition()
				if got.Resource.Goroutine() != test.stG {
					t.Errorf("expected resource to be %d, got %d", test.stG, got.Resource.Goroutine())
				}
				from, to := got.Goroutine()
				if from != test.from {
					t.Errorf("from got=%s want=%s", from, test.from)
				}
				if to != test.to {
					t.Errorf("to got=%s want=%s", to, test.to)
				}
				if got.Reason != test.reason {
					t.Errorf("expected reason to be %s, got %s", test.reason, got.Reason)
				}
				checkStack(t, got.Stack, test.stStack, stStack)
				checkTime(t, ev, Time(42+i))
			})
		}
	})

	t.Run("ProcTransition", func(t *testing.T) {
		tests := []struct {
			name      string
			proc      ProcID
			schedProc ProcID
			from      ProcState
			to        ProcState
			valid     bool
		}{
			{name: "ProcStart", proc: 1, schedProc: 99, from: ProcIdle, to: ProcRunning, valid: true},
			{name: "ProcStop", proc: 2, schedProc: 2, from: ProcRunning, to: ProcIdle, valid: true},
			{name: "ProcSteal", proc: 3, schedProc: 99, from: ProcRunning, to: ProcIdle, valid: true},
			{name: "ProcSteal lost info", proc: 4, schedProc: 99, from: ProcIdle, to: ProcIdle, valid: true},
			{name: "ProcStatus", proc: 5, schedProc: 99, from: ProcUndetermined, to: ProcRunning, valid: true},
		}
		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				st := MakeProcStateTransition(test.proc, test.from, test.to)
				ev, err := MakeEvent(EventConfig[StateTransition]{
					Kind:    EventStateTransition,
					Time:    Time(42 + i),
					Proc:    test.schedProc,
					Details: st,
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				gotSt := ev.StateTransition()
				from, to := gotSt.Proc()
				if from != test.from {
					t.Errorf("from got=%s want=%s", from, test.from)
				}
				if to != test.to {
					t.Errorf("to got=%s want=%s", to, test.to)
				}
				if ev.Proc() != test.schedProc {
					t.Errorf("expected proc to be %d, got %d", test.schedProc, ev.Proc())
				}
				if gotSt.Resource.Proc() != test.proc {
					t.Errorf("expected resource to be %d, got %d", test.proc, gotSt.Resource.Proc())
				}
			})
		}
	})

	t.Run("Sync", func(t *testing.T) {
		tests := []struct {
			name    string
			kind    EventKind
			n       int
			clock   *ClockSnapshot
			batches map[string][]ExperimentalBatch
			valid   bool
		}{
			{
				name:  "invalid kind",
				n:     1,
				valid: false,
			},
			{
				name:    "N",
				kind:    EventSync,
				n:       1,
				batches: map[string][]ExperimentalBatch{},
				valid:   true,
			},
			{
				name:    "N+ClockSnapshot",
				kind:    EventSync,
				n:       1,
				batches: map[string][]ExperimentalBatch{},
				clock: &ClockSnapshot{
					Trace: 1,
					Wall:  time.Unix(59, 123456789),
					Mono:  2,
				},
				valid: true,
			},
			{
				name: "N+Batches",
				kind: EventSync,
				n:    1,
				batches: map[string][]ExperimentalBatch{
					"AllocFree": {{Thread: 1, Data: []byte{1, 2, 3}}},
				},
				valid: true,
			},
			{
				name: "unknown experiment",
				kind: EventSync,
				n:    1,
				batches: map[string][]ExperimentalBatch{
					"does-not-exist": {{Thread: 1, Data: []byte{1, 2, 3}}},
				},
				valid: false,
			},
		}
		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Sync]{
					Kind:    test.kind,
					Time:    Time(42 + i),
					Details: Sync{N: test.n, ClockSnapshot: test.clock, ExperimentalBatches: test.batches},
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				got := ev.Sync()
				checkTime(t, ev, Time(42+i))
				if got.N != test.n {
					t.Errorf("expected N to be %d, got %d", test.n, got.N)
				}
				if test.clock != nil && got.ClockSnapshot == nil {
					t.Fatalf("expected ClockSnapshot to be non-nil")
				} else if test.clock == nil && got.ClockSnapshot != nil {
					t.Fatalf("expected ClockSnapshot to be nil")
				} else if test.clock != nil && got.ClockSnapshot != nil {
					if got.ClockSnapshot.Trace != test.clock.Trace {
						t.Errorf("expected ClockSnapshot.Trace to be %d, got %d", test.clock.Trace, got.ClockSnapshot.Trace)
					}
					if !got.ClockSnapshot.Wall.Equal(test.clock.Wall) {
						t.Errorf("expected ClockSnapshot.Wall to be %s, got %s", test.clock.Wall, got.ClockSnapshot.Wall)
					}
					if got.ClockSnapshot.Mono != test.clock.Mono {
						t.Errorf("expected ClockSnapshot.Mono to be %d, got %d", test.clock.Mono, got.ClockSnapshot.Mono)
					}
				}
				if !reflect.DeepEqual(got.ExperimentalBatches, test.batches) {
					t.Errorf("expected ExperimentalBatches to be %#v, got %#v", test.batches, got.ExperimentalBatches)
				}
			})
		}
	})

	t.Run("Task", func(t *testing.T) {
		tests := []struct {
			name   string
			kind   EventKind
			id     TaskID
			parent TaskID
			typ    string
			valid  bool
		}{
			{name: "no task", kind: EventTaskBegin, id: NoTask, parent: 1, typ: "type-0", valid: false},
			{name: "invalid kind", kind: EventMetric, id: 1, parent: 2, typ: "type-1", valid: false},
			{name: "EvUserTaskBegin", kind: EventTaskBegin, id: 2, parent: 3, typ: "type-2", valid: true},
			{name: "EvUserTaskEnd", kind: EventTaskEnd, id: 3, parent: 4, typ: "type-3", valid: true},
			{name: "no parent", kind: EventTaskBegin, id: 4, parent: NoTask, typ: "type-4", valid: true},
		}

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Task]{
					Kind:    test.kind,
					Time:    Time(42 + i),
					Details: Task{ID: test.id, Parent: test.parent, Type: test.typ},
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				got := ev.Task()
				if got.ID != test.id {
					t.Errorf("expected ID to be %d, got %d", test.id, got.ID)
				}
				if got.Parent != test.parent {
					t.Errorf("expected Parent to be %d, got %d", test.parent, got.Parent)
				}
				if got.Type != test.typ {
					t.Errorf("expected Type to be %s, got %s", test.typ, got.Type)
				}
			})
		}
	})

	t.Run("Region", func(t *testing.T) {
		tests := []struct {
			name  string
			kind  EventKind
			task  TaskID
			typ   string
			valid bool
		}{
			{name: "invalid kind", kind: EventMetric, task: 1, typ: "type-1", valid: false},
			{name: "EvUserRegionBegin", kind: EventRegionBegin, task: 2, typ: "type-2", valid: true},
			{name: "EvUserRegionEnd", kind: EventRegionEnd, task: 3, typ: "type-3", valid: true},
		}

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Region]{
					Kind:    test.kind,
					Time:    Time(42 + i),
					Details: Region{Task: test.task, Type: test.typ},
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				got := ev.Region()
				if got.Task != test.task {
					t.Errorf("expected Task to be %d, got %d", test.task, got.Task)
				}
				if got.Type != test.typ {
					t.Errorf("expected Type to be %s, got %s", test.typ, got.Type)
				}
			})
		}
	})

	t.Run("Log", func(t *testing.T) {
		tests := []struct {
			name     string
			kind     EventKind
			task     TaskID
			category string
			message  string
			valid    bool
		}{
			{name: "invalid kind", kind: EventMetric, task: 1, category: "category-1", message: "message-1", valid: false},
			{name: "basic", kind: EventLog, task: 2, category: "category-2", message: "message-2", valid: true},
		}

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[Log]{
					Kind:    test.kind,
					Time:    Time(42 + i),
					Details: Log{Task: test.task, Category: test.category, Message: test.message},
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				got := ev.Log()
				if got.Task != test.task {
					t.Errorf("expected Task to be %d, got %d", test.task, got.Task)
				}
				if got.Category != test.category {
					t.Errorf("expected Category to be %s, got %s", test.category, got.Category)
				}
				if got.Message != test.message {
					t.Errorf("expected Message to be %s, got %s", test.message, got.Message)
				}
			})
		}

	})

	t.Run("StackSample", func(t *testing.T) {
		tests := []struct {
			name  string
			kind  EventKind
			stack Stack
			valid bool
		}{
			{name: "invalid kind", kind: EventMetric, stack: stk1, valid: false},
			{name: "basic", kind: EventStackSample, stack: stk1, valid: true},
		}

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				ev, err := MakeEvent(EventConfig[StackSample]{
					Kind:  test.kind,
					Time:  Time(42 + i),
					Stack: test.stack,
					// N.b. Details defaults to StackSample{}, so we can
					// omit it here.
				})
				if !checkValid(t, err, test.valid) {
					return
				}
				checkTime(t, ev, Time(42+i))
				got := ev.Stack()
				checkStack(t, got, test.stack, schedStack)
			})
		}

	})
}

func TestMakeStack(t *testing.T) {
	frames := []StackFrame{
		{PC: 1, Func: "foo", File: "foo.go", Line: 10},
		{PC: 2, Func: "bar", File: "bar.go", Line: 20},
	}
	got := slices.Collect(MakeStack(frames).Frames())
	if len(got) != len(frames) {
		t.Errorf("got=%d want=%d", len(got), len(frames))
	}
	for i := range got {
		if got[i] != frames[i] {
			t.Errorf("got=%v want=%v", got[i], frames[i])
		}
	}
}

func TestPanicEvent(t *testing.T) {
	// Use a sync event for this because it doesn't have any extra metadata.
	ev := syncEvent(nil, 0, 0)

	mustPanic(t, func() {
		_ = ev.Range()
	})
	mustPanic(t, func() {
		_ = ev.Metric()
	})
	mustPanic(t, func() {
		_ = ev.Log()
	})
	mustPanic(t, func() {
		_ = ev.Task()
	})
	mustPanic(t, func() {
		_ = ev.Region()
	})
	mustPanic(t, func() {
		_ = ev.Label()
	})
	mustPanic(t, func() {
		_ = ev.RangeAttributes()
	})
}

func mustPanic(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("failed to panic")
		}
	}()
	f()
}
