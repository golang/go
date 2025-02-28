// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tracev2

// Event types in the trace, args are given in square brackets.
//
// Naming scheme:
//   - Time range event pairs have suffixes "Begin" and "End".
//   - "Start", "Stop", "Create", "Destroy", "Block", "Unblock"
//     are suffixes reserved for scheduling resources.
//
// NOTE: If you add an event type, make sure you also update all
// tables in this file!
const (
	EvNone EventType = iota // unused

	// Structural events.
	EvEventBatch // start of per-M batch of events [generation, M ID, timestamp, batch length]
	EvStacks     // start of a section of the stack table [...EvStack]
	EvStack      // stack table entry [ID, ...{PC, func string ID, file string ID, line #}]
	EvStrings    // start of a section of the string dictionary [...EvString]
	EvString     // string dictionary entry [ID, length, string]
	EvCPUSamples // start of a section of CPU samples [...EvCPUSample]
	EvCPUSample  // CPU profiling sample [timestamp, M ID, P ID, goroutine ID, stack ID]
	EvFrequency  // timestamp units per sec [freq]

	// Procs.
	EvProcsChange // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack ID]
	EvProcStart   // start of P [timestamp, P ID, P seq]
	EvProcStop    // stop of P [timestamp]
	EvProcSteal   // P was stolen [timestamp, P ID, P seq, M ID]
	EvProcStatus  // P status at the start of a generation [timestamp, P ID, status]

	// Goroutines.
	EvGoCreate            // goroutine creation [timestamp, new goroutine ID, new stack ID, stack ID]
	EvGoCreateSyscall     // goroutine appears in syscall (cgo callback) [timestamp, new goroutine ID]
	EvGoStart             // goroutine starts running [timestamp, goroutine ID, goroutine seq]
	EvGoDestroy           // goroutine ends [timestamp]
	EvGoDestroySyscall    // goroutine ends in syscall (cgo callback) [timestamp]
	EvGoStop              // goroutine yields its time, but is runnable [timestamp, reason, stack ID]
	EvGoBlock             // goroutine blocks [timestamp, reason, stack ID]
	EvGoUnblock           // goroutine is unblocked [timestamp, goroutine ID, goroutine seq, stack ID]
	EvGoSyscallBegin      // syscall enter [timestamp, P seq, stack ID]
	EvGoSyscallEnd        // syscall exit [timestamp]
	EvGoSyscallEndBlocked // syscall exit and it blocked at some point [timestamp]
	EvGoStatus            // goroutine status at the start of a generation [timestamp, goroutine ID, thread ID, status]

	// STW.
	EvSTWBegin // STW start [timestamp, kind]
	EvSTWEnd   // STW done [timestamp]

	// GC events.
	EvGCActive           // GC active [timestamp, seq]
	EvGCBegin            // GC start [timestamp, seq, stack ID]
	EvGCEnd              // GC done [timestamp, seq]
	EvGCSweepActive      // GC sweep active [timestamp, P ID]
	EvGCSweepBegin       // GC sweep start [timestamp, stack ID]
	EvGCSweepEnd         // GC sweep done [timestamp, swept bytes, reclaimed bytes]
	EvGCMarkAssistActive // GC mark assist active [timestamp, goroutine ID]
	EvGCMarkAssistBegin  // GC mark assist start [timestamp, stack ID]
	EvGCMarkAssistEnd    // GC mark assist done [timestamp]
	EvHeapAlloc          // gcController.heapLive change [timestamp, heap alloc in bytes]
	EvHeapGoal           // gcController.heapGoal() change [timestamp, heap goal in bytes]

	// Annotations.
	EvGoLabel         // apply string label to current running goroutine [timestamp, label string ID]
	EvUserTaskBegin   // trace.NewTask [timestamp, internal task ID, internal parent task ID, name string ID, stack ID]
	EvUserTaskEnd     // end of a task [timestamp, internal task ID, stack ID]
	EvUserRegionBegin // trace.{Start,With}Region [timestamp, internal task ID, name string ID, stack ID]
	EvUserRegionEnd   // trace.{End,With}Region [timestamp, internal task ID, name string ID, stack ID]
	EvUserLog         // trace.Log [timestamp, internal task ID, key string ID, value string ID, stack]

	// Coroutines. Added in Go 1.23.
	EvGoSwitch        // goroutine switch (coroswitch) [timestamp, goroutine ID, goroutine seq]
	EvGoSwitchDestroy // goroutine switch and destroy [timestamp, goroutine ID, goroutine seq]
	EvGoCreateBlocked // goroutine creation (starts blocked) [timestamp, new goroutine ID, new stack ID, stack ID]

	// GoStatus with stack. Added in Go 1.23.
	EvGoStatusStack // goroutine status at the start of a generation, with a stack [timestamp, goroutine ID, M ID, status, stack ID]

	// Batch event for an experimental batch with a custom format. Added in Go 1.23.
	EvExperimentalBatch // start of extra data [experiment ID, generation, M ID, timestamp, batch length, batch data...]

	// Sync batch. Added in Go 1.25. Previously a lone EvFrequency event.
	EvSync          // start of a sync batch [...EvFrequency|EvClockSnapshot]
	EvClockSnapshot // snapshot of trace, mono and wall clocks [timestamp, mono, sec, nsec]

	NumEvents
)

func (ev EventType) Experimental() bool {
	return ev > MaxEvent && ev < MaxExperimentalEvent
}

// Experiments.
const (
	// AllocFree is the alloc-free events experiment.
	AllocFree Experiment = 1 + iota

	NumExperiments
)

func Experiments() []string {
	return experiments[:]
}

var experiments = [...]string{
	NoExperiment: "None",
	AllocFree:    "AllocFree",
}

// Experimental events.
const (
	MaxEvent EventType = 127 + iota

	// Experimental events for AllocFree.

	// Experimental heap span events. Added in Go 1.23.
	EvSpan      // heap span exists [timestamp, id, npages, type/class]
	EvSpanAlloc // heap span alloc [timestamp, id, npages, type/class]
	EvSpanFree  // heap span free [timestamp, id]

	// Experimental heap object events. Added in Go 1.23.
	EvHeapObject      // heap object exists [timestamp, id, type]
	EvHeapObjectAlloc // heap object alloc [timestamp, id, type]
	EvHeapObjectFree  // heap object free [timestamp, id]

	// Experimental goroutine stack events. Added in Go 1.23.
	EvGoroutineStack      // stack exists [timestamp, id, order]
	EvGoroutineStackAlloc // stack alloc [timestamp, id, order]
	EvGoroutineStackFree  // stack free [timestamp, id]

	MaxExperimentalEvent
)

const NumExperimentalEvents = MaxExperimentalEvent - MaxEvent

// MaxTimedEventArgs is the maximum number of arguments for timed events.
const MaxTimedEventArgs = 5

func Specs() []EventSpec {
	return specs[:]
}

var specs = [...]EventSpec{
	// "Structural" Events.
	EvEventBatch: {
		Name: "EventBatch",
		Args: []string{"gen", "m", "time", "size"},
	},
	EvStacks: {
		Name: "Stacks",
	},
	EvStack: {
		Name:    "Stack",
		Args:    []string{"id", "nframes"},
		IsStack: true,
	},
	EvStrings: {
		Name: "Strings",
	},
	EvString: {
		Name:    "String",
		Args:    []string{"id"},
		HasData: true,
	},
	EvCPUSamples: {
		Name: "CPUSamples",
	},
	EvCPUSample: {
		Name: "CPUSample",
		Args: []string{"time", "m", "p", "g", "stack"},
		// N.B. There's clearly a timestamp here, but these Events
		// are special in that they don't appear in the regular
		// M streams.
		StackIDs: []int{4},
	},
	EvFrequency: {
		Name: "Frequency",
		Args: []string{"freq"},
	},
	EvExperimentalBatch: {
		Name:    "ExperimentalBatch",
		Args:    []string{"exp", "gen", "m", "time"},
		HasData: true, // Easier to represent for raw readers.
	},
	EvSync: {
		Name: "Sync",
	},

	// "Timed" Events.
	EvProcsChange: {
		Name:         "ProcsChange",
		Args:         []string{"dt", "procs_value", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvProcStart: {
		Name:         "ProcStart",
		Args:         []string{"dt", "p", "p_seq"},
		IsTimedEvent: true,
	},
	EvProcStop: {
		Name:         "ProcStop",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvProcSteal: {
		Name:         "ProcSteal",
		Args:         []string{"dt", "p", "p_seq", "m"},
		IsTimedEvent: true,
	},
	EvProcStatus: {
		Name:         "ProcStatus",
		Args:         []string{"dt", "p", "pstatus"},
		IsTimedEvent: true,
	},
	EvGoCreate: {
		Name:         "GoCreate",
		Args:         []string{"dt", "new_g", "new_stack", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3, 2},
	},
	EvGoCreateSyscall: {
		Name:         "GoCreateSyscall",
		Args:         []string{"dt", "new_g"},
		IsTimedEvent: true,
	},
	EvGoStart: {
		Name:         "GoStart",
		Args:         []string{"dt", "g", "g_seq"},
		IsTimedEvent: true,
	},
	EvGoDestroy: {
		Name:         "GoDestroy",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvGoDestroySyscall: {
		Name:         "GoDestroySyscall",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvGoStop: {
		Name:         "GoStop",
		Args:         []string{"dt", "reason_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvGoBlock: {
		Name:         "GoBlock",
		Args:         []string{"dt", "reason_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvGoUnblock: {
		Name:         "GoUnblock",
		Args:         []string{"dt", "g", "g_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3},
	},
	EvGoSyscallBegin: {
		Name:         "GoSyscallBegin",
		Args:         []string{"dt", "p_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvGoSyscallEnd: {
		Name:         "GoSyscallEnd",
		Args:         []string{"dt"},
		StartEv:      EvGoSyscallBegin,
		IsTimedEvent: true,
	},
	EvGoSyscallEndBlocked: {
		Name:         "GoSyscallEndBlocked",
		Args:         []string{"dt"},
		StartEv:      EvGoSyscallBegin,
		IsTimedEvent: true,
	},
	EvGoStatus: {
		Name:         "GoStatus",
		Args:         []string{"dt", "g", "m", "gstatus"},
		IsTimedEvent: true,
	},
	EvSTWBegin: {
		Name:         "STWBegin",
		Args:         []string{"dt", "kind_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvSTWEnd: {
		Name:         "STWEnd",
		Args:         []string{"dt"},
		StartEv:      EvSTWBegin,
		IsTimedEvent: true,
	},
	EvGCActive: {
		Name:         "GCActive",
		Args:         []string{"dt", "gc_seq"},
		IsTimedEvent: true,
		StartEv:      EvGCBegin,
	},
	EvGCBegin: {
		Name:         "GCBegin",
		Args:         []string{"dt", "gc_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvGCEnd: {
		Name:         "GCEnd",
		Args:         []string{"dt", "gc_seq"},
		StartEv:      EvGCBegin,
		IsTimedEvent: true,
	},
	EvGCSweepActive: {
		Name:         "GCSweepActive",
		Args:         []string{"dt", "p"},
		StartEv:      EvGCSweepBegin,
		IsTimedEvent: true,
	},
	EvGCSweepBegin: {
		Name:         "GCSweepBegin",
		Args:         []string{"dt", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{1},
	},
	EvGCSweepEnd: {
		Name:         "GCSweepEnd",
		Args:         []string{"dt", "swept_value", "reclaimed_value"},
		StartEv:      EvGCSweepBegin,
		IsTimedEvent: true,
	},
	EvGCMarkAssistActive: {
		Name:         "GCMarkAssistActive",
		Args:         []string{"dt", "g"},
		StartEv:      EvGCMarkAssistBegin,
		IsTimedEvent: true,
	},
	EvGCMarkAssistBegin: {
		Name:         "GCMarkAssistBegin",
		Args:         []string{"dt", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{1},
	},
	EvGCMarkAssistEnd: {
		Name:         "GCMarkAssistEnd",
		Args:         []string{"dt"},
		StartEv:      EvGCMarkAssistBegin,
		IsTimedEvent: true,
	},
	EvHeapAlloc: {
		Name:         "HeapAlloc",
		Args:         []string{"dt", "heapalloc_value"},
		IsTimedEvent: true,
	},
	EvHeapGoal: {
		Name:         "HeapGoal",
		Args:         []string{"dt", "heapgoal_value"},
		IsTimedEvent: true,
	},
	EvGoLabel: {
		Name:         "GoLabel",
		Args:         []string{"dt", "label_string"},
		IsTimedEvent: true,
		StringIDs:    []int{1},
	},
	EvUserTaskBegin: {
		Name:         "UserTaskBegin",
		Args:         []string{"dt", "task", "parent_task", "name_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{4},
		StringIDs:    []int{3},
	},
	EvUserTaskEnd: {
		Name:         "UserTaskEnd",
		Args:         []string{"dt", "task", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvUserRegionBegin: {
		Name:         "UserRegionBegin",
		Args:         []string{"dt", "task", "name_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3},
		StringIDs:    []int{2},
	},
	EvUserRegionEnd: {
		Name:         "UserRegionEnd",
		Args:         []string{"dt", "task", "name_string", "stack"},
		StartEv:      EvUserRegionBegin,
		IsTimedEvent: true,
		StackIDs:     []int{3},
		StringIDs:    []int{2},
	},
	EvUserLog: {
		Name:         "UserLog",
		Args:         []string{"dt", "task", "key_string", "value_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{4},
		StringIDs:    []int{2, 3},
	},
	EvGoSwitch: {
		Name:         "GoSwitch",
		Args:         []string{"dt", "g", "g_seq"},
		IsTimedEvent: true,
	},
	EvGoSwitchDestroy: {
		Name:         "GoSwitchDestroy",
		Args:         []string{"dt", "g", "g_seq"},
		IsTimedEvent: true,
	},
	EvGoCreateBlocked: {
		Name:         "GoCreateBlocked",
		Args:         []string{"dt", "new_g", "new_stack", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3, 2},
	},
	EvGoStatusStack: {
		Name:         "GoStatusStack",
		Args:         []string{"dt", "g", "m", "gstatus", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{4},
	},
	EvClockSnapshot: {
		Name:         "ClockSnapshot",
		Args:         []string{"dt", "mono", "sec", "nsec"},
		IsTimedEvent: true,
	},

	// Experimental events.

	EvSpan: {
		Name:         "Span",
		Args:         []string{"dt", "id", "npages_value", "kindclass"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvSpanAlloc: {
		Name:         "SpanAlloc",
		Args:         []string{"dt", "id", "npages_value", "kindclass"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvSpanFree: {
		Name:         "SpanFree",
		Args:         []string{"dt", "id"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvHeapObject: {
		Name:         "HeapObject",
		Args:         []string{"dt", "id", "type"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvHeapObjectAlloc: {
		Name:         "HeapObjectAlloc",
		Args:         []string{"dt", "id", "type"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvHeapObjectFree: {
		Name:         "HeapObjectFree",
		Args:         []string{"dt", "id"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvGoroutineStack: {
		Name:         "GoroutineStack",
		Args:         []string{"dt", "id", "order"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvGoroutineStackAlloc: {
		Name:         "GoroutineStackAlloc",
		Args:         []string{"dt", "id", "order"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
	EvGoroutineStackFree: {
		Name:         "GoroutineStackFree",
		Args:         []string{"dt", "id"},
		IsTimedEvent: true,
		Experiment:   AllocFree,
	},
}

// GoStatus is the status of a goroutine.
//
// They correspond directly to the various goroutine states.
type GoStatus uint8

const (
	GoBad GoStatus = iota
	GoRunnable
	GoRunning
	GoSyscall
	GoWaiting
)

func (s GoStatus) String() string {
	switch s {
	case GoRunnable:
		return "Runnable"
	case GoRunning:
		return "Running"
	case GoSyscall:
		return "Syscall"
	case GoWaiting:
		return "Waiting"
	}
	return "Bad"
}

// ProcStatus is the status of a P.
//
// They mostly correspond to the various P states.
type ProcStatus uint8

const (
	ProcBad ProcStatus = iota
	ProcRunning
	ProcIdle
	ProcSyscall

	// ProcSyscallAbandoned is a special case of
	// ProcSyscall. It's used in the very specific case
	// where the first a P is mentioned in a generation is
	// part of a ProcSteal event. If that's the first time
	// it's mentioned, then there's no GoSyscallBegin to
	// connect the P stealing back to at that point. This
	// special state indicates this to the parser, so it
	// doesn't try to find a GoSyscallEndBlocked that
	// corresponds with the ProcSteal.
	ProcSyscallAbandoned
)

func (s ProcStatus) String() string {
	switch s {
	case ProcRunning:
		return "Running"
	case ProcIdle:
		return "Idle"
	case ProcSyscall:
		return "Syscall"
	}
	return "Bad"
}

const (
	// MaxBatchSize sets the maximum size that a batch can be.
	//
	// Directly controls the trace batch size in the runtime.
	//
	// NOTE: If this number decreases, the trace format version must change.
	MaxBatchSize = 64 << 10

	// Maximum number of PCs in a single stack trace.
	//
	// Since events contain only stack ID rather than whole stack trace,
	// we can allow quite large values here.
	//
	// Directly controls the maximum number of frames per stack
	// in the runtime.
	//
	// NOTE: If this number decreases, the trace format version must change.
	MaxFramesPerStack = 128

	// MaxEventTrailerDataSize controls the amount of trailer data that
	// an event can have in bytes. Must be smaller than MaxBatchSize.
	// Controls the maximum string size in the trace.
	//
	// Directly controls the maximum such value in the runtime.
	//
	// NOTE: If this number decreases, the trace format version must change.
	MaxEventTrailerDataSize = 1 << 10
)
