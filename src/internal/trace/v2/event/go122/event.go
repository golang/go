// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go122

import (
	"fmt"
	"internal/trace/v2/event"
)

const (
	EvNone event.Type = iota // unused

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
)

// EventString returns the name of a Go 1.22 event.
func EventString(typ event.Type) string {
	if int(typ) < len(specs) {
		return specs[typ].Name
	}
	return fmt.Sprintf("Invalid(%d)", typ)
}

func Specs() []event.Spec {
	return specs[:]
}

var specs = [...]event.Spec{
	// "Structural" Events.
	EvEventBatch: event.Spec{
		Name: "EventBatch",
		Args: []string{"gen", "m", "time", "size"},
	},
	EvStacks: event.Spec{
		Name: "Stacks",
	},
	EvStack: event.Spec{
		Name:    "Stack",
		Args:    []string{"id", "nframes"},
		IsStack: true,
	},
	EvStrings: event.Spec{
		Name: "Strings",
	},
	EvString: event.Spec{
		Name:    "String",
		Args:    []string{"id"},
		HasData: true,
	},
	EvCPUSamples: event.Spec{
		Name: "CPUSamples",
	},
	EvCPUSample: event.Spec{
		Name: "CPUSample",
		Args: []string{"time", "m", "p", "g", "stack"},
		// N.B. There's clearly a timestamp here, but these Events
		// are special in that they don't appear in the regular
		// M streams.
	},
	EvFrequency: event.Spec{
		Name: "Frequency",
		Args: []string{"freq"},
	},

	// "Timed" Events.
	EvProcsChange: event.Spec{
		Name:         "ProcsChange",
		Args:         []string{"dt", "procs_value", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvProcStart: event.Spec{
		Name:         "ProcStart",
		Args:         []string{"dt", "p", "p_seq"},
		IsTimedEvent: true,
	},
	EvProcStop: event.Spec{
		Name:         "ProcStop",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvProcSteal: event.Spec{
		Name:         "ProcSteal",
		Args:         []string{"dt", "p", "p_seq", "m"},
		IsTimedEvent: true,
	},
	EvProcStatus: event.Spec{
		Name:         "ProcStatus",
		Args:         []string{"dt", "p", "pstatus"},
		IsTimedEvent: true,
	},
	EvGoCreate: event.Spec{
		Name:         "GoCreate",
		Args:         []string{"dt", "new_g", "new_stack", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3, 2},
	},
	EvGoCreateSyscall: event.Spec{
		Name:         "GoCreateSyscall",
		Args:         []string{"dt", "new_g"},
		IsTimedEvent: true,
	},
	EvGoStart: event.Spec{
		Name:         "GoStart",
		Args:         []string{"dt", "g", "g_seq"},
		IsTimedEvent: true,
	},
	EvGoDestroy: event.Spec{
		Name:         "GoDestroy",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvGoDestroySyscall: event.Spec{
		Name:         "GoDestroySyscall",
		Args:         []string{"dt"},
		IsTimedEvent: true,
	},
	EvGoStop: event.Spec{
		Name:         "GoStop",
		Args:         []string{"dt", "reason_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvGoBlock: event.Spec{
		Name:         "GoBlock",
		Args:         []string{"dt", "reason_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvGoUnblock: event.Spec{
		Name:         "GoUnblock",
		Args:         []string{"dt", "g", "g_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3},
	},
	EvGoSyscallBegin: event.Spec{
		Name:         "GoSyscallBegin",
		Args:         []string{"dt", "p_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvGoSyscallEnd: event.Spec{
		Name:         "GoSyscallEnd",
		Args:         []string{"dt"},
		StartEv:      EvGoSyscallBegin,
		IsTimedEvent: true,
	},
	EvGoSyscallEndBlocked: event.Spec{
		Name:         "GoSyscallEndBlocked",
		Args:         []string{"dt"},
		StartEv:      EvGoSyscallBegin,
		IsTimedEvent: true,
	},
	EvGoStatus: event.Spec{
		Name:         "GoStatus",
		Args:         []string{"dt", "g", "m", "gstatus"},
		IsTimedEvent: true,
	},
	EvSTWBegin: event.Spec{
		Name:         "STWBegin",
		Args:         []string{"dt", "kind_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
		StringIDs:    []int{1},
	},
	EvSTWEnd: event.Spec{
		Name:         "STWEnd",
		Args:         []string{"dt"},
		StartEv:      EvSTWBegin,
		IsTimedEvent: true,
	},
	EvGCActive: event.Spec{
		Name:         "GCActive",
		Args:         []string{"dt", "gc_seq"},
		IsTimedEvent: true,
		StartEv:      EvGCBegin,
	},
	EvGCBegin: event.Spec{
		Name:         "GCBegin",
		Args:         []string{"dt", "gc_seq", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvGCEnd: event.Spec{
		Name:         "GCEnd",
		Args:         []string{"dt", "gc_seq"},
		StartEv:      EvGCBegin,
		IsTimedEvent: true,
	},
	EvGCSweepActive: event.Spec{
		Name:         "GCSweepActive",
		Args:         []string{"dt", "p"},
		StartEv:      EvGCSweepBegin,
		IsTimedEvent: true,
	},
	EvGCSweepBegin: event.Spec{
		Name:         "GCSweepBegin",
		Args:         []string{"dt", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{1},
	},
	EvGCSweepEnd: event.Spec{
		Name:         "GCSweepEnd",
		Args:         []string{"dt", "swept_value", "reclaimed_value"},
		StartEv:      EvGCSweepBegin,
		IsTimedEvent: true,
	},
	EvGCMarkAssistActive: event.Spec{
		Name:         "GCMarkAssistActive",
		Args:         []string{"dt", "g"},
		StartEv:      EvGCMarkAssistBegin,
		IsTimedEvent: true,
	},
	EvGCMarkAssistBegin: event.Spec{
		Name:         "GCMarkAssistBegin",
		Args:         []string{"dt", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{1},
	},
	EvGCMarkAssistEnd: event.Spec{
		Name:         "GCMarkAssistEnd",
		Args:         []string{"dt"},
		StartEv:      EvGCMarkAssistBegin,
		IsTimedEvent: true,
	},
	EvHeapAlloc: event.Spec{
		Name:         "HeapAlloc",
		Args:         []string{"dt", "heapalloc_value"},
		IsTimedEvent: true,
	},
	EvHeapGoal: event.Spec{
		Name:         "HeapGoal",
		Args:         []string{"dt", "heapgoal_value"},
		IsTimedEvent: true,
	},
	EvGoLabel: event.Spec{
		Name:         "GoLabel",
		Args:         []string{"dt", "label_string"},
		IsTimedEvent: true,
		StringIDs:    []int{1},
	},
	EvUserTaskBegin: event.Spec{
		Name:         "UserTaskBegin",
		Args:         []string{"dt", "task", "parent_task", "name_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{4},
		StringIDs:    []int{3},
	},
	EvUserTaskEnd: event.Spec{
		Name:         "UserTaskEnd",
		Args:         []string{"dt", "task", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{2},
	},
	EvUserRegionBegin: event.Spec{
		Name:         "UserRegionBegin",
		Args:         []string{"dt", "task", "name_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{3},
		StringIDs:    []int{2},
	},
	EvUserRegionEnd: event.Spec{
		Name:         "UserRegionEnd",
		Args:         []string{"dt", "task", "name_string", "stack"},
		StartEv:      EvUserRegionBegin,
		IsTimedEvent: true,
		StackIDs:     []int{3},
		StringIDs:    []int{2},
	},
	EvUserLog: event.Spec{
		Name:         "UserLog",
		Args:         []string{"dt", "task", "key_string", "value_string", "stack"},
		IsTimedEvent: true,
		StackIDs:     []int{4},
		StringIDs:    []int{2, 3},
	},
}

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

type ProcStatus uint8

const (
	ProcBad ProcStatus = iota
	ProcRunning
	ProcIdle
	ProcSyscall
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
	// Various format-specific constants.
	MaxBatchSize      = 64 << 10
	MaxFramesPerStack = 128
	MaxStringSize     = 1 << 10
)
