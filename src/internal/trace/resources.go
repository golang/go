// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "fmt"

// ThreadID is the runtime-internal M structure's ID. This is unique
// for each OS thread.
type ThreadID int64

// NoThread indicates that the relevant events don't correspond to any
// thread in particular.
const NoThread = ThreadID(-1)

// ProcID is the runtime-internal G structure's id field. This is unique
// for each P.
type ProcID int64

// NoProc indicates that the relevant events don't correspond to any
// P in particular.
const NoProc = ProcID(-1)

// GoID is the runtime-internal G structure's goid field. This is unique
// for each goroutine.
type GoID int64

// NoGoroutine indicates that the relevant events don't correspond to any
// goroutine in particular.
const NoGoroutine = GoID(-1)

// GoState represents the state of a goroutine.
//
// New GoStates may be added in the future. Users of this type must be robust
// to that possibility.
type GoState uint8

const (
	GoUndetermined GoState = iota // No information is known about the goroutine.
	GoNotExist                    // Goroutine does not exist.
	GoRunnable                    // Goroutine is runnable but not running.
	GoRunning                     // Goroutine is running.
	GoWaiting                     // Goroutine is waiting on something to happen.
	GoSyscall                     // Goroutine is in a system call.
)

// Executing returns true if the state indicates that the goroutine is executing
// and bound to its thread.
func (s GoState) Executing() bool {
	return s == GoRunning || s == GoSyscall
}

// String returns a human-readable representation of a GoState.
//
// The format of the returned string is for debugging purposes and is subject to change.
func (s GoState) String() string {
	switch s {
	case GoUndetermined:
		return "Undetermined"
	case GoNotExist:
		return "NotExist"
	case GoRunnable:
		return "Runnable"
	case GoRunning:
		return "Running"
	case GoWaiting:
		return "Waiting"
	case GoSyscall:
		return "Syscall"
	}
	return "Bad"
}

// ProcState represents the state of a proc.
//
// New ProcStates may be added in the future. Users of this type must be robust
// to that possibility.
type ProcState uint8

const (
	ProcUndetermined ProcState = iota // No information is known about the proc.
	ProcNotExist                      // Proc does not exist.
	ProcRunning                       // Proc is running.
	ProcIdle                          // Proc is idle.
)

// Executing returns true if the state indicates that the proc is executing
// and bound to its thread.
func (s ProcState) Executing() bool {
	return s == ProcRunning
}

// String returns a human-readable representation of a ProcState.
//
// The format of the returned string is for debugging purposes and is subject to change.
func (s ProcState) String() string {
	switch s {
	case ProcUndetermined:
		return "Undetermined"
	case ProcNotExist:
		return "NotExist"
	case ProcRunning:
		return "Running"
	case ProcIdle:
		return "Idle"
	}
	return "Bad"
}

// ResourceKind indicates a kind of resource that has a state machine.
//
// New ResourceKinds may be added in the future. Users of this type must be robust
// to that possibility.
type ResourceKind uint8

const (
	ResourceNone      ResourceKind = iota // No resource.
	ResourceGoroutine                     // Goroutine.
	ResourceProc                          // Proc.
	ResourceThread                        // Thread.
)

// String returns a human-readable representation of a ResourceKind.
//
// The format of the returned string is for debugging purposes and is subject to change.
func (r ResourceKind) String() string {
	switch r {
	case ResourceNone:
		return "None"
	case ResourceGoroutine:
		return "Goroutine"
	case ResourceProc:
		return "Proc"
	case ResourceThread:
		return "Thread"
	}
	return "Bad"
}

// ResourceID represents a generic resource ID.
type ResourceID struct {
	// Kind is the kind of resource this ID is for.
	Kind ResourceKind
	id   int64
}

// MakeResourceID creates a general resource ID from a specific resource's ID.
func MakeResourceID[T interface{ GoID | ProcID | ThreadID }](id T) ResourceID {
	var rd ResourceID
	var a any = id
	switch a.(type) {
	case GoID:
		rd.Kind = ResourceGoroutine
	case ProcID:
		rd.Kind = ResourceProc
	case ThreadID:
		rd.Kind = ResourceThread
	}
	rd.id = int64(id)
	return rd
}

// Goroutine obtains a GoID from the resource ID.
//
// r.Kind must be ResourceGoroutine or this function will panic.
func (r ResourceID) Goroutine() GoID {
	if r.Kind != ResourceGoroutine {
		panic(fmt.Sprintf("attempted to get GoID from %s resource ID", r.Kind))
	}
	return GoID(r.id)
}

// Proc obtains a ProcID from the resource ID.
//
// r.Kind must be ResourceProc or this function will panic.
func (r ResourceID) Proc() ProcID {
	if r.Kind != ResourceProc {
		panic(fmt.Sprintf("attempted to get ProcID from %s resource ID", r.Kind))
	}
	return ProcID(r.id)
}

// Thread obtains a ThreadID from the resource ID.
//
// r.Kind must be ResourceThread or this function will panic.
func (r ResourceID) Thread() ThreadID {
	if r.Kind != ResourceThread {
		panic(fmt.Sprintf("attempted to get ThreadID from %s resource ID", r.Kind))
	}
	return ThreadID(r.id)
}

// String returns a human-readable string representation of the ResourceID.
//
// This representation is subject to change and is intended primarily for debugging.
func (r ResourceID) String() string {
	if r.Kind == ResourceNone {
		return r.Kind.String()
	}
	return fmt.Sprintf("%s(%d)", r.Kind, r.id)
}

// StateTransition provides details about a StateTransition event.
type StateTransition struct {
	// Resource is the resource this state transition is for.
	Resource ResourceID

	// Reason is a human-readable reason for the state transition.
	Reason string

	// Stack is the stack trace of the resource making the state transition.
	//
	// This is distinct from the result (Event).Stack because it pertains to
	// the transitioning resource, not any of the ones executing the event
	// this StateTransition came from.
	//
	// An example of this difference is the NotExist -> Runnable transition for
	// goroutines, which indicates goroutine creation. In this particular case,
	// a Stack here would refer to the starting stack of the new goroutine, and
	// an (Event).Stack would refer to the stack trace of whoever created the
	// goroutine.
	Stack Stack

	// The actual transition data. Stored in a neutral form so that
	// we don't need fields for every kind of resource.
	oldState uint8
	newState uint8
}

// MakeGoStateTransition creates a goroutine state transition.
func MakeGoStateTransition(id GoID, from, to GoState) StateTransition {
	return StateTransition{
		Resource: ResourceID{Kind: ResourceGoroutine, id: int64(id)},
		oldState: uint8(from),
		newState: uint8(to),
	}
}

// MakeProcStateTransition creates a proc state transition.
func MakeProcStateTransition(id ProcID, from, to ProcState) StateTransition {
	return StateTransition{
		Resource: ResourceID{Kind: ResourceProc, id: int64(id)},
		oldState: uint8(from),
		newState: uint8(to),
	}
}

// Goroutine returns the state transition for a goroutine.
//
// Transitions to and from states that are Executing are special in that
// they change the future execution context. In other words, future events
// on the same thread will feature the same goroutine until it stops running.
//
// Panics if d.Resource.Kind is not ResourceGoroutine.
func (d StateTransition) Goroutine() (from, to GoState) {
	if d.Resource.Kind != ResourceGoroutine {
		panic("Goroutine called on non-Goroutine state transition")
	}
	return GoState(d.oldState), GoState(d.newState)
}

// Proc returns the state transition for a proc.
//
// Transitions to and from states that are Executing are special in that
// they change the future execution context. In other words, future events
// on the same thread will feature the same goroutine until it stops running.
//
// Panics if d.Resource.Kind is not ResourceProc.
func (d StateTransition) Proc() (from, to ProcState) {
	if d.Resource.Kind != ResourceProc {
		panic("Proc called on non-Proc state transition")
	}
	return ProcState(d.oldState), ProcState(d.newState)
}
