// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"fmt"
	"os"
)

/*
 * Hooks and events
 */

// An EventHandler is a function that takes an event and returns a
// response to that event and possibly an error.  If an event handler
// returns an error, the process stops and no other handlers for that
// event are executed.
type EventHandler func(e Event) (EventAction, os.Error)

// An EventAction is an event handler's response to an event.  If all
// of an event's handlers execute without returning errors, their
// results are combined as follows: If any handler returned
// EAContinue, then the process resumes (without returning from
// WaitStop); otherwise, if any handler returned EAStop, the process
// remains stopped; otherwise, if all handlers returned EADefault, the
// process resumes.  A handler may return EARemoveSelf bit-wise or'd
// with any other action to indicate that the handler should be
// removed from the hook.
type EventAction int

const (
	EARemoveSelf EventAction = 0x100
	EADefault    EventAction = iota
	EAStop
	EAContinue
)

// A EventHook allows event handlers to be added and removed.
type EventHook interface {
	AddHandler(EventHandler)
	RemoveHandler(EventHandler)
	NumHandler() int
	handle(e Event) (EventAction, os.Error)
	String() string
}

// EventHook is almost, but not quite, suitable for user-defined
// events.  If we want user-defined events, make EventHook a struct,
// special-case adding and removing handlers in breakpoint hooks, and
// provide a public interface for posting events to hooks.

type Event interface {
	Process() *Process
	Goroutine() *Goroutine
	String() string
}

type commonHook struct {
	// Head of handler chain
	head *handler
	// Number of non-internal handlers
	len int
}

type handler struct {
	eh EventHandler
	// True if this handler must be run before user-defined
	// handlers in order to ensure correctness.
	internal bool
	// True if this handler has been removed from the chain.
	removed bool
	next    *handler
}

func (h *commonHook) AddHandler(eh EventHandler) {
	h.addHandler(eh, false)
}

func (h *commonHook) addHandler(eh EventHandler, internal bool) {
	// Ensure uniqueness of handlers
	h.RemoveHandler(eh)

	if !internal {
		h.len++
	}
	// Add internal handlers to the beginning
	if internal || h.head == nil {
		h.head = &handler{eh, internal, false, h.head}
		return
	}
	// Add handler after internal handlers
	// TODO(austin) This should probably go on the end instead
	prev := h.head
	for prev.next != nil && prev.internal {
		prev = prev.next
	}
	prev.next = &handler{eh, internal, false, prev.next}
}

func (h *commonHook) RemoveHandler(eh EventHandler) {
	plink := &h.head
	for l := *plink; l != nil; plink, l = &l.next, l.next {
		if l.eh == eh {
			if !l.internal {
				h.len--
			}
			l.removed = true
			*plink = l.next
			break
		}
	}
}

func (h *commonHook) NumHandler() int { return h.len }

func (h *commonHook) handle(e Event) (EventAction, os.Error) {
	action := EADefault
	plink := &h.head
	for l := *plink; l != nil; plink, l = &l.next, l.next {
		if l.removed {
			continue
		}
		a, err := l.eh(e)
		if a&EARemoveSelf == EARemoveSelf {
			if !l.internal {
				h.len--
			}
			l.removed = true
			*plink = l.next
			a &^= EARemoveSelf
		}
		if err != nil {
			return EAStop, err
		}
		if a > action {
			action = a
		}
	}
	return action, nil
}

type commonEvent struct {
	// The process of this event
	p *Process
	// The goroutine of this event.
	t *Goroutine
}

func (e *commonEvent) Process() *Process { return e.p }

func (e *commonEvent) Goroutine() *Goroutine { return e.t }

/*
 * Standard event handlers
 */

// EventPrint is a standard event handler that prints events as they
// occur.  It will not cause the process to stop.
func EventPrint(ev Event) (EventAction, os.Error) {
	// TODO(austin) Include process name here?
	fmt.Fprintf(os.Stderr, "*** %v\n", ev.String())
	return EADefault, nil
}

// EventStop is a standard event handler that causes the process to stop.
func EventStop(ev Event) (EventAction, os.Error) {
	return EAStop, nil
}

/*
 * Breakpoints
 */

type breakpointHook struct {
	commonHook
	p  *Process
	pc proc.Word
}

// A Breakpoint event occurs when a process reaches a particular
// program counter.  When this event is handled, the current goroutine
// will be the goroutine that reached the program counter.
type Breakpoint struct {
	commonEvent
	osThread proc.Thread
	pc       proc.Word
}

func (h *breakpointHook) AddHandler(eh EventHandler) {
	h.addHandler(eh, false)
}

func (h *breakpointHook) addHandler(eh EventHandler, internal bool) {
	// We register breakpoint events lazily to avoid holding
	// references to breakpoints without handlers.  Be sure to use
	// the "canonical" breakpoint if there is one.
	if cur, ok := h.p.breakpointHooks[h.pc]; ok {
		h = cur
	}
	oldhead := h.head
	h.commonHook.addHandler(eh, internal)
	if oldhead == nil && h.head != nil {
		h.p.proc.AddBreakpoint(h.pc)
		h.p.breakpointHooks[h.pc] = h
	}
}

func (h *breakpointHook) RemoveHandler(eh EventHandler) {
	oldhead := h.head
	h.commonHook.RemoveHandler(eh)
	if oldhead != nil && h.head == nil {
		h.p.proc.RemoveBreakpoint(h.pc)
		h.p.breakpointHooks[h.pc] = nil, false
	}
}

func (h *breakpointHook) String() string {
	// TODO(austin) Include process name?
	// TODO(austin) Use line:pc or at least sym+%#x
	return fmt.Sprintf("breakpoint at %#x", h.pc)
}

func (b *Breakpoint) PC() proc.Word { return b.pc }

func (b *Breakpoint) String() string {
	// TODO(austin) Include process name and goroutine
	// TODO(austin) Use line:pc or at least sym+%#x
	return fmt.Sprintf("breakpoint at %#x", b.pc)
}

/*
 * Goroutine create/exit
 */

type goroutineCreateHook struct {
	commonHook
}

func (h *goroutineCreateHook) String() string { return "goroutine create" }

// A GoroutineCreate event occurs when a process creates a new
// goroutine.  When this event is handled, the current goroutine will
// be the newly created goroutine.
type GoroutineCreate struct {
	commonEvent
	parent *Goroutine
}

// Parent returns the goroutine that created this goroutine.  May be
// nil if this event is the creation of the first goroutine.
func (e *GoroutineCreate) Parent() *Goroutine { return e.parent }

func (e *GoroutineCreate) String() string {
	// TODO(austin) Include process name
	if e.parent == nil {
		return fmt.Sprintf("%v created", e.t)
	}
	return fmt.Sprintf("%v created by %v", e.t, e.parent)
}

type goroutineExitHook struct {
	commonHook
}

func (h *goroutineExitHook) String() string { return "goroutine exit" }

// A GoroutineExit event occurs when a Go goroutine exits.
type GoroutineExit struct {
	commonEvent
}

func (e *GoroutineExit) String() string {
	// TODO(austin) Include process name
	//return fmt.Sprintf("%v exited", e.t);
	// For debugging purposes
	return fmt.Sprintf("goroutine %#x exited", e.t.g.addr().base)
}
