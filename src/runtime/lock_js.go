// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package runtime

import _ "unsafe" // for go:linkname

// js/wasm has no support for threads yet. There is no preemption.

const (
	mutex_unlocked = 0
	mutex_locked   = 1

	note_cleared = 0
	note_woken   = 1
	note_timeout = 2

	active_spin     = 4
	active_spin_cnt = 30
	passive_spin    = 1
)

func mutexContended(l *mutex) bool {
	return false
}

func lock(l *mutex) {
	lockWithRank(l, getLockRank(l))
}

func lock2(l *mutex) {
	if l.key == mutex_locked {
		// js/wasm is single-threaded so we should never
		// observe this.
		throw("self deadlock")
	}
	gp := getg()
	if gp.m.locks < 0 {
		throw("lock count")
	}
	gp.m.locks++
	l.key = mutex_locked
}

func unlock(l *mutex) {
	unlockWithRank(l)
}

func unlock2(l *mutex) {
	if l.key == mutex_unlocked {
		throw("unlock of unlocked lock")
	}
	gp := getg()
	gp.m.locks--
	if gp.m.locks < 0 {
		throw("lock count")
	}
	l.key = mutex_unlocked
}

// One-time notifications.

type noteWithTimeout struct {
	gp       *g
	deadline int64
}

var (
	notes            = make(map[*note]*g)
	notesWithTimeout = make(map[*note]noteWithTimeout)
)

func noteclear(n *note) {
	n.key = note_cleared
}

func notewakeup(n *note) {
	// gp := getg()
	if n.key == note_woken {
		throw("notewakeup - double wakeup")
	}
	cleared := n.key == note_cleared
	n.key = note_woken
	if cleared {
		goready(notes[n], 1)
	}
}

func notesleep(n *note) {
	throw("notesleep not supported by js")
}

func notetsleep(n *note, ns int64) bool {
	throw("notetsleep not supported by js")
	return false
}

// same as runtime·notetsleep, but called on user g (not g0)
func notetsleepg(n *note, ns int64) bool {
	gp := getg()
	if gp == gp.m.g0 {
		throw("notetsleepg on g0")
	}

	if ns >= 0 {
		deadline := nanotime() + ns
		delay := ns/1000000 + 1 // round up
		if delay > 1<<31-1 {
			delay = 1<<31 - 1 // cap to max int32
		}

		id := scheduleTimeoutEvent(delay)
		mp := acquirem()
		notes[n] = gp
		notesWithTimeout[n] = noteWithTimeout{gp: gp, deadline: deadline}
		releasem(mp)

		gopark(nil, nil, waitReasonSleep, traceBlockSleep, 1)

		clearTimeoutEvent(id) // note might have woken early, clear timeout

		mp = acquirem()
		delete(notes, n)
		delete(notesWithTimeout, n)
		releasem(mp)

		return n.key == note_woken
	}

	for n.key != note_woken {
		mp := acquirem()
		notes[n] = gp
		releasem(mp)

		gopark(nil, nil, waitReasonZero, traceBlockGeneric, 1)

		mp = acquirem()
		delete(notes, n)
		releasem(mp)
	}
	return true
}

// checkTimeouts resumes goroutines that are waiting on a note which has reached its deadline.
// TODO(drchase): need to understand if write barriers are really okay in this context.
//
//go:yeswritebarrierrec
func checkTimeouts() {
	now := nanotime()
	// TODO: map iteration has the write barriers in it; is that okay?
	for n, nt := range notesWithTimeout {
		if n.key == note_cleared && now >= nt.deadline {
			n.key = note_timeout
			goready(nt.gp, 1)
		}
	}
}

// events is a stack of calls from JavaScript into Go.
var events []*event

type event struct {
	// g was the active goroutine when the call from JavaScript occurred.
	// It needs to be active when returning to JavaScript.
	gp *g
	// returned reports whether the event handler has returned.
	// When all goroutines are idle and the event handler has returned,
	// then g gets resumed and returns the execution to JavaScript.
	returned bool
}

type timeoutEvent struct {
	id int32
	// The time when this timeout will be triggered.
	time int64
}

// diff calculates the difference of the event's trigger time and x.
func (e *timeoutEvent) diff(x int64) int64 {
	if e == nil {
		return 0
	}

	diff := x - idleTimeout.time
	if diff < 0 {
		diff = -diff
	}
	return diff
}

// clear cancels this timeout event.
func (e *timeoutEvent) clear() {
	if e == nil {
		return
	}

	clearTimeoutEvent(e.id)
}

// The timeout event started by beforeIdle.
var idleTimeout *timeoutEvent

// beforeIdle gets called by the scheduler if no goroutine is awake.
// If we are not already handling an event, then we pause for an async event.
// If an event handler returned, we resume it and it will pause the execution.
// beforeIdle either returns the specific goroutine to schedule next or
// indicates with otherReady that some goroutine became ready.
// TODO(drchase): need to understand if write barriers are really okay in this context.
//
//go:yeswritebarrierrec
func beforeIdle(now, pollUntil int64) (gp *g, otherReady bool) {
	delay := int64(-1)
	if pollUntil != 0 {
		// round up to prevent setTimeout being called early
		delay = (pollUntil-now-1)/1e6 + 1
		if delay > 1e9 {
			// An arbitrary cap on how long to wait for a timer.
			// 1e9 ms == ~11.5 days.
			delay = 1e9
		}
	}

	if delay > 0 && (idleTimeout == nil || idleTimeout.diff(pollUntil) > 1e6) {
		// If the difference is larger than 1 ms, we should reschedule the timeout.
		idleTimeout.clear()

		idleTimeout = &timeoutEvent{
			id:   scheduleTimeoutEvent(delay),
			time: pollUntil,
		}
	}

	if len(events) == 0 {
		// TODO: this is the line that requires the yeswritebarrierrec
		go handleAsyncEvent()
		return nil, true
	}

	e := events[len(events)-1]
	if e.returned {
		return e.gp, false
	}
	return nil, false
}

var idleStart int64

func handleAsyncEvent() {
	idleStart = nanotime()
	pause(getcallersp() - 16)
}

// clearIdleTimeout clears our record of the timeout started by beforeIdle.
func clearIdleTimeout() {
	idleTimeout.clear()
	idleTimeout = nil
}

// pause sets SP to newsp and pauses the execution of Go's WebAssembly code until an event is triggered.
func pause(newsp uintptr)

// scheduleTimeoutEvent tells the WebAssembly environment to trigger an event after ms milliseconds.
// It returns a timer id that can be used with clearTimeoutEvent.
//
//go:wasmimport gojs runtime.scheduleTimeoutEvent
func scheduleTimeoutEvent(ms int64) int32

// clearTimeoutEvent clears a timeout event scheduled by scheduleTimeoutEvent.
//
//go:wasmimport gojs runtime.clearTimeoutEvent
func clearTimeoutEvent(id int32)

// handleEvent gets invoked on a call from JavaScript into Go. It calls the event handler of the syscall/js package
// and then parks the handler goroutine to allow other goroutines to run before giving execution back to JavaScript.
// When no other goroutine is awake any more, beforeIdle resumes the handler goroutine. Now that the same goroutine
// is running as was running when the call came in from JavaScript, execution can be safely passed back to JavaScript.
func handleEvent() {
	sched.idleTime.Add(nanotime() - idleStart)

	e := &event{
		gp:       getg(),
		returned: false,
	}
	events = append(events, e)

	if !eventHandler() {
		// If we did not handle a window event, the idle timeout was triggered, so we can clear it.
		clearIdleTimeout()
	}

	// wait until all goroutines are idle
	e.returned = true
	gopark(nil, nil, waitReasonZero, traceBlockGeneric, 1)

	events[len(events)-1] = nil
	events = events[:len(events)-1]

	// return execution to JavaScript
	idleStart = nanotime()
	pause(getcallersp() - 16)
}

// eventHandler retrieves and executes handlers for pending JavaScript events.
// It returns true if an event was handled.
var eventHandler func() bool

//go:linkname setEventHandler syscall/js.setEventHandler
func setEventHandler(fn func() bool) {
	eventHandler = fn
}
