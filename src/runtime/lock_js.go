// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package runtime

import (
	_ "unsafe"
)

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

func lock(l *mutex) {
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

		gopark(nil, nil, waitReasonSleep, traceEvNone, 1)

		clearTimeoutEvent(id) // note might have woken early, clear timeout
		clearIdleID()

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

		gopark(nil, nil, waitReasonZero, traceEvNone, 1)

		mp = acquirem()
		delete(notes, n)
		releasem(mp)
	}
	return true
}

// checkTimeouts resumes goroutines that are waiting on a note which has reached its deadline.
func checkTimeouts() {
	now := nanotime()
	for n, nt := range notesWithTimeout {
		if n.key == note_cleared && now >= nt.deadline {
			n.key = note_timeout
			goready(nt.gp, 1)
		}
	}
}

var isHandlingEvent = false
var nextEventIsAsync = false
var returnedEventHandler *g

// The timeout event started by beforeIdle.
var idleID int32

// beforeIdle gets called by the scheduler if no goroutine is awake.
// If we are not already handling an event, then we pause for an async event.
// If an event handler returned, we resume it and it will pause the execution.
func beforeIdle(delay int64) bool {
	if delay > 0 {
		if delay < 1e6 {
			delay = 1
		} else if delay < 1e15 {
			delay = delay / 1e6
		} else {
			// An arbitrary cap on how long to wait for a timer.
			// 1e9 ms == ~11.5 days.
			delay = 1e9
		}
		idleID = scheduleTimeoutEvent(delay)
	}
	if !isHandlingEvent {
		nextEventIsAsync = true
		pause(getcallersp() - 16)
		return true
	}
	if returnedEventHandler != nil {
		goready(returnedEventHandler, 1)
		return true
	}
	return false
}

// clearIdleID clears our record of the timeout started by beforeIdle.
func clearIdleID() {
	if idleID != 0 {
		clearTimeoutEvent(idleID)
		idleID = 0
	}
}

// pause sets SP to newsp and pauses the execution of Go's WebAssembly code until an event is triggered.
func pause(newsp uintptr)

// scheduleTimeoutEvent tells the WebAssembly environment to trigger an event after ms milliseconds.
// It returns a timer id that can be used with clearTimeoutEvent.
func scheduleTimeoutEvent(ms int64) int32

// clearTimeoutEvent clears a timeout event scheduled by scheduleTimeoutEvent.
func clearTimeoutEvent(id int32)

func handleEvent() {
	if nextEventIsAsync {
		nextEventIsAsync = false
		checkTimeouts()
		go handleAsyncEvent()
		return
	}

	prevIsHandlingEvent := isHandlingEvent
	isHandlingEvent = true
	prevReturnedEventHandler := returnedEventHandler
	returnedEventHandler = nil

	eventHandler()

	clearIdleID()

	// wait until all goroutines are idle
	returnedEventHandler = getg()
	gopark(nil, nil, waitReasonZero, traceEvNone, 1)

	isHandlingEvent = prevIsHandlingEvent
	returnedEventHandler = prevReturnedEventHandler

	pause(getcallersp() - 16)
}

func handleAsyncEvent() {
	isHandlingEvent = true
	eventHandler()
	isHandlingEvent = false
}

var eventHandler func()

//go:linkname setEventHandler syscall/js.setEventHandler
func setEventHandler(fn func()) {
	eventHandler = fn
}
