// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package runtime

import (
	_ "unsafe"
)

// js/wasm has no support for threads yet. There is no preemption.
// Waiting for a mutex is implemented by allowing other goroutines
// to run until the mutex gets unlocked.

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
	for l.key == mutex_locked {
		mcall(gosched_m)
	}
	l.key = mutex_locked
}

func unlock(l *mutex) {
	if l.key == mutex_unlocked {
		throw("unlock of unlocked lock")
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

		id := scheduleCallback(delay)
		mp := acquirem()
		notes[n] = gp
		notesWithTimeout[n] = noteWithTimeout{gp: gp, deadline: deadline}
		releasem(mp)

		gopark(nil, nil, waitReasonSleep, traceEvNone, 1)

		clearScheduledCallback(id) // note might have woken early, clear timeout
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
		if n.key == note_cleared && now > nt.deadline {
			n.key = note_timeout
			goready(nt.gp, 1)
		}
	}
}

var returnedCallback *g

func init() {
	// At the toplevel we need an extra goroutine that handles asynchronous callbacks.
	initg := getg()
	go func() {
		returnedCallback = getg()
		goready(initg, 1)

		gopark(nil, nil, waitReasonZero, traceEvNone, 1)
		returnedCallback = nil

		pause(getcallersp() - 16)
	}()
	gopark(nil, nil, waitReasonZero, traceEvNone, 1)
}

// beforeIdle gets called by the scheduler if no goroutine is awake.
// If a callback has returned, then we resume the callback handler which
// will pause the execution.
func beforeIdle() bool {
	if returnedCallback != nil {
		goready(returnedCallback, 1)
		return true
	}
	return false
}

// pause sets SP to newsp and pauses the execution of Go's WebAssembly code until a callback is triggered.
func pause(newsp uintptr)

// scheduleCallback tells the WebAssembly environment to trigger a callback after ms milliseconds.
// It returns a timer id that can be used with clearScheduledCallback.
func scheduleCallback(ms int64) int32

// clearScheduledCallback clears a callback scheduled by scheduleCallback.
func clearScheduledCallback(id int32)

func handleCallback() {
	prevReturnedCallback := returnedCallback
	returnedCallback = nil

	checkTimeouts()
	callbackHandler()

	returnedCallback = getg()
	gopark(nil, nil, waitReasonZero, traceEvNone, 1)

	returnedCallback = prevReturnedCallback

	pause(getcallersp() - 16)
}

var callbackHandler func()

//go:linkname setCallbackHandler syscall/js.setCallbackHandler
func setCallbackHandler(fn func()) {
	callbackHandler = fn
}
