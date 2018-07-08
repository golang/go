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

var waitingForCallback *g

// sleepUntilCallback puts the current goroutine to sleep until a callback is triggered.
// It is currently only used by the callback routine of the syscall/js package.
//go:linkname sleepUntilCallback syscall/js.sleepUntilCallback
func sleepUntilCallback() {
	waitingForCallback = getg()
	gopark(nil, nil, waitReasonZero, traceEvNone, 1)
	waitingForCallback = nil
}

// pauseSchedulerUntilCallback gets called from the scheduler and pauses the execution
// of Go's WebAssembly code until a callback is triggered. Then it checks for note timeouts
// and resumes goroutines that are waiting for a callback.
func pauseSchedulerUntilCallback() bool {
	if waitingForCallback == nil && len(notesWithTimeout) == 0 {
		return false
	}

	pause()
	checkTimeouts()
	if waitingForCallback != nil {
		goready(waitingForCallback, 1)
	}
	return true
}

// pause pauses the execution of Go's WebAssembly code until a callback is triggered.
func pause()

// scheduleCallback tells the WebAssembly environment to trigger a callback after ms milliseconds.
// It returns a timer id that can be used with clearScheduledCallback.
func scheduleCallback(ms int64) int32

// clearScheduledCallback clears a callback scheduled by scheduleCallback.
func clearScheduledCallback(id int32)
