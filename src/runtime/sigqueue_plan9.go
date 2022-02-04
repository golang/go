// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements runtime support for signal handling.

package runtime

import _ "unsafe"

const qsize = 64

var sig struct {
	q     noteQueue
	inuse bool

	lock     mutex
	note     note
	sleeping bool
}

type noteData struct {
	s [_ERRMAX]byte
	n int // n bytes of s are valid
}

type noteQueue struct {
	lock mutex
	data [qsize]noteData
	ri   int
	wi   int
	full bool
}

// It is not allowed to allocate memory in the signal handler.
func (q *noteQueue) push(item *byte) bool {
	lock(&q.lock)
	if q.full {
		unlock(&q.lock)
		return false
	}
	s := gostringnocopy(item)
	copy(q.data[q.wi].s[:], s)
	q.data[q.wi].n = len(s)
	q.wi++
	if q.wi == qsize {
		q.wi = 0
	}
	if q.wi == q.ri {
		q.full = true
	}
	unlock(&q.lock)
	return true
}

func (q *noteQueue) pop() string {
	lock(&q.lock)
	q.full = false
	if q.ri == q.wi {
		unlock(&q.lock)
		return ""
	}
	note := &q.data[q.ri]
	item := string(note.s[:note.n])
	q.ri++
	if q.ri == qsize {
		q.ri = 0
	}
	unlock(&q.lock)
	return item
}

// Called from sighandler to send a signal back out of the signal handling thread.
// Reports whether the signal was sent. If not, the caller typically crashes the program.
func sendNote(s *byte) bool {
	if !sig.inuse {
		return false
	}

	// Add signal to outgoing queue.
	if !sig.q.push(s) {
		return false
	}

	lock(&sig.lock)
	if sig.sleeping {
		sig.sleeping = false
		notewakeup(&sig.note)
	}
	unlock(&sig.lock)

	return true
}

// Called to receive the next queued signal.
// Must only be called from a single goroutine at a time.
//go:linkname signal_recv os/signal.signal_recv
func signal_recv() string {
	for {
		note := sig.q.pop()
		if note != "" {
			return note
		}

		lock(&sig.lock)
		sig.sleeping = true
		noteclear(&sig.note)
		unlock(&sig.lock)
		notetsleepg(&sig.note, -1)
	}
}

// signalWaitUntilIdle waits until the signal delivery mechanism is idle.
// This is used to ensure that we do not drop a signal notification due
// to a race between disabling a signal and receiving a signal.
// This assumes that signal delivery has already been disabled for
// the signal(s) in question, and here we are just waiting to make sure
// that all the signals have been delivered to the user channels
// by the os/signal package.
//go:linkname signalWaitUntilIdle os/signal.signalWaitUntilIdle
func signalWaitUntilIdle() {
	for {
		lock(&sig.lock)
		sleeping := sig.sleeping
		unlock(&sig.lock)
		if sleeping {
			return
		}
		Gosched()
	}
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_enable os/signal.signal_enable
func signal_enable(s uint32) {
	if !sig.inuse {
		// This is the first call to signal_enable. Initialize.
		sig.inuse = true // enable reception of signals; cannot disable
		noteclear(&sig.note)
	}
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_disable os/signal.signal_disable
func signal_disable(s uint32) {
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_ignore os/signal.signal_ignore
func signal_ignore(s uint32) {
}

//go:linkname signal_ignored os/signal.signal_ignored
func signal_ignored(s uint32) bool {
	return false
}
