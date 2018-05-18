// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package runtime

// js/wasm has no support for threads yet. There is no preemption.
// Waiting for a mutex or timeout is implemented as a busy loop
// while allowing other goroutines to run.

const (
	mutex_unlocked = 0
	mutex_locked   = 1

	active_spin     = 4
	active_spin_cnt = 30
	passive_spin    = 1
)

func lock(l *mutex) {
	for l.key == mutex_locked {
		Gosched()
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
func noteclear(n *note) {
	n.key = 0
}

func notewakeup(n *note) {
	if n.key != 0 {
		print("notewakeup - double wakeup (", n.key, ")\n")
		throw("notewakeup - double wakeup")
	}
	n.key = 1
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

	deadline := nanotime() + ns
	for {
		if n.key != 0 {
			return true
		}
		Gosched()
		if ns >= 0 && nanotime() >= deadline {
			return false
		}
	}
}
