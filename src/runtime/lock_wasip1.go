// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package runtime

// wasm has no support for threads yet. There is no preemption.
// See proposal: https://github.com/WebAssembly/threads
// Waiting for a mutex or timeout is implemented as a busy loop
// while allowing other goroutines to run.

const (
	mutex_unlocked = 0
	mutex_locked   = 1

	active_spin     = 4
	active_spin_cnt = 30
)

type mWaitList struct{}

func lockVerifyMSize() {}

func mutexContended(l *mutex) bool {
	return false
}

func lock(l *mutex) {
	lockWithRank(l, getLockRank(l))
}

func lock2(l *mutex) {
	if l.key == mutex_locked {
		// wasm is single-threaded so we should never
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
	throw("notesleep not supported by wasi")
}

func notetsleep(n *note, ns int64) bool {
	throw("notetsleep not supported by wasi")
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
		if sched_yield() != 0 {
			throw("sched_yield failed")
		}
		Gosched()
		if ns >= 0 && nanotime() >= deadline {
			return false
		}
	}
}

func beforeIdle(int64, int64) (*g, bool) {
	return nil, false
}

func checkTimeouts() {}

//go:wasmimport wasi_snapshot_preview1 sched_yield
func sched_yield() errno
