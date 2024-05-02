// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package runtime

import "internal/runtime/atomic"

var netpollInited atomic.Uint32

var netpollStubLock mutex
var netpollNote note

// netpollBroken, protected by netpollBrokenLock, avoids a double notewakeup.
var netpollBrokenLock mutex
var netpollBroken bool

func netpollGenericInit() {
	netpollInited.Store(1)
}

func netpollBreak() {
	lock(&netpollBrokenLock)
	broken := netpollBroken
	netpollBroken = true
	if !broken {
		notewakeup(&netpollNote)
	}
	unlock(&netpollBrokenLock)
}

// Polls for ready network connections.
// Returns list of goroutines that become runnable.
func netpoll(delay int64) (gList, int32) {
	// Implementation for platforms that do not support
	// integrated network poller.
	if delay != 0 {
		// This lock ensures that only one goroutine tries to use
		// the note. It should normally be completely uncontended.
		lock(&netpollStubLock)

		lock(&netpollBrokenLock)
		noteclear(&netpollNote)
		netpollBroken = false
		unlock(&netpollBrokenLock)

		notetsleep(&netpollNote, delay)
		unlock(&netpollStubLock)
		// Guard against starvation in case the lock is contended
		// (eg when running TestNetpollBreak).
		osyield()
	}
	return gList{}, 0
}

func netpollinited() bool {
	return netpollInited.Load() != 0
}

func netpollAnyWaiters() bool {
	return false
}

func netpollAdjustWaiters(delta int32) {
}
