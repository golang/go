// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package runtime

import "runtime/internal/atomic"

var netpollInited uint32
var netpollWaiters uint32

var netpollStubLock mutex
var netpollNote note
var netpollBroken uint32

func netpollGenericInit() {
}

func netpollBreak() {
	if atomic.Cas(&netpollBroken, 0, 1) {
		notewakeup(&netpollNote)
	}
}

// Polls for ready network connections.
// Returns list of goroutines that become runnable.
func netpoll(delay int64) gList {
	// Implementation for platforms that do not support
	// integrated network poller.
	if delay != 0 {
		noteclear(&netpollNote)
		atomic.Store(&netpollBroken, 0)
		notetsleep(&netpollNote, delay)
	}
	return gList{}
}

func netpollinited() bool {
	return false
}
