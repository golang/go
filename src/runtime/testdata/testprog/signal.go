// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

package main

import (
	"syscall"
	"time"
)

func init() {
	register("SignalExitStatus", SignalExitStatus)
}

func SignalExitStatus() {
	syscall.Kill(syscall.Getpid(), syscall.SIGTERM)

	// Should die immediately, but we've seen flakiness on various
	// systems (see issue 14063). It's possible that the signal is
	// being delivered to a different thread and we are returning
	// and exiting before that thread runs again. Give the program
	// a little while to die to make sure we pick up the signal
	// before we return and exit the program. The time here
	// shouldn't matter--we'll never really sleep this long.
	time.Sleep(time.Second)
}
