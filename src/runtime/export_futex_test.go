// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux

package runtime

var Futexwakeup = futexwakeup

//go:nosplit
func Futexsleep(addr *uint32, val uint32, ns int64) {
	// Temporarily disable preemption so that a preemption signal
	// doesn't interrupt the system call.
	poff := debug.asyncpreemptoff
	debug.asyncpreemptoff = 1
	futexsleep(addr, val, ns)
	debug.asyncpreemptoff = poff
}
