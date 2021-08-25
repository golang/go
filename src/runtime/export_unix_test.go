// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package runtime

import "unsafe"

var NonblockingPipe = nonblockingPipe
var SetNonblock = setNonblock
var Closeonexec = closeonexec

func sigismember(mask *sigset, i int) bool {
	clear := *mask
	sigdelset(&clear, i)
	return clear != *mask
}

func Sigisblocked(i int) bool {
	var sigmask sigset
	sigprocmask(_SIG_SETMASK, nil, &sigmask)
	return sigismember(&sigmask, i)
}

type M = m

var waitForSigusr1 struct {
	rdpipe int32
	wrpipe int32
	mID    int64
}

// WaitForSigusr1 blocks until a SIGUSR1 is received. It calls ready
// when it is set up to receive SIGUSR1. The ready function should
// cause a SIGUSR1 to be sent. The r and w arguments are a pipe that
// the signal handler can use to report when the signal is received.
//
// Once SIGUSR1 is received, it returns the ID of the current M and
// the ID of the M the SIGUSR1 was received on. If the caller writes
// a non-zero byte to w, WaitForSigusr1 returns immediately with -1, -1.
func WaitForSigusr1(r, w int32, ready func(mp *M)) (int64, int64) {
	lockOSThread()
	// Make sure we can receive SIGUSR1.
	unblocksig(_SIGUSR1)

	waitForSigusr1.rdpipe = r
	waitForSigusr1.wrpipe = w

	mp := getg().m
	testSigusr1 = waitForSigusr1Callback
	ready(mp)

	// Wait for the signal. We use a pipe rather than a note
	// because write is always async-signal-safe.
	entersyscallblock()
	var b byte
	read(waitForSigusr1.rdpipe, noescape(unsafe.Pointer(&b)), 1)
	exitsyscall()

	gotM := waitForSigusr1.mID
	testSigusr1 = nil

	unlockOSThread()

	if b != 0 {
		// timeout signal from caller
		return -1, -1
	}
	return mp.id, gotM
}

// waitForSigusr1Callback is called from the signal handler during
// WaitForSigusr1. It must not have write barriers because there may
// not be a P.
//
//go:nowritebarrierrec
func waitForSigusr1Callback(gp *g) bool {
	if gp == nil || gp.m == nil {
		waitForSigusr1.mID = -1
	} else {
		waitForSigusr1.mID = gp.m.id
	}
	b := byte(0)
	write(uintptr(waitForSigusr1.wrpipe), noescape(unsafe.Pointer(&b)), 1)
	return true
}

// SendSigusr1 sends SIGUSR1 to mp.
func SendSigusr1(mp *M) {
	signalM(mp, _SIGUSR1)
}
