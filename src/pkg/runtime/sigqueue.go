// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements runtime support for signal handling.
//
// Most synchronization primitives are not available from
// the signal handler (it cannot block, allocate memory, or use locks)
// so the handler communicates with a processing goroutine
// via struct sig, below.
//
// sigsend() is called by the signal handler to queue a new signal.
// signal_recv() is called by the Go program to receive a newly queued signal.
// Synchronization between sigsend() and signal_recv() is based on the sig.state
// variable.  It can be in 3 states: 0, HASWAITER and HASSIGNAL.
// HASWAITER means that signal_recv() is blocked on sig.Note and there are no
// new pending signals.
// HASSIGNAL means that sig.mask *may* contain new pending signals,
// signal_recv() can't be blocked in this state.
// 0 means that there are no new pending signals and signal_recv() is not blocked.
// Transitions between states are done atomically with CAS.
// When signal_recv() is unblocked, it resets sig.Note and rechecks sig.mask.
// If several sigsend()'s and signal_recv() execute concurrently, it can lead to
// unnecessary rechecks of sig.mask, but must not lead to missed signals
// nor deadlocks.

package runtime

import "unsafe"

var sig struct {
	note   note
	mask   [(_NSIG + 31) / 32]uint32
	wanted [(_NSIG + 31) / 32]uint32
	recv   [(_NSIG + 31) / 32]uint32
	state  uint32
	inuse  bool
}

const (
	_HASWAITER = 1
	_HASSIGNAL = 2
)

// Called from sighandler to send a signal back out of the signal handling thread.
func sigsend(s int32) bool {
	bit := uint32(1) << uint(s&31)
	if !sig.inuse || s < 0 || int(s) >= 32*len(sig.wanted) || sig.wanted[s/32]&bit == 0 {
		return false
	}

	for {
		mask := sig.mask[s/32]
		if mask&bit != 0 {
			break // signal already in queue
		}
		if cas(&sig.mask[s/32], mask, mask|bit) {
			// Added to queue.
			// Only send a wakeup if the receiver needs a kick.
			for {
				old := atomicload(&sig.state)
				if old == _HASSIGNAL {
					break
				}

				var new uint32
				if old == _HASWAITER {
					new = 0
				} else { // old == 0
					new = _HASSIGNAL
				}
				if cas(&sig.state, old, new) {
					if old == _HASWAITER {
						notewakeup(&sig.note)
					}
					break
				}
			}
			break
		}
	}
	return true
}

// Called to receive the next queued signal.
// Must only be called from a single goroutine at a time.
func signal_recv() uint32 {
	for {
		// Serve from local copy if there are bits left.
		for i := uint32(0); i < _NSIG; i++ {
			if sig.recv[i/32]&(1<<(i&31)) != 0 {
				sig.recv[i/32] &^= 1 << (i & 31)
				return i
			}
		}

		// Check and update sig.state.
		for {
			old := atomicload(&sig.state)
			if old == _HASWAITER {
				gothrow("inconsistent state in signal_recv")
			}

			var new uint32
			if old == _HASSIGNAL {
				new = 0
			} else { // old == 0
				new = _HASWAITER
			}
			if cas(&sig.state, old, new) {
				if new == _HASWAITER {
					notetsleepg(&sig.note, -1)
					noteclear(&sig.note)
				}
				break
			}
		}

		// Get a new local copy.
		for i := range sig.mask {
			var m uint32
			for {
				m = sig.mask[i]
				if cas(&sig.mask[i], m, 0) {
					break
				}
			}
			sig.recv[i] = m
		}
	}
}

// Must only be called from a single goroutine at a time.
func signal_enable(s uint32) {
	if !sig.inuse {
		// The first call to signal_enable is for us
		// to use for initialization.  It does not pass
		// signal information in m.
		sig.inuse = true // enable reception of signals; cannot disable
		noteclear(&sig.note)
		return
	}

	if int(s) >= len(sig.wanted)*32 {
		return
	}
	sig.wanted[s/32] |= 1 << (s & 31)
	sigenable_go(s)
}

// Must only be called from a single goroutine at a time.
func signal_disable(s uint32) {
	if int(s) >= len(sig.wanted)*32 {
		return
	}
	sig.wanted[s/32] &^= 1 << (s & 31)
	sigdisable_go(s)
}

// This runs on a foreign stack, without an m or a g.  No stack split.
//go:nosplit
func badsignal(sig uintptr) {
	cgocallback(unsafe.Pointer(funcPC(sigsend)), noescape(unsafe.Pointer(&sig)), unsafe.Sizeof(sig))
}

func sigenable_m()
func sigdisable_m()

func sigenable_go(s uint32) {
	g := getg()
	g.m.scalararg[0] = uintptr(s)
	onM(sigenable_m)
}

func sigdisable_go(s uint32) {
	g := getg()
	g.m.scalararg[0] = uintptr(s)
	onM(sigdisable_m)
}
