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
// sigsend is called by the signal handler to queue a new signal.
// signal_recv is called by the Go program to receive a newly queued signal.
// Synchronization between sigsend and signal_recv is based on the sig.state
// variable. It can be in 3 states: sigIdle, sigReceiving and sigSending.
// sigReceiving means that signal_recv is blocked on sig.Note and there are no
// new pending signals.
// sigSending means that sig.mask *may* contain new pending signals,
// signal_recv can't be blocked in this state.
// sigIdle means that there are no new pending signals and signal_recv is not blocked.
// Transitions between states are done atomically with CAS.
// When signal_recv is unblocked, it resets sig.Note and rechecks sig.mask.
// If several sigsends and signal_recv execute concurrently, it can lead to
// unnecessary rechecks of sig.mask, but it cannot lead to missed signals
// nor deadlocks.

// +build !plan9

package runtime

import (
	"runtime/internal/atomic"
	_ "unsafe" // for go:linkname
)

// sig handles communication between the signal handler and os/signal.
// Other than the inuse and recv fields, the fields are accessed atomically.
//
// The wanted and ignored fields are only written by one goroutine at
// a time; access is controlled by the handlers Mutex in os/signal.
// The fields are only read by that one goroutine and by the signal handler.
// We access them atomically to minimize the race between setting them
// in the goroutine calling os/signal and the signal handler,
// which may be running in a different thread. That race is unavoidable,
// as there is no connection between handling a signal and receiving one,
// but atomic instructions should minimize it.
var sig struct {
	note    note
	mask    [(_NSIG + 31) / 32]uint32
	wanted  [(_NSIG + 31) / 32]uint32
	ignored [(_NSIG + 31) / 32]uint32
	recv    [(_NSIG + 31) / 32]uint32
	state   uint32
	inuse   bool
}

const (
	sigIdle = iota
	sigReceiving
	sigSending
)

// Called from sighandler to send a signal back out of the signal handling thread.
// Reports whether the signal was sent. If not, the caller typically crashes the program.
func sigsend(s uint32) bool {
	bit := uint32(1) << uint(s&31)
	if !sig.inuse || s >= uint32(32*len(sig.wanted)) {
		return false
	}

	if w := atomic.Load(&sig.wanted[s/32]); w&bit == 0 {
		return false
	}

	// Add signal to outgoing queue.
	for {
		mask := sig.mask[s/32]
		if mask&bit != 0 {
			return true // signal already in queue
		}
		if atomic.Cas(&sig.mask[s/32], mask, mask|bit) {
			break
		}
	}

	// Notify receiver that queue has new bit.
Send:
	for {
		switch atomic.Load(&sig.state) {
		default:
			throw("sigsend: inconsistent state")
		case sigIdle:
			if atomic.Cas(&sig.state, sigIdle, sigSending) {
				break Send
			}
		case sigSending:
			// notification already pending
			break Send
		case sigReceiving:
			if atomic.Cas(&sig.state, sigReceiving, sigIdle) {
				notewakeup(&sig.note)
				break Send
			}
		}
	}

	return true
}

// Called to receive the next queued signal.
// Must only be called from a single goroutine at a time.
//go:linkname signal_recv os/signal.signal_recv
func signal_recv() uint32 {
	for {
		// Serve any signals from local copy.
		for i := uint32(0); i < _NSIG; i++ {
			if sig.recv[i/32]&(1<<(i&31)) != 0 {
				sig.recv[i/32] &^= 1 << (i & 31)
				return i
			}
		}

		// Wait for updates to be available from signal sender.
	Receive:
		for {
			switch atomic.Load(&sig.state) {
			default:
				throw("signal_recv: inconsistent state")
			case sigIdle:
				if atomic.Cas(&sig.state, sigIdle, sigReceiving) {
					notetsleepg(&sig.note, -1)
					noteclear(&sig.note)
					break Receive
				}
			case sigSending:
				if atomic.Cas(&sig.state, sigSending, sigIdle) {
					break Receive
				}
			}
		}

		// Incorporate updates from sender into local copy.
		for i := range sig.mask {
			sig.recv[i] = atomic.Xchg(&sig.mask[i], 0)
		}
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
	// Although WaitUntilIdle seems like the right name for this
	// function, the state we are looking for is sigReceiving, not
	// sigIdle.  The sigIdle state is really more like sigProcessing.
	for atomic.Load(&sig.state) != sigReceiving {
		Gosched()
	}
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_enable os/signal.signal_enable
func signal_enable(s uint32) {
	if !sig.inuse {
		// The first call to signal_enable is for us
		// to use for initialization. It does not pass
		// signal information in m.
		sig.inuse = true // enable reception of signals; cannot disable
		noteclear(&sig.note)
		return
	}

	if s >= uint32(len(sig.wanted)*32) {
		return
	}

	w := sig.wanted[s/32]
	w |= 1 << (s & 31)
	atomic.Store(&sig.wanted[s/32], w)

	i := sig.ignored[s/32]
	i &^= 1 << (s & 31)
	atomic.Store(&sig.ignored[s/32], i)

	sigenable(s)
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_disable os/signal.signal_disable
func signal_disable(s uint32) {
	if s >= uint32(len(sig.wanted)*32) {
		return
	}
	sigdisable(s)

	w := sig.wanted[s/32]
	w &^= 1 << (s & 31)
	atomic.Store(&sig.wanted[s/32], w)
}

// Must only be called from a single goroutine at a time.
//go:linkname signal_ignore os/signal.signal_ignore
func signal_ignore(s uint32) {
	if s >= uint32(len(sig.wanted)*32) {
		return
	}
	sigignore(s)

	w := sig.wanted[s/32]
	w &^= 1 << (s & 31)
	atomic.Store(&sig.wanted[s/32], w)

	i := sig.ignored[s/32]
	i |= 1 << (s & 31)
	atomic.Store(&sig.ignored[s/32], i)
}

// Checked by signal handlers.
func signal_ignored(s uint32) bool {
	i := atomic.Load(&sig.ignored[s/32])
	return i&(1<<(s&31)) != 0
}
