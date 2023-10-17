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
//
// Synchronization between sigsend and signal_recv is based on the sig.state
// variable. It can be in three states:
// * sigReceiving means that signal_recv is blocked on sig.Note and there are
//   no new pending signals.
// * sigSending means that sig.mask *may* contain new pending signals,
//   signal_recv can't be blocked in this state.
// * sigIdle means that there are no new pending signals and signal_recv is not
//   blocked.
//
// Transitions between states are done atomically with CAS.
//
// When signal_recv is unblocked, it resets sig.Note and rechecks sig.mask.
// If several sigsends and signal_recv execute concurrently, it can lead to
// unnecessary rechecks of sig.mask, but it cannot lead to missed signals
// nor deadlocks.

//go:build !plan9

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
	note       note
	mask       [(_NSIG + 31) / 32]uint32
	wanted     [(_NSIG + 31) / 32]uint32
	ignored    [(_NSIG + 31) / 32]uint32
	recv       [(_NSIG + 31) / 32]uint32
	state      atomic.Uint32
	delivering atomic.Uint32
	inuse      bool
}

const (
	sigIdle = iota
	sigReceiving
	sigSending
)

// sigsend delivers a signal from sighandler to the internal signal delivery queue.
// It reports whether the signal was sent. If not, the caller typically crashes the program.
// It runs from the signal handler, so it's limited in what it can do.
func sigsend(s uint32) bool {
	bit := uint32(1) << uint(s&31)
	if s >= uint32(32*len(sig.wanted)) {
		return false
	}

	sig.delivering.Add(1)
	// We are running in the signal handler; defer is not available.

	if w := atomic.Load(&sig.wanted[s/32]); w&bit == 0 {
		sig.delivering.Add(-1)
		return false
	}

	// Add signal to outgoing queue.
	for {
		mask := sig.mask[s/32]
		if mask&bit != 0 {
			sig.delivering.Add(-1)
			return true // signal already in queue
		}
		if atomic.Cas(&sig.mask[s/32], mask, mask|bit) {
			break
		}
	}

	// Notify receiver that queue has new bit.
Send:
	for {
		switch sig.state.Load() {
		default:
			throw("sigsend: inconsistent state")
		case sigIdle:
			if sig.state.CompareAndSwap(sigIdle, sigSending) {
				break Send
			}
		case sigSending:
			// notification already pending
			break Send
		case sigReceiving:
			if sig.state.CompareAndSwap(sigReceiving, sigIdle) {
				if GOOS == "darwin" || GOOS == "ios" {
					sigNoteWakeup(&sig.note)
					break Send
				}
				notewakeup(&sig.note)
				break Send
			}
		}
	}

	sig.delivering.Add(-1)
	return true
}

// Called to receive the next queued signal.
// Must only be called from a single goroutine at a time.
//
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
			switch sig.state.Load() {
			default:
				throw("signal_recv: inconsistent state")
			case sigIdle:
				if sig.state.CompareAndSwap(sigIdle, sigReceiving) {
					if GOOS == "darwin" || GOOS == "ios" {
						sigNoteSleep(&sig.note)
						break Receive
					}
					notetsleepg(&sig.note, -1)
					noteclear(&sig.note)
					break Receive
				}
			case sigSending:
				if sig.state.CompareAndSwap(sigSending, sigIdle) {
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
//
//go:linkname signalWaitUntilIdle os/signal.signalWaitUntilIdle
func signalWaitUntilIdle() {
	// Although the signals we care about have been removed from
	// sig.wanted, it is possible that another thread has received
	// a signal, has read from sig.wanted, is now updating sig.mask,
	// and has not yet woken up the processor thread. We need to wait
	// until all current signal deliveries have completed.
	for sig.delivering.Load() != 0 {
		Gosched()
	}

	// Although WaitUntilIdle seems like the right name for this
	// function, the state we are looking for is sigReceiving, not
	// sigIdle.  The sigIdle state is really more like sigProcessing.
	for sig.state.Load() != sigReceiving {
		Gosched()
	}
}

// Must only be called from a single goroutine at a time.
//
//go:linkname signal_enable os/signal.signal_enable
func signal_enable(s uint32) {
	if !sig.inuse {
		// This is the first call to signal_enable. Initialize.
		sig.inuse = true // enable reception of signals; cannot disable
		if GOOS == "darwin" || GOOS == "ios" {
			sigNoteSetup(&sig.note)
		} else {
			noteclear(&sig.note)
		}
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
//
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
//
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

// sigInitIgnored marks the signal as already ignored. This is called at
// program start by initsig. In a shared library initsig is called by
// libpreinit, so the runtime may not be initialized yet.
//
//go:nosplit
func sigInitIgnored(s uint32) {
	i := sig.ignored[s/32]
	i |= 1 << (s & 31)
	atomic.Store(&sig.ignored[s/32], i)
}

// Checked by signal handlers.
//
//go:linkname signal_ignored os/signal.signal_ignored
func signal_ignored(s uint32) bool {
	i := atomic.Load(&sig.ignored[s/32])
	return i&(1<<(s&31)) != 0
}
