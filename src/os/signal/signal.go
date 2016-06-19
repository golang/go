// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"os"
	"sync"
)

var handlers struct {
	sync.Mutex
	m   map[chan<- os.Signal]*handler
	ref [numSig]int64
}

type handler struct {
	mask [(numSig + 31) / 32]uint32
}

func (h *handler) want(sig int) bool {
	return (h.mask[sig/32]>>uint(sig&31))&1 != 0
}

func (h *handler) set(sig int) {
	h.mask[sig/32] |= 1 << uint(sig&31)
}

func (h *handler) clear(sig int) {
	h.mask[sig/32] &^= 1 << uint(sig&31)
}

// Stop relaying the signals, sigs, to any channels previously registered to
// receive them and either reset the signal handlers to their original values
// (action=disableSignal) or ignore the signals (action=ignoreSignal).
func cancel(sigs []os.Signal, action func(int)) {
	handlers.Lock()
	defer handlers.Unlock()

	remove := func(n int) {
		var zerohandler handler

		for c, h := range handlers.m {
			if h.want(n) {
				handlers.ref[n]--
				h.clear(n)
				if h.mask == zerohandler.mask {
					delete(handlers.m, c)
				}
			}
		}

		action(n)
	}

	if len(sigs) == 0 {
		for n := 0; n < numSig; n++ {
			remove(n)
		}
	} else {
		for _, s := range sigs {
			remove(signum(s))
		}
	}
}

// Ignore causes the provided signals to be ignored. If they are received by
// the program, nothing will happen. Ignore undoes the effect of any prior
// calls to Notify for the provided signals.
// If no signals are provided, all incoming signals will be ignored.
func Ignore(sig ...os.Signal) {
	cancel(sig, ignoreSignal)
}

// Notify causes package signal to relay incoming signals to c.
// If no signals are provided, all incoming signals will be relayed to c.
// Otherwise, just the provided signals will.
//
// Package signal will not block sending to c: the caller must ensure
// that c has sufficient buffer space to keep up with the expected
// signal rate. For a channel used for notification of just one signal value,
// a buffer of size 1 is sufficient.
//
// It is allowed to call Notify multiple times with the same channel:
// each call expands the set of signals sent to that channel.
// The only way to remove signals from the set is to call Stop.
//
// It is allowed to call Notify multiple times with different channels
// and the same signals: each channel receives copies of incoming
// signals independently.
func Notify(c chan<- os.Signal, sig ...os.Signal) {
	if c == nil {
		panic("os/signal: Notify using nil channel")
	}

	handlers.Lock()
	defer handlers.Unlock()

	h := handlers.m[c]
	if h == nil {
		if handlers.m == nil {
			handlers.m = make(map[chan<- os.Signal]*handler)
		}
		h = new(handler)
		handlers.m[c] = h
	}

	add := func(n int) {
		if n < 0 {
			return
		}
		if !h.want(n) {
			h.set(n)
			if handlers.ref[n] == 0 {
				enableSignal(n)
			}
			handlers.ref[n]++
		}
	}

	if len(sig) == 0 {
		for n := 0; n < numSig; n++ {
			add(n)
		}
	} else {
		for _, s := range sig {
			add(signum(s))
		}
	}
}

// Reset undoes the effect of any prior calls to Notify for the provided
// signals.
// If no signals are provided, all signal handlers will be reset.
func Reset(sig ...os.Signal) {
	cancel(sig, disableSignal)
}

// Stop causes package signal to stop relaying incoming signals to c.
// It undoes the effect of all prior calls to Notify using c.
// When Stop returns, it is guaranteed that c will receive no more signals.
func Stop(c chan<- os.Signal) {
	handlers.Lock()
	defer handlers.Unlock()

	h := handlers.m[c]
	if h == nil {
		return
	}
	delete(handlers.m, c)

	for n := 0; n < numSig; n++ {
		if h.want(n) {
			handlers.ref[n]--
			if handlers.ref[n] == 0 {
				disableSignal(n)
			}
		}
	}
}

func process(sig os.Signal) {
	n := signum(sig)
	if n < 0 {
		return
	}

	handlers.Lock()
	defer handlers.Unlock()

	for c, h := range handlers.m {
		if h.want(n) {
			// send but do not block for it
			select {
			case c <- sig:
			default:
			}
		}
	}
}
