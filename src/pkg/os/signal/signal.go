// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package signal implements access to incoming signals.
package signal

// BUG(rsc): This package is not yet implemented on Plan 9 and Windows.

import (
	"os"
	"sync"
)

var handlers struct {
	sync.Mutex
	list []handler
}

type handler struct {
	c   chan<- os.Signal
	sig os.Signal
	all bool
}

// Notify causes package signal to relay incoming signals to c.
// If no signals are listed, all incoming signals will be relayed to c.
// Otherwise, just the listed signals will.
//
// Package signal will not block sending to c: the caller must ensure
// that c has sufficient buffer space to keep up with the expected
// signal rate.  For a channel used for notification of just one signal value,
// a buffer of size 1 is sufficient.
//
func Notify(c chan<- os.Signal, sig ...os.Signal) {
	if c == nil {
		panic("os/signal: Notify using nil channel")
	}

	handlers.Lock()
	defer handlers.Unlock()
	if len(sig) == 0 {
		enableSignal(nil)
		handlers.list = append(handlers.list, handler{c: c, all: true})
	} else {
		for _, s := range sig {
			// We use nil as a special wildcard value for enableSignal,
			// so filter it out of the list of arguments.  This is safe because
			// we will never get an incoming nil signal, so discarding the
			// registration cannot affect the observed behavior.
			if s != nil {
				enableSignal(s)
				handlers.list = append(handlers.list, handler{c: c, sig: s})
			}
		}
	}
}

func process(sig os.Signal) {
	handlers.Lock()
	defer handlers.Unlock()

	for _, h := range handlers.list {
		if h.all || h.sig == sig {
			// send but do not block for it
			select {
			case h.c <- sig:
			default:
			}
		}
	}
}
