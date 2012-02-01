// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

// Package signal implements operating system-independent signal handling.
package signal

import (
	"os"
	"runtime"
)

// Incoming is the global signal channel.
// All signals received by the program will be delivered to this channel.
var Incoming <-chan os.Signal

func process(ch chan<- os.Signal) {
	for {
		var mask uint32 = runtime.Sigrecv()
		for sig := uint(0); sig < 32; sig++ {
			if mask&(1<<sig) != 0 {
				ch <- os.UnixSignal(sig)
			}
		}
	}
}

func init() {
	runtime.Siginit()
	ch := make(chan os.Signal) // Done here so Incoming can have type <-chan Signal
	Incoming = ch
	go process(ch)
}

// BUG(rsc): This package is unavailable on Plan 9 and Windows.
