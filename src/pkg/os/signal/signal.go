// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package signal implements operating system-independent signal handling.
package signal

import (
	"runtime"
	"strconv"
)

// A Signal can represent any operating system signal.
type Signal interface {
	String() string
}

type UnixSignal int32

func (sig UnixSignal) String() string {
	s := runtime.Signame(int32(sig))
	if len(s) > 0 {
		return s
	}
	return "Signal " + strconv.Itoa(int(sig))
}

// Incoming is the global signal channel.
// All signals received by the program will be delivered to this channel.
var Incoming <-chan Signal

func process(ch chan<- Signal) {
	for {
		var mask uint32 = runtime.Sigrecv()
		for sig := uint(0); sig < 32; sig++ {
			if mask&(1<<sig) != 0 {
				ch <- UnixSignal(sig)
			}
		}
	}
}

func init() {
	runtime.Siginit()
	ch := make(chan Signal) // Done here so Incoming can have type <-chan Signal
	Incoming = ch
	go process(ch)
}
