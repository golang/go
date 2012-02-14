// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package signal

import (
	"os"
	"syscall"
)

// In assembly.
func signal_enable(uint32)
func signal_recv() uint32

func loop() {
	for {
		process(syscall.Signal(signal_recv()))
	}
}

func init() {
	signal_enable(0) // first call - initialize
	go loop()
}

func enableSignal(sig os.Signal) {
	switch sig := sig.(type) {
	case nil:
		signal_enable(^uint32(0))
	case syscall.Signal:
		signal_enable(uint32(sig))
	default:
		// Can ignore: this signal (whatever it is) will never come in.
	}
}
