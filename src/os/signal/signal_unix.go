// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux nacl netbsd openbsd solaris windows

package signal

import (
	"os"
	"syscall"
)

// Defined by the runtime package.
func signal_disable(uint32)
func signal_enable(uint32)
func signal_ignore(uint32)
func signal_ignored(uint32) bool
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

const (
	numSig = 65 // max across all systems
)

func signum(sig os.Signal) int {
	switch sig := sig.(type) {
	case syscall.Signal:
		i := int(sig)
		if i < 0 || i >= numSig {
			return -1
		}
		return i
	default:
		return -1
	}
}

func enableSignal(sig int) {
	signal_enable(uint32(sig))
}

func disableSignal(sig int) {
	signal_disable(uint32(sig))
}

func ignoreSignal(sig int) {
	signal_ignore(uint32(sig))
}

func signalIgnored(sig int) bool {
	return signal_ignored(uint32(sig))
}
