// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"os"
	"syscall"
)

var sigtab = make(map[os.Signal]int)

// Defined by the runtime package.
func signal_disable(uint32)
func signal_enable(uint32)
func signal_ignore(uint32)
func signal_ignored(uint32) bool
func signal_recv() string

func init() {
	watchSignalLoop = loop
}

func loop() {
	for {
		process(syscall.Note(signal_recv()))
	}
}

const numSig = 256

func signum(sig os.Signal) int {
	switch sig := sig.(type) {
	case syscall.Note:
		n, ok := sigtab[sig]
		if !ok {
			n = len(sigtab) + 1
			if n > numSig {
				return -1
			}
			sigtab[sig] = n
		}
		return n
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
