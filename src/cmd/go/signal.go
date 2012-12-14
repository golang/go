// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"os/signal"
	"sync"
)

// interrupted is closed, if go process is interrupted.
var interrupted = make(chan struct{})

// processSignals setups signal handler.
func processSignals() {
	sig := make(chan os.Signal)
	signal.Notify(sig, signalsToIgnore...)
	go func() {
		<-sig
		close(interrupted)
	}()
}

var onceProcessSignals sync.Once

// startSigHandlers start signal handlers.
func startSigHandlers() {
	onceProcessSignals.Do(processSignals)
}
