// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"os"
	"os/signal"
	"sync"
)

// Interrupted is closed when the go command receives an interrupt signal.
var Interrupted = make(chan struct{})

// processSignals setups signal handler.
func processSignals() {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, signalsToIgnore...)
	go func() {
		<-sig
		close(Interrupted)
	}()
}

var processSignalsOnce = sync.OnceFunc(processSignals)

// StartSigHandlers starts the signal handlers.
func StartSigHandlers() {
	processSignalsOnce()
}
