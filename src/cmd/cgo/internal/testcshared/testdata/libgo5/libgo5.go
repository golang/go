// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import (
	"os"
	"os/signal"
	"syscall"
	"time"
)

// The channel used to read SIGIO signals.
var sigioChan chan os.Signal

// CatchSIGIO starts catching SIGIO signals.
//
//export CatchSIGIO
func CatchSIGIO() {
	sigioChan = make(chan os.Signal, 1)
	signal.Notify(sigioChan, syscall.SIGIO)
}

// ResetSIGIO stops catching SIGIO signals.
//
//export ResetSIGIO
func ResetSIGIO() {
	signal.Reset(syscall.SIGIO)
}

// AwaitSIGIO blocks indefinitely until a SIGIO is reported.
//
//export AwaitSIGIO
func AwaitSIGIO() {
	<-sigioChan
}

// SawSIGIO reports whether we saw a SIGIO within a brief pause.
//
//export SawSIGIO
func SawSIGIO() bool {
	timer := time.NewTimer(100 * time.Millisecond)
	select {
	case <-sigioChan:
		timer.Stop()
		return true
	case <-timer.C:
		return false
	}
}

func main() {
}
