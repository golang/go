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
//export CatchSIGIO
func CatchSIGIO() {
	sigioChan = make(chan os.Signal, 1)
	signal.Notify(sigioChan, syscall.SIGIO)
}

// ResetSIGIO stops catching SIGIO signals.
//export ResetSIGIO
func ResetSIGIO() {
	signal.Reset(syscall.SIGIO)
}

// SawSIGIO reports whether we saw a SIGIO.
//export SawSIGIO
func SawSIGIO() C.int {
	select {
	case <-sigioChan:
		return 1
	case <-time.After(5 * time.Second):
		return 0
	}
}

// ProvokeSIGPIPE provokes a kernel-initiated SIGPIPE.
//export ProvokeSIGPIPE
func ProvokeSIGPIPE() {
	r, w, err := os.Pipe()
	if err != nil {
		panic(err)
	}
	r.Close()
	defer w.Close()
	w.Write([]byte("some data"))
}

func main() {
}
