// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import (
	"fmt"
	"os"
	"runtime"
)

// RunGoroutines starts some goroutines that don't do anything.
// The idea is to get some threads going, so that a signal will be delivered
// to a thread started by Go.
//export RunGoroutines
func RunGoroutines() {
	for i := 0; i < 4; i++ {
		go func() {
			runtime.LockOSThread()
			select {}
		}()
	}
}

var P *byte

// TestSEGV makes sure that an invalid address turns into a run-time Go panic.
//export TestSEGV
func TestSEGV() {
	defer func() {
		if recover() == nil {
			fmt.Fprintln(os.Stderr, "no panic from segv")
			os.Exit(1)
		}
	}()
	*P = 0
	fmt.Fprintln(os.Stderr, "continued after segv")
	os.Exit(1)
}

// Noop ensures that the Go runtime is initialized.
//export Noop
func Noop() {
}

func main() {
}
