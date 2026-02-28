// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// Raise SIGPIPE.
static void CRaiseSIGPIPE() {
	int fds[2];

	if (pipe(fds) == -1) {
		perror("pipe");
		exit(EXIT_FAILURE);
	}
	// Close the reader end
	close(fds[0]);
	// Write to the writer end to provoke a SIGPIPE
	if (write(fds[1], "some data", 9) != -1) {
		fprintf(stderr, "write to a closed pipe succeeded\n");
		exit(EXIT_FAILURE);
	}
	close(fds[1]);
}
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
)

// RunGoroutines starts some goroutines that don't do anything.
// The idea is to get some threads going, so that a signal will be delivered
// to a thread started by Go.
//
//export RunGoroutines
func RunGoroutines() {
	for i := 0; i < 4; i++ {
		go func() {
			runtime.LockOSThread()
			select {}
		}()
	}
}

// Block blocks the current thread while running Go code.
//
//export Block
func Block() {
	select {}
}

var P *byte

// TestSEGV makes sure that an invalid address turns into a run-time Go panic.
//
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
//
//export Noop
func Noop() {
}

// Raise SIGPIPE.
//
//export GoRaiseSIGPIPE
func GoRaiseSIGPIPE() {
	C.CRaiseSIGPIPE()
}

func main() {
}
