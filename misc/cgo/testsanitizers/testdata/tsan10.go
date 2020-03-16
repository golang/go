// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program hung when run under the C/C++ ThreadSanitizer.
// TSAN defers asynchronous signals until the signaled thread calls into libc.
// Since the Go runtime makes direct futex syscalls, Go runtime threads could
// run for an arbitrarily long time without triggering the libc interceptors.
// See https://golang.org/issue/18717.

import (
	"os"
	"os/signal"
	"syscall"
)

/*
#cgo CFLAGS: -g -fsanitize=thread
#cgo LDFLAGS: -g -fsanitize=thread
*/
import "C"

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGUSR1)
	defer signal.Stop(c)
	syscall.Kill(syscall.Getpid(), syscall.SIGUSR1)
	<-c
}
