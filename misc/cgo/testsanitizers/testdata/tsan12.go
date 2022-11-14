// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program hung when run under the C/C++ ThreadSanitizer. TSAN installs a
// libc interceptor that writes signal handlers to a global variable within the
// TSAN runtime instead of making a sigaction system call. A bug in
// syscall.runtime_AfterForkInChild corrupted TSAN's signal forwarding table
// during calls to (*os/exec.Cmd).Run, causing the parent process to fail to
// invoke signal handlers.

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
)

import "C"

func main() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGUSR1)

	if err := exec.Command("true").Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Unexpected error from `true`: %v", err)
		os.Exit(1)
	}

	syscall.Kill(syscall.Getpid(), syscall.SIGUSR1)
	<-ch
}
