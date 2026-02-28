// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program hung when run under the C/C++ ThreadSanitizer. TSAN defers
// asynchronous signals until the signaled thread calls into libc. The runtime's
// sysmon goroutine idles itself using direct usleep syscalls, so it could
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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void raise_usr2(int signo) {
	raise(SIGUSR2);
}

static void register_handler(int signo) {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_ONSTACK;
	sa.sa_handler = raise_usr2;

	if (sigaction(SIGUSR1, &sa, NULL) != 0) {
		perror("failed to register SIGUSR1 handler");
		exit(EXIT_FAILURE);
	}
}
*/
import "C"

func main() {
	ch := make(chan os.Signal)
	signal.Notify(ch, syscall.SIGUSR2)

	C.register_handler(C.int(syscall.SIGUSR1))
	syscall.Kill(syscall.Getpid(), syscall.SIGUSR1)

	<-ch
}
