// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program failed when run under the C/C++ ThreadSanitizer.  The TSAN
// sigaction function interceptor returned SIG_DFL instead of the Go runtime's
// handler in registerSegvForwarder.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sigaction prev_sa;

void forwardSignal(int signo, siginfo_t *info, void *context) {
	// One of sa_sigaction and/or sa_handler
	if ((prev_sa.sa_flags&SA_SIGINFO) != 0) {
		prev_sa.sa_sigaction(signo, info, context);
		return;
	}
	if (prev_sa.sa_handler != SIG_IGN && prev_sa.sa_handler != SIG_DFL) {
		prev_sa.sa_handler(signo);
		return;
	}

	fprintf(stderr, "No Go handler to forward to!\n");
	abort();
}

void registerSegvFowarder() {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
	sa.sa_sigaction = forwardSignal;

	if (sigaction(SIGSEGV, &sa, &prev_sa) != 0) {
		perror("failed to register SEGV forwarder");
		exit(EXIT_FAILURE);
	}
}
*/
import "C"

func main() {
	C.registerSegvFowarder()

	defer func() {
		recover()
	}()
	var nilp *int
	*nilp = 42
}
