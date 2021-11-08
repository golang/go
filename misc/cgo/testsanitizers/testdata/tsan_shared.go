// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program failed with SIGSEGV when run under the C/C++ ThreadSanitizer.
// The Go runtime had re-registered the C handler with the wrong flags due to a
// typo, resulting in null pointers being passed for the info and context
// parameters to the handler.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

void check_params(int signo, siginfo_t *info, void *context) {
	ucontext_t* uc = (ucontext_t*)(context);

	if (info->si_signo != signo) {
		fprintf(stderr, "info->si_signo does not match signo.\n");
		abort();
	}

	if (uc->uc_stack.ss_size == 0) {
		fprintf(stderr, "uc_stack has size 0.\n");
		abort();
	}
}


// Set up the signal handler in a high priority constructor, so
// that it is installed before the Go code starts.

static void register_handler(void) __attribute__ ((constructor (200)));

static void register_handler() {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_SIGINFO;
	sa.sa_sigaction = check_params;

	if (sigaction(SIGUSR1, &sa, NULL) != 0) {
		perror("failed to register SIGUSR1 handler");
		exit(EXIT_FAILURE);
	}
}
*/
import "C"

import "syscall"

func init() {
	C.raise(C.int(syscall.SIGUSR1))
}

func main() {}
