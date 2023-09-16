// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sigaction sa;
struct sigaction osa;

static void (*oldHandler)(int, siginfo_t*, void*);

static void handler(int signo, siginfo_t* info, void* ctxt) {
	if (oldHandler) {
		oldHandler(signo, info, ctxt);
	}
}

int install_handler() {
	// Install our own signal handler.
	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
	memset(&osa, 0, sizeof osa);
	sigemptyset(&osa.sa_mask);
	if (sigaction(SIGSEGV, &sa, &osa) < 0) {
		perror("sigaction");
		return 2;
	}
	if (osa.sa_handler == SIG_DFL) {
		fprintf(stderr, "Go runtime did not install signal handler\n");
		return 2;
	}
	// gccgo does not set SA_ONSTACK for SIGSEGV.
	if (getenv("GCCGO") == NULL && (osa.sa_flags&SA_ONSTACK) == 0) {
		fprintf(stderr, "Go runtime did not install signal handler\n");
		return 2;
	}
	oldHandler = osa.sa_sigaction;

	return 0;
}

int check_handler() {
	if (sigaction(SIGSEGV, NULL, &sa) < 0) {
		perror("sigaction check");
		return 2;
	}
	if (sa.sa_sigaction != handler) {
		fprintf(stderr, "ERROR: wrong signal handler: %p != %p\n", sa.sa_sigaction, handler);
		return 2;
	}
	return 0;
}

