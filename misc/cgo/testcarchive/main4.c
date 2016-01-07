// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test a C thread that calls sigaltstack and then calls Go code.

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>

#include "libgo4.h"

static void die(const char* msg) {
	perror(msg);
	exit(EXIT_FAILURE);
}

static int ok = 1;

static void ioHandler(int signo, siginfo_t* info, void* ctxt) {
}

// Set up the SIGIO signal handler in a high priority constructor, so
// that it is installed before the Go code starts.

static void init(void) __attribute__ ((constructor (200)));

static void init() {
	struct sigaction sa;

	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = ioHandler;
	if (sigemptyset(&sa.sa_mask) < 0) {
		die("sigemptyset");
	}
	sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
	if (sigaction(SIGIO, &sa, NULL) < 0) {
		die("sigaction");
	}
}

// Test raising SIGIO on a C thread with an alternate signal stack
// when there is a Go signal handler for SIGIO.
static void* thread1(void* arg) {
	pthread_t* ptid = (pthread_t*)(arg);
	stack_t ss;
	int i;
	stack_t nss;

	// Set up an alternate signal stack for this thread.
	memset(&ss, 0, sizeof ss);
	ss.ss_sp = malloc(SIGSTKSZ);
	if (ss.ss_sp == NULL) {
		die("malloc");
	}
	ss.ss_flags = 0;
	ss.ss_size = SIGSTKSZ;
	if (sigaltstack(&ss, NULL) < 0) {
		die("sigaltstack");
	}

	// Send ourselves a SIGIO.  This will be caught by the Go
	// signal handler which should forward to the C signal
	// handler.
	i = pthread_kill(*ptid, SIGIO);
	if (i != 0) {
		fprintf(stderr, "pthread_kill: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	// Wait until the signal has been delivered.
	i = 0;
	while (SIGIOCount() == 0) {
		if (sched_yield() < 0) {
			perror("sched_yield");
		}
		i++;
		if (i > 100000) {
			fprintf(stderr, "looping too long waiting for signal\n");
			exit(EXIT_FAILURE);
		}
	}

	// We should still be on the same signal stack.
	if (sigaltstack(NULL, &nss) < 0) {
		die("sigaltstack check");
	}
	if ((nss.ss_flags & SS_DISABLE) != 0) {
		fprintf(stderr, "sigaltstack disabled on return from Go\n");
		ok = 0;
	} else if (nss.ss_sp != ss.ss_sp) {
		fprintf(stderr, "sigalstack changed on return from Go\n");
		ok = 0;
	}

	return NULL;
}

// Test calling a Go function to raise SIGIO on a C thread with an
// alternate signal stack when there is a Go signal handler for SIGIO.
static void* thread2(void* arg) {
	pthread_t* ptid = (pthread_t*)(arg);
	stack_t ss;
	int i;
	int oldcount;
	stack_t nss;

	// Set up an alternate signal stack for this thread.
	memset(&ss, 0, sizeof ss);
	ss.ss_sp = malloc(SIGSTKSZ);
	if (ss.ss_sp == NULL) {
		die("malloc");
	}
	ss.ss_flags = 0;
	ss.ss_size = SIGSTKSZ;
	if (sigaltstack(&ss, NULL) < 0) {
		die("sigaltstack");
	}

	oldcount = SIGIOCount();

	// Call a Go function that will call a C function to send us a
	// SIGIO.
	GoRaiseSIGIO(ptid);

	// Wait until the signal has been delivered.
	i = 0;
	while (SIGIOCount() == oldcount) {
		if (sched_yield() < 0) {
			perror("sched_yield");
		}
		i++;
		if (i > 100000) {
			fprintf(stderr, "looping too long waiting for signal\n");
			exit(EXIT_FAILURE);
		}
	}

	// We should still be on the same signal stack.
	if (sigaltstack(NULL, &nss) < 0) {
		die("sigaltstack check");
	}
	if ((nss.ss_flags & SS_DISABLE) != 0) {
		fprintf(stderr, "sigaltstack disabled on return from Go\n");
		ok = 0;
	} else if (nss.ss_sp != ss.ss_sp) {
		fprintf(stderr, "sigalstack changed on return from Go\n");
		ok = 0;
	}

	return NULL;
}

int main(int argc, char **argv) {
	pthread_t tid;
	int i;

	// Tell the Go library to start looking for SIGIO.
	GoCatchSIGIO();

	i = pthread_create(&tid, NULL, thread1, (void*)(&tid));
	if (i != 0) {
		fprintf(stderr, "pthread_create: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	i = pthread_join(tid, NULL);
	if (i != 0) {
		fprintf(stderr, "pthread_join: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	i = pthread_create(&tid, NULL, thread2, (void*)(&tid));
	if (i != 0) {
		fprintf(stderr, "pthread_create: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	i = pthread_join(tid, NULL);
	if (i != 0) {
		fprintf(stderr, "pthread_join: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	if (!ok) {
		exit(EXIT_FAILURE);
	}

	printf("PASS\n");
	return 0;
}
