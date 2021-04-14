// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

/*
#cgo CFLAGS: -pthread
#cgo LDFLAGS: -pthread

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

int *p;
static void sigsegv() {
	*p = 1;
	fprintf(stderr, "ERROR: C SIGSEGV not thrown on caught?.\n");
	exit(2);
}

static void segvhandler(int signum) {
	if (signum == SIGSEGV) {
		fprintf(stdout, "ok\ttestsigfwd\n");
		exit(0);  // success
	}
}

static volatile sig_atomic_t sigioSeen;

// Use up some stack space.
static void recur(int i, char *p) {
	char a[1024];

	*p = '\0';
	if (i > 0) {
		recur(i - 1, a);
	}
}

static void iohandler(int signum) {
	char a[1024];

	recur(4, a);
	sigioSeen = 1;
}

static void* sigioThread(void* arg __attribute__ ((unused))) {
	raise(SIGIO);
	return NULL;
}

static void sigioOnThread() {
	pthread_t tid;
	int i;

	pthread_create(&tid, NULL, sigioThread, NULL);
	pthread_join(tid, NULL);

	// Wait until the signal has been delivered.
	i = 0;
	while (!sigioSeen) {
		if (sched_yield() < 0) {
			perror("sched_yield");
		}
		i++;
		if (i > 10000) {
			fprintf(stderr, "looping too long waiting for signal\n");
			exit(EXIT_FAILURE);
		}
	}
}

static void __attribute__ ((constructor)) sigsetup(void) {
	struct sigaction act;

	memset(&act, 0, sizeof act);
	act.sa_handler = segvhandler;
	sigaction(SIGSEGV, &act, NULL);

	act.sa_handler = iohandler;
	sigaction(SIGIO, &act, NULL);
}
*/
import "C"

var p *byte

func f() (ret bool) {
	defer func() {
		if recover() == nil {
			fmt.Errorf("ERROR: couldn't raise SIGSEGV in Go.")
			C.exit(2)
		}
		ret = true
	}()
	*p = 1
	return false
}

func main() {
	// Test that the signal originating in Go is handled (and recovered) by Go.
	if !f() {
		fmt.Errorf("couldn't recover from SIGSEGV in Go.")
		C.exit(2)
	}

	// Test that the signal originating in C is handled by C.
	C.sigsegv()
}
