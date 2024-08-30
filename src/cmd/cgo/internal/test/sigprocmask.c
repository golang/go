// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

extern void IntoGoAndBack();

int CheckBlocked() {
	sigset_t mask;
	sigprocmask(SIG_BLOCK, NULL, &mask);
	return sigismember(&mask, SIGIO);
}

static void* sigthreadfunc(void* unused) {
	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGIO);
	sigprocmask(SIG_BLOCK, &mask, NULL);
	IntoGoAndBack();
	return NULL;
}

int RunSigThread() {
	int tries;
	pthread_t thread;
	int r;
	struct timespec ts;

	for (tries = 0; tries < 20; tries++) {
		r = pthread_create(&thread, NULL, &sigthreadfunc, NULL);
		if (r == 0) {
			return pthread_join(thread, NULL);
		}
		if (r != EAGAIN) {
			return r;
		}
		ts.tv_sec = 0;
		ts.tv_nsec = (tries + 1) * 1000 * 1000; // Milliseconds.
		nanosleep(&ts, NULL);
	}
	return EAGAIN;
}
