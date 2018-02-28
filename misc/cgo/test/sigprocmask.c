// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

#include <signal.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
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
	pthread_t thread;
	int r;

	r = pthread_create(&thread, NULL, &sigthreadfunc, NULL);
	if (r != 0)
		return r;
	return pthread_join(thread, NULL);
}
