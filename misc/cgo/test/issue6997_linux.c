// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static pthread_t thread;

static void* threadfunc(void* dummy) {
	while(1) {
		sleep(1);
	}
}

int StartThread() {
	return pthread_create(&thread, NULL, &threadfunc, NULL);
}

int CancelThread() {
	void *r;
	pthread_cancel(thread);
	pthread_join(thread, &r);
	return (r == PTHREAD_CANCELED);
}
