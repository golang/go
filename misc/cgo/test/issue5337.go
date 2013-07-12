// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package cgotest

/*
#include <signal.h>
#include <pthread.h>

static void *thread1(void *p) {
	(void)p;
	pthread_kill(pthread_self(), SIGPROF);
	return NULL;
}
void test5337() {
	pthread_t tid;
	pthread_create(&tid, 0, thread1, NULL);
	pthread_join(tid, 0);
}
*/
import "C"

import "testing"

// Verify that we can withstand SIGPROF received on foreign threads
func test5337(t *testing.T) {
	C.test5337()
}
