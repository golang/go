// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test case for issue 66427.
// Running under TSAN, this fails with "signal handler
// spoils errno".

/*
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>

void go_callback();

static void *thr(void *arg) {
	int i;
	for (i = 0; i < 10; i++)
		go_callback();
	return 0;
}

static void *sendthr(void *arg) {
	pthread_t th = *(pthread_t*)arg;
	while (1) {
		int r = pthread_kill(th, SIGWINCH);
		if (r < 0)
			break;
	}
	return 0;
}

static void foo() {
	pthread_t *th = malloc(sizeof(pthread_t));
	pthread_t th2;
	pthread_create(th, 0, thr, 0);
	pthread_create(&th2, 0, sendthr, th);
	pthread_join(*th, 0);
}
*/
import "C"

import (
	"time"
)

//export go_callback
func go_callback() {}

func main() {
	go func() {
		for {
			C.foo()
		}
	}()

	time.Sleep(1000 * time.Millisecond)
}
