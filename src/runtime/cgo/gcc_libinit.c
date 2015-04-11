// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd
// +build !ppc64,!ppc64le

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // strerror

static pthread_cond_t runtime_init_cond;
static pthread_mutex_t runtime_init_mu;
static int runtime_init_done;

void
x_cgo_sys_thread_create(void* (*func)(void*), void* arg) {
	pthread_t p;
	int err = pthread_create(&p, NULL, func, arg);
	if (err != 0) {
		fprintf(stderr, "pthread_create failed: %s", strerror(err));
		abort();
	}
}

void
_cgo_wait_runtime_init_done() {
	pthread_mutex_lock(&runtime_init_mu);
	while (runtime_init_done == 0) {
		pthread_cond_wait(&runtime_init_cond, &runtime_init_mu);
	}
	pthread_mutex_unlock(&runtime_init_mu);
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	pthread_mutex_lock(&runtime_init_mu);
	runtime_init_done = 1;
	pthread_cond_broadcast(&runtime_init_cond);
	pthread_mutex_unlock(&runtime_init_mu);
}