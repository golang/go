// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows

#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include "_cgo_export.h"

#define CTHREADS 2
#define CHECKCALLS 100

static void* checkBindMThread(void* thread) {
	int i;
	for (i = 0; i < CHECKCALLS; i++) {
		GoCheckBindM((uintptr_t)thread);
		usleep(1);
	}
	return NULL;
}

void CheckBindM() {
	int i;
	pthread_t s[CTHREADS];

	for (i = 0; i < CTHREADS; i++) {
		pthread_create(&s[i], NULL, checkBindMThread, &s[i]);
	}
	for (i = 0; i < CTHREADS; i++) {
		pthread_join(s[i], NULL);
	}
}
