// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// The unix C definitions for trace.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#include <pthread.h>
#include <assert.h>

extern void goCalledFromC(void);
extern void goCalledFromCThread(void);

static void* cCalledFromCThread(void *p) {
	goCalledFromCThread();
	return NULL;
}

void cCalledFromGo(void) {
	goCalledFromC();

	pthread_t thread;
	assert(pthread_create(&thread, NULL, cCalledFromCThread, NULL) == 0);
	assert(pthread_join(thread, NULL) == 0);
}
