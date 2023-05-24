// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program failed when run under the C/C++ ThreadSanitizer.
// There was no TSAN synchronization for the call to the cgo
// traceback routine.

/*
#cgo CFLAGS: -g -fsanitize=thread
#cgo LDFLAGS: -g -fsanitize=thread

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

struct tracebackArg {
	uintptr_t  Context;
	uintptr_t  SigContext;
	uintptr_t* Buf;
	uintptr_t  Max;
};

void tsanTraceback(struct tracebackArg *arg) {
	arg->Buf[0] = 0;
}

static void* spin(void *arg) {
	size_t n;
	struct timeval tvstart, tvnow;
	int diff;
	void *prev;
	void *cur;

	prev = NULL;
	gettimeofday(&tvstart, NULL);
	for (n = 0; n < 1<<20; n++) {
		cur = malloc(n);
		free(prev);
		prev = cur;

		gettimeofday(&tvnow, NULL);
		diff = (tvnow.tv_sec - tvstart.tv_sec) * 1000 * 1000 + (tvnow.tv_usec - tvstart.tv_usec);

		// Profile frequency is 100Hz so we should definitely
		// get some signals in 50 milliseconds.
		if (diff > 50 * 1000) {
			break;
		}
	}

	free(prev);

	return NULL;
}

static void runThreads(int n) {
	pthread_t ids[64];
	int i;

	if (n > 64) {
		n = 64;
	}
	for (i = 0; i < n; i++) {
		pthread_create(&ids[i], NULL, spin, NULL);
	}
	for (i = 0; i < n; i++) {
		pthread_join(ids[i], NULL);
	}
}
*/
import "C"

import (
	"io"
	"runtime"
	"runtime/pprof"
	"unsafe"
)

func main() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.tsanTraceback), nil, nil)
	pprof.StartCPUProfile(io.Discard)
	C.runThreads(C.int(runtime.GOMAXPROCS(0)))
	pprof.StopCPUProfile()
}
