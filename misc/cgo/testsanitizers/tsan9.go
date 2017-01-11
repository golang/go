// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program failed when run under the C/C++ ThreadSanitizer. The
// TSAN library was not keeping track of whether signals should be
// delivered on the alternate signal stack, and the Go signal handler
// was not preserving callee-saved registers from C callers.

/*
#cgo CFLAGS: -g -fsanitize=thread
#cgo LDFLAGS: -g -fsanitize=thread

#include <stdlib.h>
#include <sys/time.h>

void spin() {
	size_t n;
	struct timeval tvstart, tvnow;
	int diff;
	void *prev = NULL, *cur;

	gettimeofday(&tvstart, NULL);
	for (n = 0; n < 1<<20; n++) {
		cur = malloc(n);
		free(prev);
		prev = cur;

		gettimeofday(&tvnow, NULL);
		diff = (tvnow.tv_sec - tvstart.tv_sec) * 1000 * 1000 + (tvnow.tv_usec - tvstart.tv_usec);

		// Profile frequency is 100Hz so we should definitely
		// get a signal in 50 milliseconds.
		if (diff > 50 * 1000) {
			break;
		}
	}

	free(prev);
}
*/
import "C"

import (
	"io/ioutil"
	"runtime/pprof"
	"time"
)

func goSpin() {
	start := time.Now()
	for n := 0; n < 1<<20; n++ {
		_ = make([]byte, n)
		if time.Since(start) > 50*time.Millisecond {
			break
		}
	}
}

func main() {
	pprof.StartCPUProfile(ioutil.Discard)
	go C.spin()
	goSpin()
	pprof.StopCPUProfile()
}
