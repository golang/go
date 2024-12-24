// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests CPU profiling.

//go:build ignore

package main

/*
#include <pthread.h>

void go_callback();
void go_callback2();

static void *thr(void *arg) {
    go_callback();
    return 0;
}

static void foo() {
    pthread_t th;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 << 10);
    pthread_create(&th, &attr, thr, 0);
    pthread_join(th, 0);
}

static void bar() {
    go_callback2();
}
*/
import "C"

import (
	"log"
	"os"
	"runtime"
	"runtime/trace"
)

//export go_callback
func go_callback() {
	// Do another call into C, just to test that path too.
	C.bar()
}

//export go_callback2
func go_callback2() {
	runtime.GC()
}

func main() {
	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	// Do a whole bunch of cgocallbacks.
	const n = 10
	done := make(chan bool)
	for i := 0; i < n; i++ {
		go func() {
			C.foo()
			done <- true
		}()
	}
	for i := 0; i < n; i++ {
		<-done
	}

	// Do something to steal back any Ps from the Ms, just
	// for coverage.
	runtime.GC()

	// End of traced execution.
	trace.Stop()
}
