// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program failed when run under the C/C++ ThreadSanitizer.
//
// cgocallback on a new thread calls into runtime.needm -> _cgo_getstackbound
// to update gp.stack.lo with the stack bounds. If the G itself is passed to
// _cgo_getstackbound, then writes to the same G can be seen on multiple
// threads (when the G is reused after thread exit). This would trigger TSAN.

/*
#include <pthread.h>

void go_callback();

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
*/
import "C"

import (
	"time"
)

//export go_callback
func go_callback() {
}

func main() {
	for i := 0; i < 2; i++ {
		go func() {
			for {
				C.foo()
			}
		}()
	}

	time.Sleep(1000*time.Millisecond)
}
