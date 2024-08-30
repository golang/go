// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package main

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

import "runtime"

func init() {
	register("CgoToGoCallGoexit", func() {
		println("expect: runtime.Goexit called in a thread that was not created by the Go runtime")
		CgoToGoCallGoexit()
	})
}

func CgoToGoCallGoexit() {
	C.foo()
}

//export go_callback
func go_callback() {
	runtime.Goexit()
}
