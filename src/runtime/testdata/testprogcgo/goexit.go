// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows

package main

/*
#include <pthread.h>

void go_callback2();

static void *thr2(void *arg) {
    go_callback2();
    return 0;
}

static void foo3() {
    pthread_t th;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 << 10);
    pthread_create(&th, &attr, thr2, 0);
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
	C.foo3()
}

//export go_callback2
func go_callback2() {
	runtime.Goexit()
}
