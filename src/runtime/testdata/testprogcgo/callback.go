// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

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

import (
	"fmt"
	"runtime"
)

func init() {
	register("CgoCallbackGC", CgoCallbackGC)
}

//export go_callback
func go_callback() {
	runtime.GC()
	grow()
	runtime.GC()
}

var cnt int

func grow() {
	x := 10000
	sum := 0
	if grow1(&x, &sum) == 0 {
		panic("bad")
	}
}

func grow1(x, sum *int) int {
	if *x == 0 {
		return *sum + 1
	}
	*x--
	sum1 := *sum + *x
	return grow1(x, &sum1)
}

func CgoCallbackGC() {
	const P = 100
	done := make(chan bool)
	// allocate a bunch of stack frames and spray them with pointers
	for i := 0; i < P; i++ {
		go func() {
			grow()
			done <- true
		}()
	}
	for i := 0; i < P; i++ {
		<-done
	}
	// now give these stack frames to cgo callbacks
	for i := 0; i < P; i++ {
		go func() {
			C.foo()
			done <- true
		}()
	}
	for i := 0; i < P; i++ {
		<-done
	}
	fmt.Printf("OK\n")
}
