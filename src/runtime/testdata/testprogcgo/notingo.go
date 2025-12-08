// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows

package main

/*
#include <stdatomic.h>
#include <stddef.h>
#include <pthread.h>

extern void Ready();

static _Atomic int spinning;
static _Atomic int released;

static void* enterGoThenSpinTwice(void* arg __attribute__ ((unused))) {
	Ready();
	atomic_fetch_add(&spinning, 1);
	while(atomic_load(&released) == 0) {};

	Ready();
	atomic_fetch_add(&spinning, 1);
	while(1) {};
	return NULL;
}

static void SpinTwiceInNewCThread() {
	pthread_t tid;
	pthread_create(&tid, NULL, enterGoThenSpinTwice, NULL);
}

static int Spinning() {
	return atomic_load(&spinning);
}

static void Release() {
	atomic_store(&spinning, 0);
	atomic_store(&released, 1);
}
*/
import "C"

import (
	"os"
	"runtime"
	"runtime/metrics"
)

func init() {
	register("NotInGoMetricCallback", NotInGoMetricCallback)
}

func NotInGoMetricCallback() {
	const N = 10
	s := []metrics.Sample{{Name: "/sched/goroutines/not-in-go:goroutines"}}

	// Create N new C threads that have called into Go at least once.
	for range N {
		C.SpinTwiceInNewCThread()
	}

	// Synchronize with spinning threads twice.
	//
	// This helps catch bad accounting by taking at least a couple other
	// codepaths which would cause the accounting to change.
	for i := range 2 {
		// Make sure they pass through Go.
		// N.B. Ready is called twice by the new threads.
		for j := range N {
			<-readyCh
			if j == 2 {
				// Try to trigger an update in the immediate STW handoff case.
				runtime.ReadMemStats(&m)
			}
		}

		// Make sure they're back in C.
		for C.Spinning() < N {
		}

		// Do something that stops the world to take all the Ps back.
		runtime.ReadMemStats(&m)

		if i == 0 {
			C.Release()
		}
	}

	// Read not-in-go.
	metrics.Read(s)
	if n := s[0].Value.Uint64(); n != 0 {
		println("expected 0 not-in-go goroutines, found", n)
		os.Exit(2)
	}
	println("OK")
}

var m runtime.MemStats
var readyCh = make(chan bool)

//export Ready
func Ready() {
	readyCh <- true
}
