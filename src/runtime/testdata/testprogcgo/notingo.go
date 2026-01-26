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
extern void BlockForeverInGo();

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

static void* enterGoThenWait(void* arg __attribute__ ((unused))) {
	BlockForeverInGo();
	return NULL;
}

static void WaitInGoInNewCThread() {
	pthread_t tid;
	pthread_create(&tid, NULL, enterGoThenWait, NULL);
}

static void SpinForever() {
	atomic_fetch_add(&spinning, 1);
	while(1) {};
}
*/
import "C"

import (
	"os"
	"runtime"
	"runtime/metrics"
	"sync/atomic"
)

func init() {
	register("NotInGoMetricCgoCall", NotInGoMetricCgoCall)
	register("NotInGoMetricCgoCallback", NotInGoMetricCgoCallback)
	register("NotInGoMetricCgoCallAndCallback", NotInGoMetricCgoCallAndCallback)
}

// NotInGoMetric just double-checks that N goroutines in cgo count as the metric reading N.
func NotInGoMetricCgoCall() {
	const N = 10

	// Spin up the same number of goroutines that will all wait in a cgo call.
	for range N {
		go func() {
			C.SpinForever()
		}()
	}

	// Make sure we're all blocked and spinning.
	for C.Spinning() < N {
	}

	// Read not-in-go before taking the Ps back.
	s := []metrics.Sample{{Name: "/sched/goroutines/not-in-go:goroutines"}}
	failed := false
	metrics.Read(s)
	if n := s[0].Value.Uint64(); n != N {
		println("pre-STW: expected", N, "not-in-go goroutines, found", n)
	}

	// Do something that stops the world to take all the Ps back.
	//
	// This will force a re-accounting of some of the goroutines and
	// re-checking not-in-go will help catch bugs.
	runtime.ReadMemStats(&m)

	// Read not-in-go.
	metrics.Read(s)
	if n := s[0].Value.Uint64(); n != N {
		println("post-STW: expected", N, "not-in-go goroutines, found", n)
	}

	// Fail if we get a bad reading.
	if failed {
		os.Exit(2)
	}
	println("OK")
}

// NotInGoMetricCgoCallback tests that threads that called into Go, then returned
// to C with *no* Go on the stack, are *not* counted as not-in-go in the
// runtime/metrics package.
func NotInGoMetricCgoCallback() {
	const N = 10

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
	s := []metrics.Sample{{Name: "/sched/goroutines/not-in-go:goroutines"}}
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

// NotInGoMetricCgoCallAndCallback tests that threads that called into Go are not
// keeping the count of not-in-go threads negative. Specifically, needm sets
// isExtraInC to false, breaking some of the invariants behind the not-in-go
// runtime/metrics metric, causing the underlying count to break if we don't
// account for this. In go.dev/cl/726964 this amounts to nGsyscallNoP being negative.
// Unfortunately the runtime/metrics package masks a negative nGsyscallNoP because
// it can transiently go negative due to a race. Therefore, this test checks
// the condition by making sure not-in-go is positive when we expect it to be.
// That is, threads in a cgo callback are *not* cancelling out threads in a
// regular cgo call.
func NotInGoMetricCgoCallAndCallback() {
	const N = 10

	// Spin up some threads that will do a cgo callback and just wait in Go.
	// These threads are the ones we're worried about having the incorrect
	// accounting that skews the count later.
	for range N {
		C.WaitInGoInNewCThread()
	}

	// Spin up the same number of goroutines that will all wait in a cgo call.
	for range N {
		go func() {
			C.SpinForever()
		}()
	}

	// Make sure we're all blocked and spinning.
	for C.Spinning() < N || blockedForever.Load() < N {
	}

	// Read not-in-go before taking the Ps back.
	s := []metrics.Sample{{Name: "/sched/goroutines/not-in-go:goroutines"}}
	failed := false
	metrics.Read(s)
	if n := s[0].Value.Uint64(); n != N {
		println("pre-STW: expected", N, "not-in-go goroutines, found", n)
	}

	// Do something that stops the world to take all the Ps back.
	//
	// This will force a re-accounting of some of the goroutines and
	// re-checking not-in-go will help catch bugs.
	runtime.ReadMemStats(&m)

	// Read not-in-go.
	metrics.Read(s)
	if n := s[0].Value.Uint64(); n != N {
		println("post-STW: expected", N, "not-in-go goroutines, found", n)
	}

	// Fail if we get a bad reading.
	if failed {
		os.Exit(2)
	}
	println("OK")
}

var blockedForever atomic.Uint32

//export BlockForeverInGo
func BlockForeverInGo() {
	blockedForever.Add(1)
	select {}
}
