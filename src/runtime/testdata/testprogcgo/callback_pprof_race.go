// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows

package main

// Test program for TestCgoCallbackPprofRace: high-frequency C→Go
// callbacks under continuous CPU profiling, exercising the #70529
// race window in cgocallbackg.

/*
#include <pthread.h>
#include <sched.h>

extern void GoCallback70529();

static volatile int stop70529 = 0;

static void *callback_race_worker(void *arg) {
    while (!stop70529) {
        GoCallback70529();
        sched_yield();
    }
    return 0;
}

static void start_callback_race_workers(int n) {
    pthread_t tids[16];
    for (int i = 0; i < n && i < 16; i++) {
        pthread_create(&tids[i], 0, callback_race_worker, 0);
    }
}

static void stop_callback_race_workers(void) {
    stop70529 = 1;
}
*/
import "C"

import (
	"bytes"
	"fmt"
	"runtime/pprof"
	"sync/atomic"
	"time"
)

func init() {
	register("CgoCallbackPprofRace", CgoCallbackPprofRace)
}

func CgoCallbackPprofRace() {
	C.start_callback_race_workers(8)

	for round := 0; round < 50; round++ {
		var buf bytes.Buffer
		if err := pprof.StartCPUProfile(&buf); err != nil {
			continue
		}
		time.Sleep(200 * time.Millisecond)
		pprof.StopCPUProfile()
	}

	C.stop_callback_race_workers()
	time.Sleep(100 * time.Millisecond)

	fmt.Printf("OK\n")
}

//export GoCallback70529
func GoCallback70529() {
	atomic.AddUint64(&callbackSink70529, 1)
}

var callbackSink70529 uint64
