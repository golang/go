// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <unistd.h>

unsigned int sleep(unsigned int seconds);

extern void BackgroundSleep(int);
void twoSleep(int);
*/
import "C"

import (
	"runtime"
	"testing"
	"time"
)

var sleepDone = make(chan bool)

func parallelSleep(n int) {
	C.twoSleep(C.int(n))
	<-sleepDone
}

//export BackgroundSleep
func BackgroundSleep(n int32) {
	go func() {
		C.sleep(C.uint(n))
		sleepDone <- true
	}()
}

// wasteCPU starts a background goroutine to waste CPU
// to cause the power management to raise the CPU frequency.
// On ARM this has the side effect of making sleep more accurate.
func wasteCPU() chan struct{} {
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-done:
				return
			default:
			}
		}
	}()
	// pause for a short amount of time to allow the
	// power management to recognise load has risen.
	<-time.After(300 * time.Millisecond)
	return done
}

func testParallelSleep(t *testing.T) {
	if runtime.GOARCH == "arm" {
		// on ARM, the 1.3s deadline is frequently missed,
		// and burning cpu seems to help
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
		defer close(wasteCPU())
	}

	sleepSec := 1
	start := time.Now()
	parallelSleep(sleepSec)
	dt := time.Since(start)
	t.Logf("sleep(%d) slept for %v", sleepSec, dt)
	// bug used to run sleeps in serial, producing a 2*sleepSec-second delay.
	if dt >= time.Duration(sleepSec)*1300*time.Millisecond {
		t.Fatalf("parallel %d-second sleeps slept for %f seconds", sleepSec, dt.Seconds())
	}
}
