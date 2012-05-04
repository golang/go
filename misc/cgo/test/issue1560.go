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
func BackgroundSleep(n int) {
	go func() {
		C.sleep(C.uint(n))
		sleepDone <- true
	}()
}

func testParallelSleep(t *testing.T) {
	sleepSec := 1
	if runtime.GOARCH == "arm" {
		// on ARM, the 1.3s deadline is frequently missed,
		// so increase sleep time to 2s
		sleepSec = 2
	}
	start := time.Now()
	parallelSleep(sleepSec)
	dt := time.Now().Sub(start)
	// bug used to run sleeps in serial, producing a 2*sleepSec-second delay.
	if dt >= time.Duration(sleepSec)*1300*time.Millisecond {
		t.Fatalf("parallel %d-second sleeps slept for %f seconds", sleepSec, dt.Seconds())
	}
}
