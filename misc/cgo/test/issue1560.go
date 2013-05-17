// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
// mysleep returns the absolute start time in ms.
long long mysleep(int seconds);

// twoSleep returns the absolute start time of the first sleep
// in ms.
long long twoSleep(int);
*/
import "C"

import (
	"testing"
	"time"
)

var sleepDone = make(chan int64)

// parallelSleep returns the absolute difference between the start time
// of the two sleeps.
func parallelSleep(n int) int64 {
	t := int64(C.twoSleep(C.int(n))) - <-sleepDone
	if t < 0 {
		return -t
	}
	return t
}

//export BackgroundSleep
func BackgroundSleep(n int32) {
	go func() {
		sleepDone <- int64(C.mysleep(C.int(n)))
	}()
}

func testParallelSleep(t *testing.T) {
	sleepSec := 1
	dt := time.Duration(parallelSleep(sleepSec)) * time.Millisecond
	t.Logf("difference in start time for two sleep(%d) is %v", sleepSec, dt)
	// bug used to run sleeps in serial, producing a 2*sleepSec-second delay.
	// we detect if the start times of those sleeps are > 0.5*sleepSec-second.
	if dt >= time.Duration(sleepSec)*time.Second/2 {
		t.Fatalf("parallel %d-second sleeps slept for %f seconds", sleepSec, dt.Seconds())
	}
}
