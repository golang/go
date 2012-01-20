// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <unistd.h>

unsigned int sleep(unsigned int seconds);

extern void BackgroundSleep(int);
void twoSleep(int n) {
	BackgroundSleep(n);
	sleep(n);
}
*/
import "C"

import (
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
	start := time.Now()
	parallelSleep(1)
	dt := time.Now().Sub(start)
	// bug used to run sleeps in serial, producing a 2-second delay.
	if dt >= 1300*time.Millisecond {
		t.Fatalf("parallel 1-second sleeps slept for %f seconds", dt.Seconds())
	}
}
