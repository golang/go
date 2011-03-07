// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <unistd.h>

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
func BackgroundSleep(n int){
	go func(){
		C.sleep(C.uint(n))
		sleepDone <- true
	}()
}

func TestParallelSleep(t *testing.T) {
	dt := -time.Nanoseconds()
	parallelSleep(1)
	dt += time.Nanoseconds()
	// bug used to run sleeps in serial, producing a 2-second delay.
	if dt >= 1.3e9 {
		t.Fatalf("parallel 1-second sleeps slept for %f seconds", float64(dt)/1e9)
	}
}
