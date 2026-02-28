// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stress test setgid and thread creation. A thread
// can get a SIGSETXID signal early on at thread
// initialization, causing crash. See issue 53374.

package cgotest

/*
#include <sys/types.h>
#include <unistd.h>
*/
import "C"

import (
	"runtime"
	"testing"
)

func testSetgidStress(t *testing.T) {
	const N = 50
	ch := make(chan int, N)
	for i := 0; i < N; i++ {
		go func() {
			C.setgid(0)
			ch <- 1
			runtime.LockOSThread() // so every goroutine uses a new thread
		}()
	}
	for i := 0; i < N; i++ {
		<-ch
	}
}
