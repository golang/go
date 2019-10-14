// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that SIGSETXID runs on signal stack, since it's likely to
// overflow if it runs on the Go stack.

package cgotest

/*
#include <sys/types.h>
#include <unistd.h>
*/
import "C"

import (
	"runtime"
	"sync/atomic"
	"testing"

	"cgotest/issue9400"
)

func test9400(t *testing.T) {
	// We synchronize through a shared variable, so we need two procs
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))

	// Start signaller
	atomic.StoreInt32(&issue9400.Baton, 0)
	go func() {
		// Wait for RewindAndSetgid
		for atomic.LoadInt32(&issue9400.Baton) == 0 {
			runtime.Gosched()
		}
		// Broadcast SIGSETXID
		runtime.LockOSThread()
		C.setgid(0)
		// Indicate that signalling is done
		atomic.StoreInt32(&issue9400.Baton, 0)
	}()

	// Grow the stack and put down a test pattern
	const pattern = 0x123456789abcdef
	var big [1024]uint64 // len must match assembly
	for i := range big {
		big[i] = pattern
	}

	// Temporarily rewind the stack and trigger SIGSETXID
	issue9400.RewindAndSetgid()

	// Check test pattern
	for i := range big {
		if big[i] != pattern {
			t.Fatalf("entry %d of test pattern is wrong; %#x != %#x", i, big[i], uint64(pattern))
		}
	}
}
