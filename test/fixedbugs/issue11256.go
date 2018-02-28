// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that stack barriers are reset when a goroutine exits without
// returning.

package main

import (
	"runtime"
	"sync/atomic"
	"time"
)

func main() {
	// Let the garbage collector run concurrently.
	runtime.GOMAXPROCS(2)

	var x [100][]byte

	for i := range x {
		var done int32

		go func() {
			// Use enough stack to get stack barriers, but
			// not so much that we go over _FixedStack.
			// There's a very narrow window here on most
			// OSs, so we basically can't do anything (not
			// even a time.Sleep or a channel).
			var buf [1024]byte
			buf[0]++
			for atomic.LoadInt32(&done) == 0 {
				runtime.Gosched()
			}
			atomic.StoreInt32(&done, 0)
			// Exit without unwinding stack barriers.
			runtime.Goexit()
		}()

		// Generate some garbage.
		x[i] = make([]byte, 1024*1024)

		// Give GC some time to install stack barriers in the G.
		time.Sleep(50 * time.Microsecond)
		atomic.StoreInt32(&done, 1)
		for atomic.LoadInt32(&done) == 1 {
			runtime.Gosched()
		}
	}
}
