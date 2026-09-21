// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
)

func init() {
	register("mexitSTW", mexitSTW)
}

// Stress test for pp.oldm pointing to an exited M.
//
// If pp.oldm points to an exited M it should be ignored and another M used
// instead. To stress:
//
// 1. Start and exit many threads (thus setting oldm on some P).
// 2. Meanwhile, frequently stop the world.
//
// If procresize incorrect attempts to assign a P to an exited M, likely
// failure modes are:
//
// 1. Crash in startTheWorldWithSema attempting to access the M, if it is nil.
//
// 2. Memory corruption elsewhere after startTheWorldWithSema writes to the M,
// if it is not nil, but is freed and reused for another allocation.
//
// 3. Hang on a subsequent stop the world waiting for the P to stop, if the M
// object is valid, but the M is exited, because startTheWorldWithSema didn't
// actually wake anything to run the P. The P is _Pidle, but not in the pidle
// list, thus startTheWorldWithSema will wake for it to actively stop.
//
// For this to go wrong, an exited M must fail to clear mp.self and must leave
// the M on the sched.midle list.
//
// Similar to TraceSTW.
func mexitSTW() {
	// Ensure we have multiple Ps, but not too many, as we want the
	// runnable goroutines likely to run on Ps with oldm set.
	runtime.GOMAXPROCS(4)

	// Background busy work so there is always something runnable.
	for i := range 2 {
		go traceSTWTarget(i)
	}

	// Wait for children to start running.
	ping.Store(1)
	for pong[0].Load() != 1 {}
	for pong[1].Load() != 1 {}

	for range 100 {
		// Exit a thread. The last P to run this will have it in oldm.
		go func() {
			runtime.LockOSThread()
		}()

		// STW
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
	}

	stop.Store(true)

	println("OK")
}
