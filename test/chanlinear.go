// +build darwin linux
// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that dequeueing from a pending channel doesn't
// take linear time.

package main

import (
	"fmt"
	"runtime"
	"time"
)

// checkLinear asserts that the running time of f(n) is in O(n).
// tries is the initial number of iterations.
func checkLinear(typ string, tries int, f func(n int)) {
	// Depending on the machine and OS, this test might be too fast
	// to measure with accurate enough granularity. On failure,
	// make it run longer, hoping that the timing granularity
	// is eventually sufficient.

	timeF := func(n int) time.Duration {
		t1 := time.Now()
		f(n)
		return time.Since(t1)
	}

	t0 := time.Now()

	n := tries
	fails := 0
	for {
		runtime.GC()
		t1 := timeF(n)
		runtime.GC()
		t2 := timeF(2 * n)

		// should be 2x (linear); allow up to 3x
		if t2 < 3*t1 {
			if false {
				fmt.Println(typ, "\t", time.Since(t0))
			}
			return
		}
		// If n ops run in under a second and the ratio
		// doesn't work out, make n bigger, trying to reduce
		// the effect that a constant amount of overhead has
		// on the computed ratio.
		if t1 < 1*time.Second {
			n *= 2
			continue
		}
		// Once the test runs long enough for n ops,
		// try to get the right ratio at least once.
		// If five in a row all fail, give up.
		if fails++; fails >= 5 {
			panic(fmt.Sprintf("%s: too slow: %d channels: %v; %d channels: %v\n",
				typ, n, t1, 2*n, t2))
		}
	}
}

func main() {
	checkLinear("chanSelect", 1000, func(n int) {
		const messages = 10
		c := make(chan bool) // global channel
		var a []chan bool    // local channels for each goroutine
		for i := 0; i < n; i++ {
			d := make(chan bool)
			a = append(a, d)
			go func() {
				for j := 0; j < messages; j++ {
					// queue ourselves on the global channel
					select {
					case <-c:
					case <-d:
					}
				}
			}()
		}
		for i := 0; i < messages; i++ {
			// wake each goroutine up, forcing it to dequeue and then enqueue
			// on the global channel.
			for _, d := range a {
				d <- true
			}
		}
	})
}
