// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that locks don't go quadratic due to runtime hash table collisions.

package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

const debug = false

// checkLinear asserts that the running time of f(n) is at least linear but sub-quadratic.
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

	n := tries
	fails := 0
	for {
		t1 := timeF(n)
		t2 := timeF(2 * n)
		if debug {
			println(n, t1.String(), 2*n, t2.String())
		}
		// should be 2x (linear); allow up to 2.5x
		if t1*3/2 < t2 && t2 < t1*5/2 {
			return
		}
		// If 2n ops run in under a second and the ratio
		// doesn't work out, make n bigger, trying to reduce
		// the effect that a constant amount of overhead has
		// on the computed ratio.
		if t2 < 1*time.Second {
			n *= 2
			continue
		}
		// Once the test runs long enough for n ops,
		// try to get the right ratio at least once.
		// If five in a row all fail, give up.
		if fails++; fails >= 5 {
			panic(fmt.Sprintf("%s: too slow: %d ops: %v; %d ops: %v\n",
				typ, n, t1, 2*n, t2))
		}
	}
}

const offset = 251 // known size of runtime hash table

func main() {
	checkLinear("lockone", 1000, func(n int) {
		ch := make(chan int)
		locks := make([]sync.RWMutex, offset+1)
		for i := 0; i < n; i++ {
			go func() {
				locks[0].Lock()
				ch <- 1
			}()
		}
		time.Sleep(1 * time.Millisecond)

		go func() {
			for j := 0; j < n; j++ {
				locks[1].Lock()
				locks[offset].Lock()
				locks[1].Unlock()
				runtime.Gosched()
				locks[offset].Unlock()
			}
		}()

		for j := 0; j < n; j++ {
			locks[1].Lock()
			locks[offset].Lock()
			locks[1].Unlock()
			runtime.Gosched()
			locks[offset].Unlock()
		}

		for i := 0; i < n; i++ {
			<-ch
			locks[0].Unlock()
		}
	})

	checkLinear("lockmany", 1000, func(n int) {
		locks := make([]sync.RWMutex, n*offset+1)

		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				locks[(i+1)*offset].Lock()
				wg.Done()
				locks[(i+1)*offset].Lock()
				locks[(i+1)*offset].Unlock()
			}(i)
		}
		wg.Wait()

		go func() {
			for j := 0; j < n; j++ {
				locks[1].Lock()
				locks[0].Lock()
				locks[1].Unlock()
				runtime.Gosched()
				locks[0].Unlock()
			}
		}()

		for j := 0; j < n; j++ {
			locks[1].Lock()
			locks[0].Lock()
			locks[1].Unlock()
			runtime.Gosched()
			locks[0].Unlock()
		}

		for i := 0; i < n; i++ {
			locks[(i+1)*offset].Unlock()
		}
	})
}
