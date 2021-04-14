// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that locks don't go quadratic due to runtime hash table collisions.

package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
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
	var buf bytes.Buffer
	inversions := 0
	for {
		t1 := timeF(n)
		t2 := timeF(2 * n)
		if debug {
			println(n, t1.String(), 2*n, t2.String())
		}
		fmt.Fprintf(&buf, "%d %v %d %v (%.1fX)\n", n, t1, 2*n, t2, float64(t2)/float64(t1))
		// should be 2x (linear); allow up to 3x
		if t1*3/2 < t2 && t2 < t1*3 {
			return
		}
		if t2 < t1 {
			if inversions++; inversions >= 5 {
				// The system must be overloaded (some builders). Give up.
				return
			}
			continue // try again; don't increment fails
		}
		// Once the test runs long enough for n ops,
		// try to get the right ratio at least once.
		// If many in a row all fail, give up.
		if fails++; fails >= 5 {
			// If 2n ops run in under a second and the ratio
			// doesn't work out, make n bigger, trying to reduce
			// the effect that a constant amount of overhead has
			// on the computed ratio.
			if t2 < time.Second*4/10 {
				fails = 0
				n *= 2
				continue
			}
			panic(fmt.Sprintf("%s: too slow: %d ops: %v; %d ops: %v\n\n%s",
				typ, n, t1, 2*n, t2, buf.String()))
		}
	}
}

const offset = 251 // known size of runtime hash table

const profile = false

func main() {
	if profile {
		f, err := os.Create("lock.prof")
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

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

	if runtime.GOARCH == "arm" && os.Getenv("GOARM") == "5" {
		// lockmany reliably fails on the linux-arm-arm5spacemonkey
		// builder. See https://golang.org/issue/24221.
		return
	}

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
