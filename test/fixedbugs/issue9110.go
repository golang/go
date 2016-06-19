// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Scenario that used to leak arbitrarily many SudoG structs.
// See golang.org/issue/9110.

package main

import (
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

func main() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(1000000) // only GC when we ask for GC

	var stats, stats1, stats2 runtime.MemStats

	release := func() {}
	for i := 0; i < 20; i++ {
		if i == 10 {
			// Should be warmed up by now.
			runtime.ReadMemStats(&stats1)
		}

		c := make(chan int)
		for i := 0; i < 10; i++ {
			go func() {
				select {
				case <-c:
				case <-c:
				case <-c:
				}
			}()
		}
		time.Sleep(1 * time.Millisecond)
		release()

		close(c) // let select put its sudog's into the cache
		time.Sleep(1 * time.Millisecond)

		// pick up top sudog
		var cond1 sync.Cond
		var mu1 sync.Mutex
		cond1.L = &mu1
		go func() {
			mu1.Lock()
			cond1.Wait()
			mu1.Unlock()
		}()
		time.Sleep(1 * time.Millisecond)

		// pick up next sudog
		var cond2 sync.Cond
		var mu2 sync.Mutex
		cond2.L = &mu2
		go func() {
			mu2.Lock()
			cond2.Wait()
			mu2.Unlock()
		}()
		time.Sleep(1 * time.Millisecond)

		// put top sudog back
		cond1.Broadcast()
		time.Sleep(1 * time.Millisecond)

		// drop cache on floor
		runtime.GC()

		// release cond2 after select has gotten to run
		release = func() {
			cond2.Broadcast()
			time.Sleep(1 * time.Millisecond)
		}
	}

	runtime.GC()

	runtime.ReadMemStats(&stats2)

	if int(stats2.HeapObjects)-int(stats1.HeapObjects) > 20 { // normally at most 1 or 2; was 300 with leak
		print("BUG: object leak: ", stats.HeapObjects, " -> ", stats1.HeapObjects, " -> ", stats2.HeapObjects, "\n")
	}
}
