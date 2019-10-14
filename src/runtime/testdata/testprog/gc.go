// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"time"
)

func init() {
	register("GCFairness", GCFairness)
	register("GCFairness2", GCFairness2)
	register("GCSys", GCSys)
	register("GCPhys", GCPhys)
	register("DeferLiveness", DeferLiveness)
}

func GCSys() {
	runtime.GOMAXPROCS(1)
	memstats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memstats)
	sys := memstats.Sys

	runtime.MemProfileRate = 0 // disable profiler

	itercount := 100000
	for i := 0; i < itercount; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	// We allocated 100 MB or (if not short) 1 GB.
	runtime.ReadMemStats(memstats)
	if sys > memstats.Sys {
		sys = 0
	} else {
		sys = memstats.Sys - sys
	}
	if sys > 16<<20 {
		fmt.Printf("using too much memory: %d bytes\n", sys)
		return
	}
	fmt.Printf("OK\n")
}

var sink []byte

func workthegc() []byte {
	sink = make([]byte, 1029)
	return sink
}

func GCFairness() {
	runtime.GOMAXPROCS(1)
	f, err := os.Open("/dev/null")
	if os.IsNotExist(err) {
		// This test tests what it is intended to test only if writes are fast.
		// If there is no /dev/null, we just don't execute the test.
		fmt.Println("OK")
		return
	}
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	for i := 0; i < 2; i++ {
		go func() {
			for {
				f.Write([]byte("."))
			}
		}()
	}
	time.Sleep(10 * time.Millisecond)
	fmt.Println("OK")
}

func GCFairness2() {
	// Make sure user code can't exploit the GC's high priority
	// scheduling to make scheduling of user code unfair. See
	// issue #15706.
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(1)
	var count [3]int64
	var sink [3]interface{}
	for i := range count {
		go func(i int) {
			for {
				sink[i] = make([]byte, 1024)
				atomic.AddInt64(&count[i], 1)
			}
		}(i)
	}
	// Note: If the unfairness is really bad, it may not even get
	// past the sleep.
	//
	// If the scheduling rules change, this may not be enough time
	// to let all goroutines run, but for now we cycle through
	// them rapidly.
	//
	// OpenBSD's scheduler makes every usleep() take at least
	// 20ms, so we need a long time to ensure all goroutines have
	// run. If they haven't run after 30ms, give it another 1000ms
	// and check again.
	time.Sleep(30 * time.Millisecond)
	var fail bool
	for i := range count {
		if atomic.LoadInt64(&count[i]) == 0 {
			fail = true
		}
	}
	if fail {
		time.Sleep(1 * time.Second)
		for i := range count {
			if atomic.LoadInt64(&count[i]) == 0 {
				fmt.Printf("goroutine %d did not run\n", i)
				return
			}
		}
	}
	fmt.Println("OK")
}

func GCPhys() {
	// This test ensures that heap-growth scavenging is working as intended.
	//
	// It sets up a specific scenario: it allocates two pairs of objects whose
	// sizes sum to size. One object in each pair is "small" (though must be
	// large enough to be considered a large object by the runtime) and one is
	// large. The small objects are kept while the large objects are freed,
	// creating two large unscavenged holes in the heap. The heap goal should
	// also be small as a result (so size must be at least as large as the
	// minimum heap size). We then allocate one large object, bigger than both
	// pairs of objects combined. This allocation, because it will tip
	// HeapSys-HeapReleased well above the heap goal, should trigger heap-growth
	// scavenging and scavenge most, if not all, of the large holes we created
	// earlier.
	const (
		// Size must be also large enough to be considered a large
		// object (not in any size-segregated span).
		size    = 4 << 20
		split   = 64 << 10
		objects = 2
	)
	// Set GOGC so that this test operates under consistent assumptions.
	debug.SetGCPercent(100)
	// Save objects which we want to survive, and condemn objects which we don't.
	// Note that we condemn objects in this way and release them all at once in
	// order to avoid having the GC start freeing up these objects while the loop
	// is still running and filling in the holes we intend to make.
	saved := make([][]byte, 0, objects+1)
	condemned := make([][]byte, 0, objects)
	for i := 0; i < 2*objects; i++ {
		if i%2 == 0 {
			saved = append(saved, make([]byte, split))
		} else {
			condemned = append(condemned, make([]byte, size-split))
		}
	}
	condemned = nil
	// Clean up the heap. This will free up every other object created above
	// (i.e. everything in condemned) creating holes in the heap.
	// Also, if the condemned objects are still being swept, its possible that
	// the scavenging that happens as a result of the next allocation won't see
	// the holes at all. We call runtime.GC() twice here so that when we allocate
	// our large object there's no race with sweeping.
	runtime.GC()
	runtime.GC()
	// Perform one big allocation which should also scavenge any holes.
	//
	// The heap goal will rise after this object is allocated, so it's very
	// important that we try to do all the scavenging in a single allocation
	// that exceeds the heap goal. Otherwise the rising heap goal could foil our
	// test.
	saved = append(saved, make([]byte, objects*size))
	// Clean up the heap again just to put it in a known state.
	runtime.GC()
	// heapBacked is an estimate of the amount of physical memory used by
	// this test. HeapSys is an estimate of the size of the mapped virtual
	// address space (which may or may not be backed by physical pages)
	// whereas HeapReleased is an estimate of the amount of bytes returned
	// to the OS. Their difference then roughly corresponds to the amount
	// of virtual address space that is backed by physical pages.
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	heapBacked := stats.HeapSys - stats.HeapReleased
	// If heapBacked does not exceed the heap goal by more than retainExtraPercent
	// then the scavenger is working as expected; the newly-created holes have been
	// scavenged immediately as part of the allocations which cannot fit in the holes.
	//
	// Since the runtime should scavenge the entirety of the remaining holes,
	// theoretically there should be no more free and unscavenged memory. However due
	// to other allocations that happen during this test we may still see some physical
	// memory over-use. 10% here is an arbitrary but very conservative threshold which
	// should easily account for any other allocations this test may have done.
	overuse := (float64(heapBacked) - float64(stats.HeapAlloc)) / float64(stats.HeapAlloc)
	if overuse <= 0.10 {
		fmt.Println("OK")
		return
	}
	// Physical memory utilization exceeds the threshold, so heap-growth scavenging
	// did not operate as expected.
	//
	// In the context of this test, this indicates a large amount of
	// fragmentation with physical pages that are otherwise unused but not
	// returned to the OS.
	fmt.Printf("exceeded physical memory overuse threshold of 10%%: %3.2f%%\n"+
		"(alloc: %d, goal: %d, sys: %d, rel: %d, objs: %d)\n", overuse*100,
		stats.HeapAlloc, stats.NextGC, stats.HeapSys, stats.HeapReleased, len(saved))
	runtime.KeepAlive(saved)
}

// Test that defer closure is correctly scanned when the stack is scanned.
func DeferLiveness() {
	var x [10]int
	escape(&x)
	fn := func() {
		if x[0] != 42 {
			panic("FAIL")
		}
	}
	defer fn()

	x[0] = 42
	runtime.GC()
	runtime.GC()
	runtime.GC()
}

//go:noinline
func escape(x interface{}) { sink2 = x; sink2 = nil }

var sink2 interface{}
