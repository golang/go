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

var maybeSaved []byte

func GCPhys() {
	// In this test, we construct a very specific scenario. We first
	// allocate N objects and drop half of their pointers on the floor,
	// effectively creating N/2 'holes' in our allocated arenas. We then
	// try to allocate objects twice as big. At the end, we measure the
	// physical memory overhead of large objects.
	//
	// The purpose of this test is to ensure that the GC scavenges free
	// spans eagerly to ensure high physical memory utilization even
	// during fragmentation.
	const (
		// Unfortunately, measuring actual used physical pages is
		// difficult because HeapReleased doesn't include the parts
		// of an arena that haven't yet been touched. So, we just
		// make objects and size sufficiently large such that even
		// 64 MB overhead is relatively small in the final
		// calculation.
		//
		// Currently, we target 480MiB worth of memory for our test,
		// computed as size * objects + (size*2) * (objects/2)
		// = 2 * size * objects
		//
		// Size must be also large enough to be considered a large
		// object (not in any size-segregated span).
		size    = 1 << 20
		objects = 240
	)
	// Save objects which we want to survive, and condemn objects which we don't.
	// Note that we condemn objects in this way and release them all at once in
	// order to avoid having the GC start freeing up these objects while the loop
	// is still running and filling in the holes we intend to make.
	saved := make([][]byte, 0, objects)
	condemned := make([][]byte, 0, objects/2+1)
	for i := 0; i < objects; i++ {
		// Write into a global, to prevent this from being optimized away by
		// the compiler in the future.
		maybeSaved = make([]byte, size)
		if i%2 == 0 {
			saved = append(saved, maybeSaved)
		} else {
			condemned = append(condemned, maybeSaved)
		}
	}
	condemned = nil
	// Clean up the heap. This will free up every other object created above
	// (i.e. everything in condemned) creating holes in the heap.
	runtime.GC()
	// Allocate many new objects of 2x size.
	for i := 0; i < objects/2; i++ {
		saved = append(saved, make([]byte, size*2))
	}
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
	// If heapBacked exceeds the amount of memory actually used for heap
	// allocated objects by 10% (post-GC HeapAlloc should be quite close to
	// the size of the working set), then fail.
	//
	// In the context of this test, that indicates a large amount of
	// fragmentation with physical pages that are otherwise unused but not
	// returned to the OS.
	overuse := (float64(heapBacked) - float64(stats.HeapAlloc)) / float64(stats.HeapAlloc)
	if overuse > 0.1 {
		fmt.Printf("exceeded physical memory overuse threshold of 10%%: %3.2f%%\n"+
			"(alloc: %d, sys: %d, rel: %d, objs: %d)\n", overuse*100, stats.HeapAlloc,
			stats.HeapSys, stats.HeapReleased, len(saved))
		return
	}
	fmt.Println("OK")
	runtime.KeepAlive(saved)
}
