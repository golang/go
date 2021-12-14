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
	"unsafe"
)

func init() {
	register("GCFairness", GCFairness)
	register("GCFairness2", GCFairness2)
	register("GCSys", GCSys)
	register("GCPhys", GCPhys)
	register("DeferLiveness", DeferLiveness)
	register("GCZombie", GCZombie)
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
	// It attempts to construct a sizeable "swiss cheese" heap, with many
	// allocChunk-sized holes. Then, it triggers a heap growth by trying to
	// allocate as much memory as would fit in those holes.
	//
	// The heap growth should cause a large number of those holes to be
	// returned to the OS.

	const (
		// The total amount of memory we're willing to allocate.
		allocTotal = 32 << 20

		// The page cache could hide 64 8-KiB pages from the scavenger today.
		maxPageCache = (8 << 10) * 64
	)

	// How big the allocations are needs to depend on the page size.
	// If the page size is too big and the allocations are too small,
	// they might not be aligned to the physical page size, so the scavenger
	// will gloss over them.
	pageSize := os.Getpagesize()
	var allocChunk int
	if pageSize <= 8<<10 {
		allocChunk = 64 << 10
	} else {
		allocChunk = 512 << 10
	}
	allocs := allocTotal / allocChunk

	// Set GC percent just so this test is a little more consistent in the
	// face of varying environments.
	debug.SetGCPercent(100)

	// Set GOMAXPROCS to 1 to minimize the amount of memory held in the page cache,
	// and to reduce the chance that the background scavenger gets scheduled.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	// Allocate allocTotal bytes of memory in allocChunk byte chunks.
	// Alternate between whether the chunk will be held live or will be
	// condemned to GC to create holes in the heap.
	saved := make([][]byte, allocs/2+1)
	condemned := make([][]byte, allocs/2)
	for i := 0; i < allocs; i++ {
		b := make([]byte, allocChunk)
		if i%2 == 0 {
			saved = append(saved, b)
		} else {
			condemned = append(condemned, b)
		}
	}

	// Run a GC cycle just so we're at a consistent state.
	runtime.GC()

	// Drop the only reference to all the condemned memory.
	condemned = nil

	// Clear the condemned memory.
	runtime.GC()

	// At this point, the background scavenger is likely running
	// and could pick up the work, so the next line of code doesn't
	// end up doing anything. That's fine. What's important is that
	// this test fails somewhat regularly if the runtime doesn't
	// scavenge on heap growth, and doesn't fail at all otherwise.

	// Make a large allocation that in theory could fit, but won't
	// because we turned the heap into swiss cheese.
	saved = append(saved, make([]byte, allocTotal/2))

	// heapBacked is an estimate of the amount of physical memory used by
	// this test. HeapSys is an estimate of the size of the mapped virtual
	// address space (which may or may not be backed by physical pages)
	// whereas HeapReleased is an estimate of the amount of bytes returned
	// to the OS. Their difference then roughly corresponds to the amount
	// of virtual address space that is backed by physical pages.
	//
	// heapBacked also subtracts out maxPageCache bytes of memory because
	// this is memory that may be hidden from the scavenger per-P. Since
	// GOMAXPROCS=1 here, subtracting it out once is fine.
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	heapBacked := stats.HeapSys - stats.HeapReleased - maxPageCache
	// If heapBacked does not exceed the heap goal by more than retainExtraPercent
	// then the scavenger is working as expected; the newly-created holes have been
	// scavenged immediately as part of the allocations which cannot fit in the holes.
	//
	// Since the runtime should scavenge the entirety of the remaining holes,
	// theoretically there should be no more free and unscavenged memory. However due
	// to other allocations that happen during this test we may still see some physical
	// memory over-use.
	overuse := (float64(heapBacked) - float64(stats.HeapAlloc)) / float64(stats.HeapAlloc)
	// Check against our overuse threshold, which is what the scavenger always reserves
	// to encourage allocation of memory that doesn't need to be faulted in.
	//
	// Add additional slack in case the page size is large and the scavenger
	// can't reach that memory because it doesn't constitute a complete aligned
	// physical page. Assume the worst case: a full physical page out of each
	// allocation.
	threshold := 0.1 + float64(pageSize)/float64(allocChunk)
	if overuse <= threshold {
		fmt.Println("OK")
		return
	}
	// Physical memory utilization exceeds the threshold, so heap-growth scavenging
	// did not operate as expected.
	//
	// In the context of this test, this indicates a large amount of
	// fragmentation with physical pages that are otherwise unused but not
	// returned to the OS.
	fmt.Printf("exceeded physical memory overuse threshold of %3.2f%%: %3.2f%%\n"+
		"(alloc: %d, goal: %d, sys: %d, rel: %d, objs: %d)\n", threshold*100, overuse*100,
		stats.HeapAlloc, stats.NextGC, stats.HeapSys, stats.HeapReleased, len(saved))
	runtime.KeepAlive(saved)
	runtime.KeepAlive(condemned)
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

// Test zombie object detection and reporting.
func GCZombie() {
	// Allocate several objects of unusual size (so free slots are
	// unlikely to all be re-allocated by the runtime).
	const size = 190
	const count = 8192 / size
	keep := make([]*byte, 0, (count+1)/2)
	free := make([]uintptr, 0, (count+1)/2)
	zombies := make([]*byte, 0, len(free))
	for i := 0; i < count; i++ {
		obj := make([]byte, size)
		p := &obj[0]
		if i%2 == 0 {
			keep = append(keep, p)
		} else {
			free = append(free, uintptr(unsafe.Pointer(p)))
		}
	}

	// Free the unreferenced objects.
	runtime.GC()

	// Bring the free objects back to life.
	for _, p := range free {
		zombies = append(zombies, (*byte)(unsafe.Pointer(p)))
	}

	// GC should detect the zombie objects.
	runtime.GC()
	println("failed")
	runtime.KeepAlive(keep)
	runtime.KeepAlive(zombies)
}
